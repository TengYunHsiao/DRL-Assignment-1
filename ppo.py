import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from simple_custom_taxi_env import SimpleTaxiEnv  # Custom Taxi environment
import wandb  # Logging tool for tracking experiments

# 設定設備（CUDA 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_observation(observation):
    """
    Converts the environment observation into a PyTorch tensor.

    Args:
        observation (tuple or list): The state representation from the environment.

    Returns:
        torch.Tensor: The processed state tensor.
    
    Raises:
        ValueError: If the state does not have exactly 16 elements.
    """
    try:
        state_data = observation[0]  # Extract the tuple part of the observation
        state_list = list(state_data)
    except:
        state_list = list(observation)

    if len(state_list) != 16:
        raise ValueError(f"Expected state to have 16 elements, but got {len(state_list)}")

    return torch.tensor(state_list, dtype=torch.float32, device=device)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Neural Network model for PPO.

    - The Actor network outputs action logits.
    - The Critic network outputs the estimated state value.
    """

    def __init__(self, input_dim=16, action_dim=6):
        super(ActorCriticNetwork, self).__init__()

        # Hidden layers
        self.hidden_layer_1 = nn.Linear(input_dim, 64)
        self.hidden_layer_2 = nn.Linear(64, 64)

        # Output layers
        self.actor_output = nn.Linear(64, action_dim)  # Action logits
        self.critic_output = nn.Linear(64, 1)  # State-value estimate

    def forward(self, x):
        """
        Forward pass of the Actor-Critic network.

        Args:
            x (torch.Tensor): The input tensor representing the state.

        Returns:
            logits (torch.Tensor): Action logits for policy.
            value (torch.Tensor): State-value estimate for critic.
        """
        x = F.relu(self.hidden_layer_1(x))
        x = F.relu(self.hidden_layer_2(x))

        logits = self.actor_output(x)
        value = self.critic_output(x)

        return logits, value


class PPO:
    """
    Proximal Policy Optimization (PPO) training class.
    """

    def __init__(self, env, model, lr=3e-4, gamma=0.99, clip_ratio=0.2, value_coef=0.5,
                 entropy_coef=0.01, gae_lambda=0.95, num_epochs=10, batch_size=64):
        """
        Initializes the PPO algorithm.

        Args:
            env: The environment.
            model: Actor-Critic model.
            lr: Learning rate.
            gamma: Discount factor.
            clip_ratio: Clipping factor for PPO.
            value_coef: Coefficient for value function loss.
            entropy_coef: Coefficient for entropy regularization.
            gae_lambda: Generalized Advantage Estimation lambda.
            num_epochs: Number of epochs per PPO update.
            batch_size: Minibatch size.
        """
        self.env = env
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # PPO hyperparameters
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def collect_trajectories(self, steps_per_rollout=2048):
        """
        Collects experience trajectories for PPO training.

        Args:
            steps_per_rollout: Number of environment steps before updating the model.

        Returns:
            Tuple of collected states, actions, log probabilities, advantages, returns, and mean reward.
        """
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        episode_rewards = []
        current_episode_reward = 0
        total_steps = 0

        state, _ = self.env.reset()  # Reset the environment

        # Collect data until reaching the desired number of steps
        while total_steps < steps_per_rollout:
            done = False
            while not done and total_steps < steps_per_rollout:
                state_tensor = preprocess_observation(state)

                with torch.no_grad():
                    logits, value = self.model(state_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                    log_prob = torch.log(probs[action]).item()

                next_state, reward, done, _ = self.env.step(action)

                # Store collected data
                states.append(state_tensor)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value.item())
                dones.append(done)

                # Track episode rewards
                current_episode_reward += reward
                if done or total_steps == steps_per_rollout - 1:
                    episode_rewards.append(current_episode_reward)
                    current_episode_reward = 0

                state = next_state
                total_steps += 1

        # Compute Generalized Advantage Estimation (GAE)
        advantages = []
        gae = 0
        next_value = values[-1] if not dones[-1] else 0

        for t in reversed(range(total_steps)):
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - int(dones[t]))
            advantages.insert(0, gae)
            next_value = values[t]

        returns = [advantages[t] + values[t] for t in range(total_steps)]
        mean_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0

        return states, actions, log_probs, advantages, returns, mean_reward

    def update_policy(self, states, actions, old_log_probs, advantages, returns):
        """
        Performs the PPO policy update using collected trajectories.

        Args:
            states: Collected state observations.
            actions: Taken actions.
            old_log_probs: Log probabilities from the old policy.
            advantages: Computed advantages.
            returns: Computed returns.
        """
        states = torch.stack(states).to(device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        for _ in range(self.num_epochs):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                logits, values = self.model(batch_states)
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log(probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1))
                ratios = torch.exp(log_probs - batch_old_log_probs)

                policy_loss = -torch.min(ratios * batch_advantages, 
                                         torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages).mean()
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, num_iterations=5000, steps_per_rollout=2048):
        wandb.init(project="ppo-training")  # Initialize wandb logging
        for i in tqdm(range(num_iterations)):
            states, actions, old_log_probs, advantages, returns, mean_reward = self.collect_trajectories(steps_per_rollout)
            self.update_policy(states, actions, old_log_probs, advantages, returns)
            wandb.log({"total_reward": mean_reward})


if __name__ == "__main__":
    env = SimpleTaxiEnv(fuel_limit=5000)
    model = ActorCriticNetwork().to(device)
    ppo = PPO(env, model)
    ppo.train(num_iterations=2000, steps_per_rollout=256)
    torch.save(model.state_dict(), 'trained_model.pth')

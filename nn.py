import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from simple_custom_taxi_env import SimpleTaxiEnv
import wandb

# 設定運行設備 (使用 GPU 如果可用，否則使用 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    """
    Actor-Critic 神經網絡模型，負責學習策略與評估。
    """
    def __init__(self, input_dim=16, action_dim=6):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # 第一層全連接層
        self.fc2 = nn.Linear(64, 64)  # 第二層全連接層
        self.actor = nn.Linear(64, action_dim)  # Actor 負責動作選擇
        self.critic = nn.Linear(64, 1)  # Critic 負責狀態值評估

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 激活函數 ReLU
        x = F.relu(self.fc2(x))
        logits = self.actor(x)  # Actor 的輸出
        value = self.critic(x)  # Critic 的輸出
        return logits, value

def preprocess_state(state):
    """
    將輸入的狀態轉換為 PyTorch Tensor。
    """
    try:
        state_tuple = state[0]  # 提取狀態數據
        state_list = list(state_tuple)
    except:
        state_list = list(state)
    if len(state_list) != 16:
        raise ValueError(f"Expected state to have 16 elements, got {len(state_list)}")
    return torch.tensor(state_list, dtype=torch.float32, device=device)

class PPO:
    """
    近端策略優化 (Proximal Policy Optimization, PPO) 算法實現。
    """
    def __init__(self, env, model, lr=3e-4, gamma=0.99, clip_ratio=0.2, value_coef=0.5, 
                 entropy_coef=0.01, gae_lambda=0.95, num_epochs=10, batch_size=64):
        self.gamma = gamma  # 折扣因子
        self.clip_ratio = clip_ratio  # PPO 目標函數裁剪範圍
        self.value_coef = value_coef  # Critic 的損失權重
        self.entropy_coef = entropy_coef  # 熵的權重
        self.gae_lambda = gae_lambda  # GAE 參數
        
        self.env = env
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        self.num_epochs = num_epochs  # 更新周期數
        self.batch_size = batch_size  # 批量大小

    def collect_trajectories(self, steps_per_rollout=2048):
        """
        采集環境數據，進行策略學習。
        """
        states = []
        values = []
        dones = []
        actions = []
        log_probs = []
        rewards = []
        episode_rewards = []
        current_episode_reward = 0
        total_steps = 0
        state, _ = self.env.reset()
        
        while total_steps < steps_per_rollout:
            done = False
            while not done and total_steps < steps_per_rollout:
                state_tensor = preprocess_state(state)
                with torch.no_grad():
                    logits, value = self.model(state_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                    log_prob = torch.log(probs[action]).item()
                next_state, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                values.append(value.item())
                dones.append(done)
                states.append(state_tensor)
                actions.append(action)
                log_probs.append(log_prob)
                current_episode_reward += reward
                if done or total_steps == steps_per_rollout - 1:
                    episode_rewards.append(current_episode_reward)
                    current_episode_reward = 0
                state = next_state
                total_steps += 1
        
        if total_steps > 0 and not dones[-1]:
            with torch.no_grad():
                _, v_next = self.model(preprocess_state(state))
                v_next = v_next.item()
        else:
            v_next = 0
        
        next_values = [
            0 if dones[t] else (values[t + 1] if t + 1 < total_steps else v_next)
            for t in range(total_steps)
        ]
        
        deltas = [
            rewards[t] + self.gamma * next_values[t] - values[t]
            for t in range(total_steps)
        ]
        
        advantages = []
        gae = 0
        for t in reversed(range(total_steps)):
            gae = deltas[t] + self.gamma * self.gae_lambda * gae * (1 - int(dones[t]))
            advantages.insert(0, gae)
        
        returns = [advantages[t] + values[t] for t in range(total_steps)]
        mean_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
        
        return states, actions, log_probs, advantages, returns, mean_reward
    
    def update(self, states, actions, old_log_probs, advantages, returns):
        """
        使用 PPO 損失函數來更新策略。
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
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                logits, values = self.model(batch_states)
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log(probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1))
                ratios = torch.exp(log_probs - batch_old_log_probs)


                
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

if __name__ == "__main__":
    env = SimpleTaxiEnv(fuel_limit=5000)
    model = ActorCritic().to(device)
    ppo = PPO(env, model)
    ppo.train(num_iterations=2000, steps_per_rollout=256)
    torch.save(model.state_dict(), 'trained_model.pth')

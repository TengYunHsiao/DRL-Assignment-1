import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# 设置设备（CPU 或 GPU）
device = "cpu"

def convert_observation_to_tensor(observation):
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
        # Extract the state tuple (ignoring potential dictionary elements)
        state_data = observation[0]
        state_list = list(state_data)
    except:
        state_list = list(observation)

    # Ensure the state has the correct number of elements
    if len(state_list) != 16:
        raise ValueError(f"Expected state to have 16 elements, but got {len(state_list)}")

    # Convert state to a tensor
    return torch.tensor(state_list, dtype=torch.float32, device=device)


def select_action(observation, model):
    """
    Selects an action based on the given observation.

    Args:
        observation (tuple or list): The environment state.
        model (torch.nn.Module): The trained Actor-Critic model.

    Returns:
        int: The selected action index.
    """
    try:
        # Preprocess the state into a tensor
        state_tensor = convert_observation_to_tensor(observation)

        with torch.no_grad():  # Disable gradient computation for inference
            action_logits, _ = model(state_tensor)  # Get action probabilities
            selected_action = torch.argmax(action_logits).item()  # Choose action with highest probability
    except:
        # If an error occurs, choose a random action (fallback)
        selected_action = random.choice([0, 1])

    return selected_action


# 定义 Actor-Critic 神经网络
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim=16, action_dim=6):
        """
        Initializes the Actor-Critic network.

        Args:
            input_dim (int): The number of input features (state dimension).
            action_dim (int): The number of possible actions.
        """
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
        x = F.relu(self.hidden_layer_1(x))  # Apply ReLU activation
        x = F.relu(self.hidden_layer_2(x))
        
        logits = self.actor_output(x)  # Compute action logits
        value = self.critic_output(x)  # Compute state-value estimate

        return logits, value


# 加载训练好的模型
actor_critic_model = ActorCriticNetwork(input_dim=16, action_dim=6)
actor_critic_model.load_state_dict(torch.load('trained_model.pth', map_location=device))
actor_critic_model.eval()  # 设置为评估模式

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


def get_action(obs):
    try:
        state = convert_observation_to_tensor(obs)
        with torch.no_grad():
            logits, _ = actor_critic_model(state)
            action = torch.argmax(logits).item()
    except:
        action = random.choice([0, 1])
    return action


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

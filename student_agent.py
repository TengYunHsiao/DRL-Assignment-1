import os  # 用於處理文件和目錄操作
import random  # 用於隨機選擇行為
import numpy as np  # 用於數值計算

import torch  # PyTorch 深度學習框架
import torch.nn as nn  # PyTorch 神經網絡模組
import torch.nn.functional as F  # PyTorch 的激活函數與其他功能函數

class ActorCritic(nn.Module):
    """
    Actor-Critic 模型:
    - Actor 負責策略選擇 (輸出動作機率或 logits)
    - Critic 負責評估當前狀態的價值 (Value function)
    """
    def __init__(self, input_dim=16, action_dim=6):
        """
        初始化 Actor-Critic 網絡
        :param input_dim: 狀態空間的維度 (預設為 16)
        :param action_dim: 動作空間的維度 (預設為 6)
        """
        super(ActorCritic, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)  # 第一個全連接層，將輸入映射到 64 維度
        self.layer2 = nn.Linear(64, 64)  # 第二個全連接層，保持 64 維度
        self.actor = nn.Linear(64, action_dim)  # Actor 預測動作 logits
        self.critic = nn.Linear(64, 1)  # Critic 預測當前狀態的價值 (單輸出)

    def forward(self, x):
        """
        前向傳播函數，輸入狀態 x，輸出 Actor 的 logits 和 Critic 的狀態價值。
        :param x: 輸入狀態 (tensor)
        :return: (logits, value) - Actor 的 logits 和 Critic 的值估計
        """
        x = F.relu(self.layer1(x))  # 第一層 ReLU 激活
        x = F.relu(self.layer2(x))  # 第二層 ReLU 激活
        logits = self.actor(x)  # 計算動作 logits (未歸一化)
        value = self.critic(x)  # 計算狀態價值 (Critic 輸出)
        return logits, value

# 初始化 Actor-Critic 模型，狀態維度為 16，動作維度為 6
model = ActorCritic(input_dim=16, action_dim=6)

device = "cpu"  # 設定運行設備 (這裡預設為 CPU)

# 加載已訓練的模型權重
model.load_state_dict(torch.load('trained_model.pth', map_location=device))
model.eval()  # 設置模型為評估模式 (避免 Dropout 或 BatchNorm 影響)


def preprocess_state(state):
    """
    預處理輸入狀態，確保其轉換為 PyTorch tensor。
    :param state: 觀測到的環境狀態 (通常是元組或列表)
    :return: 轉換為 PyTorch tensor 的狀態 (float32, device = CPU/GPU)
    """
    try:
        state_tuple = state[0]  # 嘗試提取元組內的狀態 (如果有)
        state_list = list(state_tuple)  # 轉換為列表
    except:
        state_list = list(state)  # 若非元組，則直接轉換為列表
    
    return torch.tensor(state_list, dtype=torch.float32, device=device)  # 轉換為 PyTorch tensor


def get_action(obs):
    """
    根據當前觀測 obs 選擇動作。
    :param obs: 環境提供的當前觀測 (狀態)
    :return: 選擇的動作 (整數類型)
    """
    try:
        state = preprocess_state(obs)  # 預處理觀測狀態
        with torch.no_grad():  # 禁用梯度計算 (推理時不需要計算梯度)
            logits, _ = model(state)  # 獲取動作 logits
            action = torch.argmax(logits).item()  # 選擇 logits 最大值對應的動作
    except:
        action = random.choice([0, 1])  # 若發生錯誤，隨機選擇動作 (0 或 1)
    
    return action
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from simple_custom_taxi_env import SimpleTaxiEnv
import wandb

import torch  
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from simple_custom_taxi_env import SimpleTaxiEnv  # 假設環境檔案為 simple_taxi_env.py
import wandb  # 用於訓練監控與紀錄

# 設定運算裝置：若 GPU 可用則使用 cuda，否則使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim=16, action_dim=6):
        """
        Actor-Critic 模型初始化
        :param state_dim: 輸入狀態的維度 (預設16)
        :param action_dim: 可選擇的動作數量 (預設6)
        """
        super(ActorCritic, self).__init__()
        # 第一個全連接層：從 state_dim 到 64 維隱藏層
        self.layer1 = nn.Linear(state_dim, 64)
        # 第二個全連接層：從 64 到 64 維隱藏層
        self.layer2 = nn.Linear(64, 64)
        # Actor 層：輸出 action_dim 個 logits，用於決定動作
        self.actor = nn.Linear(64, action_dim)
        # Critic 層：輸出一個數值，作為狀態價值估計
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        """
        前向傳播
        :param x: 輸入狀態 tensor
        :return: (action_logits, state_value)
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        action_logits = self.actor(x)
        state_value = self.critic(x)
        return action_logits, state_value

def preprocess_state(raw_state):
    """
    將環境返回的狀態轉換為神經網路所需的 tensor 格式，並檢查狀態長度
    :param raw_state: 原始狀態資料
    :return: 處理後的狀態 tensor
    """
    try:
        # 從環境返回的 tuple 中提取狀態，忽略其他資訊（例如 info dict）
        state_tuple = raw_state[0]
        state_list = list(state_tuple)
    except:
        state_list = list(raw_state)
    if len(state_list) != 16:
        raise ValueError(f"期望狀態包含 16 個元素，但收到 {len(state_list)} 個")
    return torch.tensor(state_list, dtype=torch.float32, device=device)

class PPO:
    def __init__(self, environment, actor_critic_model, learning_rate=3e-4, gamma=0.99, 
                 clip_epsilon=0.2, value_loss_coef=0.5, entropy_loss_coef=0.01, gae_lambda=0.95, 
                 num_epochs=10, batch_size=64):
        """
        PPO 代理初始化
        :param environment: 模擬環境
        :param actor_critic_model: Actor-Critic 模型
        :param learning_rate: 學習率
        :param gamma: 折扣因子
        :param clip_epsilon: PPO clipping 的比例 (epsilon)
        :param value_loss_coef: 價值損失的係數
        :param entropy_loss_coef: 熵正則項的係數，促進探索
        :param gae_lambda: GAE 的 lambda 參數
        :param num_epochs: 每次更新使用的 epoch 數
        :param batch_size: mini-batch 大小
        """
        self.env = environment
        self.model = actor_critic_model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_loss_coef = entropy_loss_coef
        self.gae_lambda = gae_lambda
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def collect_trajectories(self, rollout_steps=2048):
        """
        收集 rollout_steps 數量的經驗數據 (trajectories)
        :param rollout_steps: 每次收集的步數 (預設 2048)
        :return: (state_buffer, action_buffer, logprob_buffer, advantages, returns, mean_episode_reward)
        """
        # 初始化各種經驗數據的存儲容器
        state_buffer = []
        action_buffer = []
        logprob_buffer = []
        reward_buffer = []
        value_buffer = []
        done_buffer = []
        episode_reward_list = []
        episode_reward_sum = 0
        steps_collected = 0
        
        # 重置環境並獲取初始狀態
        current_state, _ = self.env.reset()
        
        # 開始收集經驗數據
        while steps_collected < rollout_steps:
            episode_done = False
            while not episode_done and steps_collected < rollout_steps:
                # 前處理當前狀態
                state_tensor = preprocess_state(current_state)
                # 利用模型計算動作概率與狀態價值估計 (不更新權重)
                with torch.no_grad():
                    action_logits, state_value = self.model(state_tensor)
                    action_probs = torch.softmax(action_logits, dim=-1)
                    chosen_action = torch.multinomial(action_probs, 1).item()
                    log_probability = torch.log(action_probs[chosen_action]).item()
                # 在環境中執行選定動作
                next_state, reward, episode_done, _ = self.env.step(chosen_action)
                # TODO: 可在此添加 reward shaping

                # 儲存本步的數據
                state_buffer.append(state_tensor)
                action_buffer.append(chosen_action)
                logprob_buffer.append(log_probability)
                reward_buffer.append(reward)
                value_buffer.append(state_value.item())
                done_buffer.append(episode_done)
                episode_reward_sum += reward
                # 當 episode 結束或達到最大步數時，記錄該 episode 總獎勵
                if episode_done or steps_collected == rollout_steps - 1:
                    episode_reward_list.append(episode_reward_sum)
                    episode_reward_sum = 0
                # 更新當前狀態及步數
                current_state = next_state
                steps_collected += 1
        
        # 若最後一步未結束，使用模型估計後續的狀態價值作為 bootstrap value
        if steps_collected > 0 and not done_buffer[-1]:
            with torch.no_grad():
                _, bootstrap_value = self.model(preprocess_state(current_state))
                bootstrap_value = bootstrap_value.item()
        else:
            bootstrap_value = 0
        
        # 根據 done 標記與估計值，計算每一步的下一狀態價值
        next_value_buffer = [
            0 if done_buffer[t] else (value_buffer[t + 1] if t + 1 < steps_collected else bootstrap_value)
            for t in range(steps_collected)
        ]
        
        # 計算 TD 殘差 (deltas)
        td_deltas = [
            reward_buffer[t] + self.gamma * next_value_buffer[t] - value_buffer[t]
            for t in range(steps_collected)
        ]
        
        # 使用 GAE 計算優勢值 (advantages)
        advantages = []
        gae = 0
        for t in reversed(range(steps_collected)):
            gae = td_deltas[t] + self.gamma * self.gae_lambda * gae * (1 - int(done_buffer[t]))
            advantages.insert(0, gae)
        
        # 計算每步的回報 (returns)：優勢值加上狀態價值
        returns = [advantages[t] + value_buffer[t] for t in range(steps_collected)]
        
        # 計算所有 episode 的平均獎勵
        mean_episode_reward = sum(episode_reward_list) / len(episode_reward_list) if episode_reward_list else 0
        
        return state_buffer, action_buffer, logprob_buffer, advantages, returns, mean_episode_reward
    
    def update(self, state_buffer, action_buffer, old_logprob_buffer, advantages, returns):
        """
        利用收集的經驗數據進行 PPO 更新
        """
        # 將收集到的數據轉換成 tensor 格式並搬移至指定裝置
        states_tensor = torch.stack(state_buffer).to(device)
        actions_tensor = torch.tensor(action_buffer, dtype=torch.long, device=device)
        old_logprobs_tensor = torch.tensor(old_logprob_buffer, dtype=torch.float32, device=device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)

        # 進行多個訓練 epoch
        for _ in range(self.num_epochs):
            # 隨機打亂資料索引
            permutation_indices = np.random.permutation(len(states_tensor))
            for start in range(0, len(states_tensor), self.batch_size):
                batch_indices = permutation_indices[start:start + self.batch_size]
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_logprobs = old_logprobs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # 前向傳播計算 logits 與狀態價值
                logits, values = self.model(batch_states)
                action_probs = torch.softmax(logits, dim=-1)
                # 根據 batch_actions 計算目前的 log probability
                current_logprobs = torch.log(action_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1))
                # 計算策略比率
                ratio = torch.exp(current_logprobs - batch_old_logprobs)
                # PPO 目標函數中的兩項 surrogate
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                # 計算狀態價值損失
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                # 計算熵損失，促進策略探索
                entropy_loss = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=-1).mean()
                # 總損失為策略損失、價值損失及熵損失的加權和
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_loss_coef * entropy_loss

                # 反向傳播更新模型參數
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, num_iterations=5000, rollout_steps=2048):
        """
        主訓練迴圈
        :param num_iterations: 總訓練迭代次數
        :param rollout_steps: 每次收集經驗的步數
        """
        wandb.init(project="ppo-training")  # 初始化 wandb 監控專案
        for iteration in tqdm(range(num_iterations)):
            (state_buffer, action_buffer, old_logprob_buffer, 
             advantages, returns, mean_episode_reward) = self.collect_trajectories(rollout_steps)
            self.update(state_buffer, action_buffer, old_logprob_buffer, advantages, returns)
            wandb.log({"mean_episode_reward": mean_episode_reward})

if __name__ == "__main__":
    # 初始化環境與模型
    taxi_environment = SimpleTaxiEnv(fuel_limit=5000)
    actor_critic_model = ActorCritic().to(device)
    # 建立 PPO 代理
    ppo_agent = PPO(taxi_environment, actor_critic_model)
    # 開始訓練，設定迭代次數與每次收集的步數
    ppo_agent.train(num_iterations=2000, rollout_steps=256)
    # 儲存訓練好的模型權重
    torch.save(actor_critic_model.state_dict(), 'trained_model.pth')

import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------------------------------------------------------------
# 超參數與基本設定
# --------------------------------------------------------------------------------

SAVE_PATH = "policy_net.pth"   # 用來存/讀 policy
INIT_GRID = 5
MAX_GRID  = 10
TOTAL_EPISODES = 10000
CHECK_INTERVAL = 100
SUCCESS_THRESHOLD = 90.0

LR = 1e-3
GAMMA = 0.99
HIDDEN_SIZE = 32

BASE_INPUT_DIM = 8
EXTRA_FEATURES = 1    # 只加入 manhattan_dist
NEW_INPUT_DIM = BASE_INPUT_DIM + EXTRA_FEATURES  # 8+1=9
ACTION_DIM = 6
MAX_STEPS_PER_EPISODE = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------------
# 全域變數
# --------------------------------------------------------------------------------

passenger_in_taxi = False
known_passenger_pos = None
known_destination_pos = None
visited_stations = set()

# --------------------------------------------------------------------------------
# Policy Network
# --------------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """
    兩層 MLP，輸入 9 維 state，輸出 6 維動作 logits。
    """
    def __init__(self, in_dim=NEW_INPUT_DIM, hid_dim=HIDDEN_SIZE, out_dim=ACTION_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        hidden = torch.relu(self.fc1(x))
        logits = self.fc2(hidden)
        return logits

    def get_distribution_and_logits(self, state_tensor):
        logits = self.forward(state_tensor)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        return dist, logits

policy_net = PolicyNetwork().to(DEVICE)

# --------------------------------------------------------------------------------
# Model I/O
# --------------------------------------------------------------------------------

def load_policy():
    if os.path.exists(SAVE_PATH):
        policy_net.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        policy_net.eval()
        print(f"[INFO] Loaded policy from {SAVE_PATH}")
    else:
        print("[INFO] No existing policy found; starting untrained.")

def save_policy():
    torch.save(policy_net.state_dict(), SAVE_PATH)
    print(f"[INFO] Saved policy to {SAVE_PATH}")

# --------------------------------------------------------------------------------
# State Compression
# --------------------------------------------------------------------------------

def compress_state(obs):
    """
    原本 8 維特徵:
      [0] obst_n
      [1] obst_s
      [2] obst_e
      [3] obst_w
      [4] rel_target_r
      [5] rel_target_c
      [6] can_pickup
      [7] can_dropoff

    新增 1 維:
      [8] manhattan_dist
    """
    global known_passenger_pos, known_destination_pos, visited_stations
    global passenger_in_taxi

    (taxi_r, taxi_c,
     s0_r, s0_c, s1_r, s1_c, s2_r, s2_c, s3_r, s3_c,
     obst_n, obst_s, obst_e, obst_w,
     passenger_look, destination_look) = obs

    # 紀錄四個站點
    stations = [(s0_r, s0_c), (s1_r, s1_c), (s2_r, s2_c), (s3_r, s3_c)]

    # 若在站點上 → 更新 visited_stations / passenger/destination
    if (taxi_r, taxi_c) in stations:
        visited_stations.add((taxi_r, taxi_c))
        if passenger_look and (known_passenger_pos is None):
            known_passenger_pos = (taxi_r, taxi_c)
        if destination_look and (known_destination_pos is None):
            known_destination_pos = (taxi_r, taxi_c)

    # 決定下一目標
    if (known_passenger_pos is None) or (known_destination_pos is None):
        # 還沒完全知道乘客 & 目的地 → 去沒訪過的站
        target_r, target_c = next(
            ((r, c) for (r, c) in stations if (r, c) not in visited_stations),
            stations[0]
        )
    else:
        # 已知全部: 若載客 → 目標是目的地；否則 → 目標是乘客
        if passenger_in_taxi:
            target_r, target_c = known_destination_pos
        else:
            target_r, target_c = known_passenger_pos

    rel_r = float(target_r - taxi_r)
    rel_c = float(target_c - taxi_c)
    can_pickup  = 1 if (not passenger_in_taxi) and (known_passenger_pos == (taxi_r, taxi_c)) else 0
    can_dropoff = 1 if passenger_in_taxi and (known_destination_pos == (taxi_r, taxi_c)) else 0

    # 新增: manhattan_dist
    manhattan_dist = abs(rel_r) + abs(rel_c)

    feats = [
        obst_n, obst_s, obst_e, obst_w,
        rel_r, rel_c,
        can_pickup, can_dropoff,
        manhattan_dist
    ]

    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# --------------------------------------------------------------------------------
# REINFORCE Helpers
# --------------------------------------------------------------------------------

def discount_and_norm_rewards(rewards, gamma=GAMMA):
    discounted = []
    running_sum = 0.0
    for r in reversed(rewards):
        running_sum = r + gamma * running_sum
        discounted.append(running_sum)
    discounted.reverse()

    arr = np.array(discounted, dtype=np.float32)
    return ((arr - arr.mean()) / (arr.std() + 1e-8)).tolist()

def pick_action_and_logprob(obs):
    s_vec = compress_state(obs)
    dist, _ = policy_net.get_distribution_and_logits(s_vec)
    a = dist.sample()
    lp = dist.log_prob(a)
    return a.item(), lp

# --------------------------------------------------------------------------------
# 將 get_action 改回「原版」，含更新全域狀態邏輯
# --------------------------------------------------------------------------------

def get_action(obs):
    """
    Selects an action based on the current state and updates global variables accordingly.
    This function is used during evaluation.
    """
    global passenger_in_taxi, known_passenger_pos, visited_stations, known_destination_pos

    # 1) 先記下 taxi 位置
    last_taxi_r, last_taxi_c, *_ = obs

    # 2) 若正載客，就讓 passenger_pos 跟著 taxi
    if passenger_in_taxi:
        known_passenger_pos = (last_taxi_r, last_taxi_c)

    # 3) 用 policy net 抽樣動作 (無梯度)
    s_vec = compress_state(obs)
    with torch.no_grad():
        dist, _logits = policy_net.get_distribution_and_logits(s_vec)
        action = dist.sample().item()

    # 4) 檢查 Pickup (action=4)
    if action == 4 and not passenger_in_taxi and (last_taxi_r, last_taxi_c) == known_passenger_pos:
        passenger_in_taxi = True
        known_passenger_pos = None  # 已成功接客 → 讓 passenger_pos 置 None

    # 5) 檢查 Dropoff (action=5)
    elif action == 5 and passenger_in_taxi:
        passenger_in_taxi = False
        known_passenger_pos = (last_taxi_r, last_taxi_c)  # 重新設回位置(最後一格)

    return action

# --------------------------------------------------------------------------------
# Adaptive Training
# --------------------------------------------------------------------------------

def adaptive_training(env_class, net, optimizer, max_episodes=TOTAL_EPISODES):
    global passenger_in_taxi, known_passenger_pos, known_destination_pos, visited_stations

    current_grid = INIT_GRID
    successes_log = []
    reward_log = []

    from simple_custom_taxi_env import SimpleTaxiEnv

    for ep in range(max_episodes):
        env = env_class(grid_size=current_grid, fuel_limit=5000)

        passenger_in_taxi = False
        known_passenger_pos = None
        known_destination_pos = None
        visited_stations = set()

        obs, info = env.reset()
        done = False
        step_count = 0
        ep_rewards = []
        log_probs = []
        total_ep_reward = 0.0

        while not done and step_count < MAX_STEPS_PER_EPISODE:
            last_r, last_c, *_ = obs
            action, lp = pick_action_and_logprob(obs)
            next_obs, reward, done, info = env.step(action)

            # 同樣在訓練階段更新 passenger_in_taxi:
            if action == 4 and not passenger_in_taxi and (last_r, last_c) == known_passenger_pos:
                passenger_in_taxi = True
                known_passenger_pos = (last_r, last_c)
            elif action == 5 and passenger_in_taxi:
                passenger_in_taxi = False
                known_passenger_pos = (last_r, last_c)
            if passenger_in_taxi:
                taxi_r, taxi_c, *_ = next_obs
                known_passenger_pos = (taxi_r, taxi_c)

            log_probs.append(lp)
            ep_rewards.append(reward)
            total_ep_reward += reward
            obs = next_obs
            step_count += 1

        success_flag = info.get("success", False)
        successes_log.append(1.0 if success_flag else 0.0)
        reward_log.append(total_ep_reward)

        # 計算 discounted returns + 更新
        disc_ret = discount_and_norm_rewards(ep_rewards, GAMMA)
        disc_ret_t = torch.tensor(disc_ret, dtype=torch.float32, requires_grad=True).to(DEVICE)

        optimizer.zero_grad()
        policy_loss = torch.stack([-lp * g for lp, g in zip(log_probs, disc_ret_t)]).sum()
        policy_loss.backward()
        optimizer.step()

        # 每 100 回合報告 SR & AvgR
        if (ep + 1) % CHECK_INTERVAL == 0:
            recent_succ = successes_log[-CHECK_INTERVAL:]
            sr = 100.0 * sum(recent_succ) / len(recent_succ)
            avg_r = np.mean(reward_log[-CHECK_INTERVAL:])
            print(f"[EP {ep+1}] Grid={current_grid}, AvgR={avg_r:.2f}, SR={sr:.2f}%")

            if sr >= SUCCESS_THRESHOLD and current_grid < MAX_GRID:
                current_grid += 1
                print(f"  [INFO] SR≥{SUCCESS_THRESHOLD}%. Grid => {current_grid}x{current_grid}.")

    print("[INFO] Training done.")
    print(f"[INFO] Final Grid: {current_grid}")

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    from simple_custom_taxi_env import SimpleTaxiEnv

    # 如果需要載入舊權重
    # load_policy()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    # 開始自適應訓練
    adaptive_training(SimpleTaxiEnv, policy_net, optimizer, max_episodes=TOTAL_EPISODES)

    # 儲存
    save_policy()

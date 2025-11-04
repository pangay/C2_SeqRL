#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pickle
import numpy as np
from env.env_general import SequenceRecoveryEnv

# -----------------------------
# 超参数
# -----------------------------
ENV_PATH = "train_data/train_sequences_6_18.txt"
EXPERT_PATH = "train_data/expert_trajectories_6_18.pkl"
OBS_SHAPE = (4, 18)
NUM_ACTIONS = 21 * 3
LR = 1e-4
BC_EPOCHS = 5000
LOG_INTERVAL = 100

# -----------------------------
# Policy 网络
# -----------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        input_dim = obs_shape[0] * obs_shape[1]
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.fc(x)

# -----------------------------
# 加载专家数据
# -----------------------------
with open(EXPERT_PATH, "rb") as f:
    expert_bc_dataset = pickle.load(f)

env = SequenceRecoveryEnv(file_path=ENV_PATH)

# -----------------------------
# 初始化网络与优化器
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = PolicyNetwork(OBS_SHAPE, NUM_ACTIONS).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

best_loss = float("inf")
best_model_path = "policy_bc_pretrained_best.pth"

# -----------------------------
# 在线轨迹 BC 训练
# -----------------------------
for epoch in range(1, BC_EPOCHS + 1):
    random.shuffle(expert_bc_dataset)
    epoch_losses = []

    for seq_idx, (sequence, action_list) in enumerate(expert_bc_dataset):
        obs, _ = env.reset(sequence=sequence)

        for step, (idx, val) in enumerate(action_list):
            expert_action = idx * env.num_colors + val

            # 预测当前状态动作
            obs_tensor = torch.tensor(obs[None, :], dtype=torch.float32, device=device)
            logits = policy_net(obs_tensor)

            # 计算 BC loss 并训练
            expert_tensor = torch.tensor([expert_action], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, expert_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            # 执行专家动作推进环境状态
            obs, _, _, _, _ = env.step(expert_action)

    # 每个 epoch 打印平均 loss
    avg_loss = np.mean(epoch_losses)
    if epoch % LOG_INTERVAL == 0 or epoch == 1:
        print(f"[Epoch {epoch}/{BC_EPOCHS}] Avg Loss: {avg_loss:.4f}")

    # 保存最佳模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(policy_net.state_dict(), best_model_path)
        print(f"✅ 新的最佳模型已保存，Epoch={epoch}, Loss={best_loss:.4f}")

# -----------------------------
# 保存最终模型
# -----------------------------
torch.save(policy_net.state_dict(), "policy_bc_pretrained_final.pth")
print("✅ BC 预训练完成，最终模型已保存！")
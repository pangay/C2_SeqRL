# 问题 只考虑了单边，未考虑循环
#!/usr/bin/env python3
import os
import csv
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

from stable_baselines3 import PPO   # ⚠️ 可以继续使用 MaskablePPO，也可换为普通 PPO
from env.env_general import SequenceRecoveryEnv


# ==========================================================
# 工具函数：冲突检测（原封不动）
# ==========================================================
def check_conflict(sequence, substring_length, num_colors):
    L = len(sequence)
    c = num_colors
    hist_map = defaultdict(list)
    for i in range(L - substring_length + 1):
        window = sequence[i:i + substring_length]
        hist = tuple(window.count(v) for v in range(c))
        if hist in hist_map:
            return False
        hist_map[hist].append(i)
    return True


# ==========================================================
# 回调与日志类
# ==========================================================
class BestModelCallback(BaseCallback):
    def __init__(self, save_path: str, verbose=1, window_size=1000):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_avg_reward = -np.inf
        self.episode_rewards = []
        self.window_size = window_size
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = float(info["episode"]["r"])
                self.episode_rewards.append(ep_reward)
                if len(self.episode_rewards) >= self.window_size:
                    avg_reward = np.mean(self.episode_rewards[-self.window_size:])
                    if avg_reward > self.best_avg_reward:
                        self.best_avg_reward = avg_reward
                        best_file = os.path.join(self.save_path, "best_maskable_ppo_model")
                        self.model.save(best_file)
                        if self.verbose > 0:
                            print(f"[BestModel] saved new best model, "
                                  f"avg_reward={avg_reward:.3f}, path={best_file}")
        return True


class RewardLoggerCallback(BaseCallback):
    """记录 reward 到 CSV"""
    def __init__(self, log_csv_path: str, verbose=1, log_every=10):
        super().__init__(verbose)
        self.log_csv_path = log_csv_path
        self.log_every = log_every
        self.episode_rewards = []
        os.makedirs(os.path.dirname(log_csv_path) or ".", exist_ok=True)
        with open(self.log_csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(["episode", "reward"])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = float(info["episode"]["r"])
                self.episode_rewards.append(ep_reward)
                ep_num = len(self.episode_rewards)
                if ep_num % self.log_every == 0:
                    self.logger.record("episode_reward", ep_reward)
                    with open(self.log_csv_path, 'a', newline='') as f:
                        csv.writer(f).writerow([ep_num, ep_reward])
        return True


# ==========================================================
# 自定义特征提取器（Flatten + MLP）
# ==========================================================
class FlattenExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input = int(np.prod(observation_space.shape))
        self.fc = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        x = observations.float().view(observations.size(0), -1)
        return self.fc(x)


# ==========================================================
# 主训练
# ==========================================================
if __name__ == "__main__":
    log_path = "./ppo_logs/ppo_sequence_nomask/"
    os.makedirs(log_path, exist_ok=True)
    new_logger = configure(log_path, ["stdout", "tensorboard", "csv"])
    BC_PRETRAIN_PATH = "policy_bc_pretrained_9_27.pth"

    file_path = "train_data/test_sequences_9_27.txt"
    base_env = SequenceRecoveryEnv(
        file_path=file_path,
        substring_length=9,
        max_steps=10000,
        log_file=os.path.join(log_path, "env_debug.log"),
        enable_logging=True
    )

    # ✅ 不再使用 Mask，只使用原始环境
    vec_env = DummyVecEnv([lambda: base_env])

    best_model_cb = BestModelCallback(save_path=log_path)
    reward_logger_cb = RewardLoggerCallback(
        log_csv_path=os.path.join(log_path, "episode_rewards.csv"),
        log_every=10
    )

    policy_kwargs = dict(
        features_extractor_class=FlattenExtractor,
        features_extractor_kwargs=dict(features_dim=256)
    )

    # ✅ MaskablePPO 仍可用，但 mask 功能不会启用；也可换为 PPO
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=20,
        gamma=0.99,
        verbose=1,
        tensorboard_log=log_path,
        policy_kwargs=policy_kwargs
    )

    model.set_logger(new_logger)

    checkpoint_cb = CheckpointCallback(
        save_freq=1000000,
        save_path=log_path,
        name_prefix="ppo_checkpoint"
    )

    #加载行为克隆权重
    bc_weights = torch.load(BC_PRETRAIN_PATH)
    model.policy.mlp_extractor.load_state_dict(bc_weights, strict=False)
    print("✅ BC 权重已加载到 PPO 网络")

    model.learn(
        total_timesteps=30000000,
        callback=[best_model_cb, reward_logger_cb, checkpoint_cb]
    )

    print("🎯 训练完成！")
    print(f"最优模型保存在: {os.path.join(log_path, 'best_maskable_ppo_model')}")
    print(f"Episode reward 已保存到 CSV: {os.path.join(log_path, 'episode_rewards.csv')}")
    print(f"环境运行日志: {os.path.join(log_path, 'env_debug.log')}")
    print(f"TensorBoard 可视化命令: tensorboard --logdir {log_path}")
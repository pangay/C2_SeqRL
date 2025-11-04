from env.sequence_env import SequenceRecoveryEnv
import numpy as np
import random


# 加载序列
file_path = "/Users/aoyupang/Desktop/BL 强化学习+优化/BL-Code/RL/grids-35x1-5x1-c4-conflicts1.txt"


# 构造环境
env = SequenceRecoveryEnv(file_path = file_path, max_value=3, substring_length=5, max_steps=20)


# 跑几个 episode，随机动作
n_episodes = 5
for ep in range(n_episodes):
    obs, info = env.reset()
    done, truncated = False, False
    total_reward = 0
    step = 0

    print(f"\n=== Episode {ep + 1} ===")
    while not (done or truncated):
        # 随机采样一个动作
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        step += 1
        total_reward += reward
        print(f"Step {step}, Action {action}, Reward {reward}, Done={done}, Truncated={truncated}, Info={info}")

    print(f"Episode {ep + 1} finished. Total reward: {total_reward}")
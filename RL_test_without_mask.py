#!/usr/bin/env python3
import os
import csv
import numpy as np
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.env_general import SequenceRecoveryEnv

# ----------------------------
# 日志设置
# ----------------------------
log_path = "./ppo_logs/ppo_sequence_mask_without_IR/"
os.makedirs(log_path, exist_ok=True)
logger = logging.getLogger("RLTest")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(log_path, "test_env.log"), mode='w', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(fh)


# ----------------------------
# 辅助：从文件读取所有序列
# ----------------------------
def load_sequences_from_file(file_path):
    sequences = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seq = [int(x) for x in line.split()]
            sequences.append(seq)
    return sequences


# ----------------------------
# 测试主函数
# ----------------------------
if __name__ == "__main__":
    model_path = os.path.join(log_path, "best_maskable_ppo_model")  # PPO 模型路径
    results_csv = os.path.join(log_path, "test_results.csv")

    # 创建环境
    file_path = "./train_data/test_sequences_120.txt"
    base_env = SequenceRecoveryEnv(
        file_path=file_path,
        substring_length=5,
        max_steps=100,  # 限制最大尝试次数
        log_file=os.path.join(log_path, "test_env.log"),
        enable_logging=False
    )
    vec_env = DummyVecEnv([lambda: base_env])

    # 加载 PPO 模型
    model = PPO.load(model_path, env=vec_env)

    # 读取测试序列
    sequences = load_sequences_from_file(file_path)
    total_sequences = len(sequences)
    logger.info(f"Total sequences to test: {total_sequences}")

    all_steps = []
    valid_count = 0

    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["seq_id", "orig_sequence", "steps", "valid", "final_sequence"])

        for seq_id, seq in enumerate(sequences):
            obs = base_env.reset(sequence=seq)
            if isinstance(obs, tuple):  # gym vs gymnasium
                obs = obs[0]

            obs = np.expand_dims(obs, 0)  # vectorized obs
            steps = 0
            done = False
            valid = False

            while not done and steps < 100:  # 最大尝试次数
                action, _ = model.predict(obs)
                if isinstance(action, (list, tuple, np.ndarray)):
                    action = int(np.asarray(action).reshape(-1)[0])
                else:
                    action = int(action)

                step_result = vec_env.step([action])
                if len(step_result) == 4:
                    obs, rewards, dones, infos = step_result
                    done_flag = bool(dones[0])
                elif len(step_result) == 5:
                    obs, rewards, terminateds, truncateds, infos = step_result
                    done_flag = bool(terminateds[0] or truncateds[0])
                else:
                    raise RuntimeError("Unexpected number of elements returned by vec_env.step()")

                steps += 1
                r0 = rewards[0] if isinstance(rewards, (list, tuple, np.ndarray)) else rewards
                if float(r0) > 0:
                    valid = True
                done = done_flag or steps >= 100

            all_steps.append(steps)
            if valid:
                valid_count += 1

            logger.info(f"[Seq {seq_id}] Done - steps={steps}, valid={valid}, final_seq={base_env.current_sequence}")
            writer.writerow([seq_id, " ".join(map(str, seq)), steps, int(valid), " ".join(map(str, base_env.current_sequence))])

    avg_steps = float(np.mean(all_steps)) if all_steps else 0.0
    valid_rate = 100.0 * valid_count / total_sequences if total_sequences > 0 else 0.0
    logger.info("=== PPO Test Summary ===")
    logger.info(f"total sequences: {total_sequences}")
    logger.info(f"average attempts (steps): {avg_steps:.2f}")
    logger.info(f"valid sequence rate: {valid_rate:.2f}%")
    logger.info(f"Per-sequence results saved to: {results_csv}")

    print("Done. Summary:")
    print(f" total_sequences = {total_sequences}")
    print(f" avg_steps = {avg_steps:.2f}")
    print(f" valid_rate = {valid_rate:.2f}%")
    print(f" results saved to {results_csv}")
#!/usr/bin/env python3
import os
import csv
import numpy as np
import logging
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from env.env_general import SequenceRecoveryEnv
from train_maskppo import dynamic_mask_fn

# ----------------------------
# 日志设置
# ----------------------------
log_path = "./ppo_logs/ppo_sequence_mask_3_5_IR/"
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
    model_path = os.path.join(log_path, "best_maskable_ppo_model.zip")
    results_csv = os.path.join(log_path, "test_results.csv")

    # 创建环境（注意这里的 base_env 用于 reset(sequence=...)）
    file_path = "./train_data/test_sequences_5_15_10000.txt"
    base_env = SequenceRecoveryEnv(
        file_path=file_path,
        substring_length = 5,
        max_steps = 1000,
        log_file = os.path.join(log_path, "test_env.log"),
        enable_logging = False
    )
    masked_env = ActionMasker(base_env, dynamic_mask_fn)
    vec_env = DummyVecEnv([lambda: masked_env])

    # 加载模型
    model = MaskablePPO.load(model_path, env=vec_env)

    # 读取文件中的所有序列（按行）
    sequences = load_sequences_from_file(file_path)
    total_sequences = len(sequences)
    logger.info(f"Total sequences to test: {total_sequences}")

    all_steps = []
    valid_count = 0

    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["seq_id", "orig_sequence", "steps", "valid", "final_sequence"])

        for seq_id, seq in enumerate(sequences):
            # 直接用 base_env.reset(sequence=seq) 来指定测试序列
            reset_res = base_env.reset(sequence=seq)
            # reset 可能返回 obs 或 (obs, info)
            if isinstance(reset_res, tuple):
                obs_single = reset_res[0]
            else:
                obs_single = reset_res

            # vec_env 是 vectorized，model.predict 期望 vectorized obs
            obs = np.expand_dims(obs_single, 0)  # shape (1, ...)

            logger.info(f"[Seq {seq_id}] Start testing sequence: {seq}")

            steps = 0
            done = False
            valid = False

            # 主循环：兼容不同 vec_env.step 返回格式（4-tuple 或 5-tuple）
            while not done:
                # 生成 action（注意返回可能是 array）
                action_arr, _ = model.predict(obs, action_masks=dynamic_mask_fn(base_env))
                # action_arr 可能已经是 scalar or array; 取第0个环境的 action
                if isinstance(action_arr, (list, tuple, np.ndarray)):
                    action = int(np.asarray(action_arr).reshape(-1)[0])
                else:
                    action = int(action_arr)

                step_result = vec_env.step([action])

                # 兼容 gym / gymnasium 返回格式
                if len(step_result) == 4:
                    obs, rewards, dones, infos = step_result
                    done_flag = bool(dones[0])
                elif len(step_result) == 5:
                    # obs, rewards, terminateds, truncateds, infos
                    obs, rewards, terminateds, truncateds, infos = step_result
                    done_flag = bool(terminateds[0] or truncateds[0])
                else:
                    raise RuntimeError("Unexpected number of elements returned by vec_env.step()")

                steps += 1

                # 记录当前序列（base_env 被 ActionMasker 包装，state 在 base_env）
                logger.info(f"[Seq {seq_id}] Step {steps} - Action: {action}, Sequence: {base_env.current_sequence}")

                # 判断是否为有效（例如 reward>0）
                # rewards 可能是 ndarray/list
                r0 = rewards[0] if isinstance(rewards, (list, tuple, np.ndarray)) else rewards
                if float(r0) > 0:
                    valid = True

                done = done_flag

            all_steps.append(steps)
            if valid:
                valid_count += 1

            logger.info(f"[Seq {seq_id}] Done - steps={steps}, valid={valid}, final_seq={base_env.current_sequence}")
            writer.writerow([seq_id, " ".join(map(str, seq)), steps, int(valid), " ".join(map(str, base_env.current_sequence))])

    # summary
    avg_steps = float(np.mean(all_steps)) if all_steps else 0.0
    valid_rate = 100.0 * valid_count / total_sequences if total_sequences > 0 else 0.0
    logger.info("=== RL Test Summary ===")
    logger.info(f"total sequences: {total_sequences}")
    logger.info(f"average attempts (steps): {avg_steps:.2f}")
    logger.info(f"valid sequence rate: {valid_rate:.2f}%")
    logger.info(f"Per-sequence results saved to: {results_csv}")

    print("Done. Summary:")
    print(f" total_sequences = {total_sequences}")
    print(f" avg_steps = {avg_steps:.2f}")
    print(f" valid_rate = {valid_rate:.2f}%")
    print(f" results saved to {results_csv}")
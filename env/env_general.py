import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import defaultdict
import random
import logging
import os
import csv

# 未考虑循环，是否要重新写？ / 暂时先不考虑
# -----------------------------
# 编码表加载
# -----------------------------
def load_encoding_table(file_path="encoding_table.csv"):
    """
    从 CSV 文件读取编码表，返回列表 [(x0, x1, x2), ...]
    """
    table = []
    with open(file_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            table.append((
                int(row["color0"]),
                int(row["color1"]),
                int(row["color2"]),
            ))
    return table


# -----------------------------
# 环境定义
# -----------------------------
class SequenceRecoveryEnv(gym.Env):
    def __init__(self, file_path, max_value=2, substring_length=5, max_steps=300,
                 log_file="env_debug.log", sequence_length = 15, enable_logging=True):
        super().__init__()
        self.file_path = file_path
        self.max_value = max_value
        self.num_colors = max_value + 1
        self.substring_length = substring_length
        self.max_steps = max_steps
        self.sequence_length = 3 * substring_length  # 固定长度 # 要修改 修改为读取文件中的序列长度
        self.original_sequence = None
        self.current_sequence = None
        self.step_count = 0
        self.prev_conflicts = 10000

        # 编码表
        self.encoding_table = load_encoding_table()

        # 动作空间： (位置 idx, 颜色 val)
        self.action_space = spaces.Discrete(self.sequence_length * self.num_colors)

        # 观测空间: 原始 + 当前 + 冲突 + m 行编码
        num_rows = 2 + 1 + 1 #substring_length
        self.observation_space = spaces.Box(
            low=-1,
            high=self.max_value,
            shape=(num_rows, self.sequence_length),
            dtype=np.float32
        )

        # logging
        self.enable_logging = enable_logging
        if self.enable_logging:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            self.logger = logging.getLogger("SequenceRecoveryEnv")
            self.logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            fh.setFormatter(formatter)
            if not self.logger.handlers:
                self.logger.addHandler(fh)
        else:
            self.logger = None

    # -----------------------------
    # 日志工具
    # -----------------------------
    def _log(self, msg):
        if self.enable_logging and self.logger:
            self.logger.info(msg)

    # -----------------------------
    # 冲突检测
    # -----------------------------
    def check_conflict(self, sequence):
        """检查序列是否有冲突（True = 无冲突合法）"""
        L = len(sequence)
        m = self.substring_length
        c = self.num_colors
        hist_map = defaultdict(list)

        for i in range(L - m + 1):
            window = sequence[i:i + m]
            hist = tuple(window.count(v) for v in range(c))
            if hist in hist_map:
                return False  # 有冲突
            hist_map[hist].append(i)
        return True

    def find_conflict_substrings(self, sequence):
        """查找所有冲突子串对"""
        L = len(sequence)
        m = self.substring_length
        c = self.num_colors
        hist_map = defaultdict(list)
        conflicts = []

        for i in range(L - m + 1):
            window = sequence[i:i + m]
            hist = [0] * c
            for val in window:
                hist[val] += 1
            hist_key = tuple(hist)
            if hist_key in hist_map:
                for prev_i in hist_map[hist_key]:
                    if prev_i != i:
                        conflicts.append((prev_i, sequence[prev_i:prev_i + m],
                                          i, window.copy()))
            hist_map[hist_key].append(i)
        return conflicts

    # -----------------------------
    # 环境核心接口 支持两个 1. 传入一条 sequence 2. 随机获取一条sequence
    # -----------------------------
    def reset(self, sequence=None, *, seed=None, options = None):
        """
        重置环境。
        - 如果 sequence=None，则随机从文件中选一条 sequence
        - 如果提供 sequence，则使用提供的 sequence
        """
        if seed is not None:
            random.seed(seed)

        if sequence is None:
            # 随机读取
            with open(self.file_path, "r") as f:
                lines = f.readlines()
                line = random.choice(lines).strip()
                self.original_sequence = list(map(int, line.split()))
        else:
            # 使用指定 sequence
            self.original_sequence = sequence.copy()

        # 初始化 current_sequence
        self.current_sequence = self.original_sequence.copy()
        self.step_count = 0

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        idx = action // self.num_colors
        val = action % self.num_colors

        if not (0 <= idx < self.sequence_length and 0 <= val < self.num_colors):
            raise ValueError("action out of range")

        # 执行动作
        self.current_sequence[idx] = val

        # 计算 reward
        self.step_count += 1
        is_valid = self.check_conflict(self.current_sequence)

        curr_conflicts_list = self.find_conflict_substrings(self.current_sequence)
        curr_conflicts = len(curr_conflicts_list)
        if curr_conflicts < self.prev_conflicts:
            self.prev_conflicts = curr_conflicts
        terminated = is_valid
        # truncated = (self.step_count >= self.max_steps) and not is_valid
        # reward = -1 * self.step_count - curr_conflicts
        # if is_valid:
        #     reward += 3000  # 找到合法解奖励 
        #     # 计算冲突数量
        truncated = (self.step_count >= self.max_steps) and not is_valid
        # 每步奖励 #后面就不敏感了
        reward = -1 * self.step_count - curr_conflicts

        # 若上一步冲突减少，可加小正奖
        if self.prev_conflicts > curr_conflicts:
            reward += 1.0  # 奖励减少冲突  # 不用担心劣化， 因为 如果 sequence 不是 solution 那么，conflict 起码大于 1
        # 成功奖励
        if is_valid:
            reward += 300.0  # 找到合法解

        obs = self._get_observation()
        info = {"idx": idx, "val": val, "is_valid": is_valid}

        self._log(f"Step={self.step_count}, Action=(idx={idx},val={val}), "
                  f"reward={reward:.3f}, terminated={terminated}, truncated={truncated},current_sequence = {self.current_sequence}, curr_conflicts = {curr_conflicts}")

        if terminated or truncated:
            info["episode"] = {"r": float(reward), "l": int(self.step_count)}

        return obs, float(reward), bool(terminated), bool(truncated), info

    # -----------------------------
    # 观测构造
    # -----------------------------
    def _get_observation(self):
        L = self.sequence_length
        m = self.substring_length
        C = self.num_colors

        obs = np.full((2 + 1 + m, L), -1, dtype=np.float32)

        # 第一行：原始序列
        obs[0, :] = self.original_sequence

        # 第二行：修改记录（如果没改过就是 -1，如果改过就写修改后的值）
        mod_record = np.full(L, -1, dtype=np.float32)
        cur_seq = np.array(self.current_sequence, dtype=np.int32)
        orig_seq = np.array(self.original_sequence, dtype=np.int32)

        mod_record = np.full(L, -1, dtype=np.float32)
        diff_mask = cur_seq != orig_seq
        mod_record[diff_mask] = cur_seq[diff_mask]
        obs[1, :] = mod_record
        # 冲突行
        conflict_counts = np.zeros(L, dtype=np.float32)
        conflicts = self.find_conflict_substrings(self.current_sequence)
        for i1, win1, i2, win2 in conflicts:
            for offset in range(m):
                conflict_counts[i1 + offset] += 1
                conflict_counts[i2 + offset] += 1
        obs[2, :] = conflict_counts

        # 编码 (逐窗口)
        for start in range(L - m + 1):
            window = self.current_sequence[start:start + m]
            hist = tuple(window.count(v) for v in range(C))
            if hist in self.encoding_table:
                idx_code = self.encoding_table.index(hist)
                for pos_in_window, seq_idx in enumerate(range(start, start + m)):
                    obs[3 + pos_in_window, seq_idx] = idx_code
        obs = obs[0:4,:]
        return obs
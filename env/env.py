import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import defaultdict
import random
import logging
import os
import csv
# 改进 原始 sequence。修改 sequence # 单独记录修改的位置，但不对原始的 sequence 进行修改，只是判断 是否修改成功 意思就是进行替换
def load_encoding_table(file_path="/encoding_table.csv"):
    """
    从 CSV 文件读取编码表，返回列表 [(x0,x1,x2,x3), ...]
    """
    table = []
    with open(file_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)  # 使用表头
        for row in reader:
            # 将每行的 color0~color3 转为整数元组
            table.append((
                int(row["color0"]),
                int(row["color1"]),
                int(row["color2"]),
                int(row["color3"])
            ))
    return table

class SequenceRecoveryEnv(gym.Env):
    """SequenceRecovery 环境（线性，不循环）"""
    def __init__(self, file_path, max_value=3, substring_length=10, max_steps=300,
                 log_file="env_debug.log", enable_logging=True):
        super().__init__()
        self.file_path = file_path
        self.max_value = max_value
        self.num_colors = max_value + 1
        self.substring_length = substring_length
        self.max_steps = max_steps
        self.sequence_length = 40
        self.current_seq = None
        self.original_sequence = None
        self.step_count = 0
        self.enable_logging = enable_logging
        self.initial_conflict_positions = None
        self.encoding_table = load_encoding_table("/Users/aoyupang/Desktop/BL 强化学习+优化/BL-Code/RL/env/encoding_table.csv")
        self.action_space = spaces.Discrete(self.sequence_length * self.num_colors)
        num_rows = 2 + self.substring_length  # 第一行 sequence, 第二行冲突数量, 其余为窗口编码索引
        num_cols = self.sequence_length

        self.observation_space = spaces.Box(
            low=-1,        # 窗口索引缺失位置用 -1，冲突数也>=0，所以可以用 -1
            high=self.max_value,  # sequence 最大值为 max_value，编码索引 <= len(encoding_table)
            shape=(num_rows, num_cols),
            dtype=np.float32
        )
        # logger
        if self.enable_logging:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            self.logger = logging.getLogger("SequenceRecoveryEnv")
            self.logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            if not self.logger.handlers:
                self.logger.addHandler(fh)
        else:
            self.logger = None

    def _log(self, msg):
        if self.enable_logging and self.logger:
            self.logger.info(msg)
    def check_conflict(self, sequence, substring_length, num_colors):
        """检查序列是否有冲突"""
        L = len(sequence)
        c = num_colors
        hist_map = defaultdict(list)
        conflicts_exist = False

        for i in range(L - substring_length + 1):
            window = sequence[i:i + substring_length]
            hist = tuple(window.count(v) for v in range(c))
            if hist in hist_map:
                conflicts_exist = True
                break
            hist_map[hist].append(i)
        return not conflicts_exist  # True 表示没有冲突，即合法
    
    def find_conflict_substrings_linear(self, sequence):
        """线性查找冲突子串对"""
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
                        conflicts.append((prev_i, sequence[prev_i:prev_i + m], i, window.copy()))
            hist_map[hist_key].append(i)
        return conflicts
    
    def _get_observation(self):
        """
        Observation:
        - 第一行: sequence 原始数值 (不归一化)
        - 第二行: 每个位置的冲突数量
        - 第3~(2+m)行: 每个位置所属子串的编码
            第3行: 该位置作为子串第1个位置的子串编码
            第4行: 该位置作为子串第2个位置的子串编码
            ...
            不存在的子串位置用 -1 填充
        返回 shape=(2+m, L) 的二维数组
        """
        L = self.sequence_length
        m = self.substring_length
        C = self.num_colors

        obs = np.full((2 + m, L), -1, dtype=np.float32)

        # 第一行: sequence 原始数值
        obs[0, :] = np.array(self.current_seq, dtype=np.float32)

        # 第二行: 每个位置的冲突数量
        conflict_counts = np.zeros(L, dtype=np.float32)
        conflicts = self.find_conflict_substrings_linear(self.current_seq)
        for i1, win1, i2, win2 in conflicts:
            for offset in range(m):
                conflict_counts[i1 + offset] += 1
                conflict_counts[i2 + offset] += 1
        obs[1, :] = conflict_counts

        # 第3~(2+m)行: 每个位置所属子串编码
        # 对每个子串起始位置
        for start in range(L - m + 1):
            window = self.current_seq[start:start + m]
            hist = tuple(window.count(v) for v in range(C))
            if hist in self.encoding_table:
                idx = self.encoding_table.index(hist)
                # 对应子串中每个位置填入编码
                for pos_in_window, seq_idx in enumerate(range(start, start + m)):
                    obs[2 + pos_in_window, seq_idx] = idx

        return obs

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        with open(self.file_path, "r") as f:
            lines = f.readlines()
            line = lines[random.randint(0, len(lines)-1)].strip()
            self.current_seq = list(map(int, line.split()))
            self.original_sequence = list(self.current_seq)
        self.step_count = 0
        self.min_conflicts = len(self.find_conflict_substrings_linear(self.current_seq))
        self._log(f"[reset] initial_seq={self.current_seq}, initial_conflicts={self.min_conflicts}")

        # 初始化 used_actions 并计算初始 mask
        self.used_actions = set()
        # 计算覆盖位置并存储
        conflicts = self.find_conflict_substrings_linear(self.original_sequence)
        positions = set()
        for i1, win1, i2, win2 in conflicts:
            for offset in range(self.substring_length):
                positions.add(i1 + offset)
                positions.add(i2 + offset)
        self.initial_conflict_positions = positions
        self._log(f"[reset-mask] 初始冲突覆盖位置={sorted(list(positions))}")
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self.used_actions.add(action)
        idx = action // self.num_colors
        val = action % self.num_colors
        
        if idx < 0 or idx >= self.sequence_length or val < 0 or val >= self.num_colors:
            raise ValueError("action out of range")

        # 模拟动作
        new_seq = self.current_seq.copy()
        new_seq[idx] = int(val)
        new_conflicts = len(self.find_conflict_substrings_linear(new_seq))
        is_valid = self.check_conflict(new_seq, self.substring_length, self.num_colors)

        # 仅当冲突为 0 
        if is_valid:
            self.current_seq[idx] = int(val)
            self.min_conflicts = new_conflicts
            action_taken = True
        else:
            action_taken = False  # 动作未被执行

        self.step_count += 1

        is_valid = self.check_conflict(self.current_seq, self.substring_length, self.num_colors)
        terminated = is_valid
        truncated = (self.step_count >= self.max_steps) and not is_valid

        # 奖励设计
        reward = -1.0 * self.step_count  # 每步基础惩罚
        if is_valid:
            reward += 30 # 完全合法奖励
        #if action_taken:
            #reward += 5.0  # 成功减少冲突奖励
        #reward = -1 * self.step_count # 当前冲突惩罚

        obs = self._get_observation()
        self._log(f"[step] obs={obs}")
        info = {
            "idx": idx,
            "val": val,
            "is_valid": is_valid,
            "action_taken": action_taken
        }

        if terminated or truncated:
            info["episode"] = {"r": float(reward), "l": int(self.step_count)}

        # 日志
        self._log(f"Step={self.step_count}, Action=(idx={idx},val={val}), action_taken={action_taken}, "
                f"reward={reward:.3f}, conflicts={len(self.find_conflict_substrings_linear(self.current_seq))}, "
                f"min_conflicts={self.min_conflicts}, terminated={terminated}, truncated={truncated}")

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        print(f"Step {self.step_count}: Sequence: {self.current_seq}")
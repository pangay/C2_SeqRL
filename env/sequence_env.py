import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import defaultdict
import random
import logging
import os

def check_conflict(sequence, substring_length, num_colors):
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

class SequenceRecoveryEnv(gym.Env):
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
        self.step_count = 0
        self.enable_logging = enable_logging

        # 动作空间：选择位置和赋值
        self.action_space = spaces.Discrete(self.sequence_length * (self.max_value + 1))
        # 观测空间：两行 [序列, 冲突分布]
        self.observation_space = spaces.Box(
            low=0.0,
            high=float(self.max_value),
            shape=(2, self.sequence_length),
            dtype=np.float32
        )

        # ---------- Logger ----------
        if self.enable_logging:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            self.logger = logging.getLogger("SequenceRecoveryEnv")
            self.logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        else:
            self.logger = None

    def _log(self, msg, *args):
        if self.enable_logging and self.logger:
            if args:
                self.logger.info(msg, *args)
            else:
                self.logger.info(msg)

    def find_conflict_cyclic_substrings(self, sequence):
        """寻找循环子串冲突"""
        L = len(sequence)
        m = self.substring_length
        c = self.max_value + 1
        cyclic_sequence = sequence + sequence[:m - 1]
        hist_map = defaultdict(list)
        conflicts = []

        for i in range(L):
            window = cyclic_sequence[i:i + m]
            hist = [0] * c
            for val in window:
                hist[val] += 1
            hist_key = tuple(hist)
            if hist_key in hist_map:
                for prev_i in hist_map[hist_key]:
                    if prev_i != i:
                        conflicts.append(
                            (prev_i % L, cyclic_sequence[prev_i:prev_i + m],
                             i % L, window.copy())
                        )
            hist_map[hist_key].append(i)
        return conflicts

    def _get_observation(self):
        """获取观测"""
        seq = np.array(self.current_seq, dtype=np.float32)
        sequence_length = self.sequence_length

        conflict_count = np.zeros(sequence_length, dtype=np.float32)
        conflicts = self.find_conflict_cyclic_substrings(self.current_seq)
        for i1, win1, i2, win2 in conflicts:
            for offset in range(self.substring_length):
                conflict_count[(i1 + offset) % sequence_length] += 1
                conflict_count[(i2 + offset) % sequence_length] += 1

        self._log(f"conflict_count: {conflict_count.tolist()}")
        # 只保留两行：序列 和 冲突分布
        obs = np.stack([seq, conflict_count], axis=0)
        return obs

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        idx_sequence = random.randint(0, 1000)
        with open(self.file_path, "r") as f:
            lines = f.readlines()
            line = lines[idx_sequence % len(lines)].strip()
            self.current_seq = list(map(int, line.split()))

        self.step_count = 0
        # 初始化最小冲突数
        self.min_conflicts = len(self.find_conflict_cyclic_substrings(self.current_seq))

        obs = self._get_observation()
        self._log(f"=== Episode Reset === Sequence: {self.current_seq}")
        return obs, {}

    def step(self, action):
        idx = action % self.sequence_length
        val = action // self.sequence_length

        # 执行动作
        self.current_seq[idx] = val
        self.step_count += 1

        # 当前冲突数
        current_conflicts = len(self.find_conflict_cyclic_substrings(self.current_seq))

        # 基础 reward
        is_valid = check_conflict(self.current_seq, self.substring_length, self.num_colors)
        terminated = is_valid
        truncated = (self.step_count >= self.max_steps) and not is_valid
        reward = 100.0 if is_valid else -1.0


        # 如果冲突数下降到了新低，额外奖励
        if current_conflicts < self.min_conflicts:
            reward += 5.0
            self.min_conflicts = current_conflicts

        # 每个冲突惩罚 -5
        reward -= 5 * current_conflicts

        obs = self._get_observation()

        self._log(
            f"Step={self.step_count}, Action=(idx={idx},val={val}), Reward={reward:.3f}, "
            f"Conflicts={current_conflicts}, MinConflicts={self.min_conflicts}, is_valid={is_valid}, "
            f"obs={obs.tolist()}"
        )

        info = {"idx": idx, "val": val, "is_valid": is_valid}
        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"Step {self.step_count}: Sequence: {self.current_seq}")
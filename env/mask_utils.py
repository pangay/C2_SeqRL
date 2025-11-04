import numpy as np

def dynamic_mask_fn(env):
    """动态 Mask 函数"""
    if getattr(env, "current_seq", None) is None:
        return np.ones((1, env.action_space.n), dtype=np.bool_)

    num_colors = getattr(env, "num_colors", env.max_value + 1)
    n_actions = env.action_space.n
    mask = np.zeros(n_actions, dtype=np.bool_)

    for idx in range(len(env.current_seq)):
        for val in range(num_colors):
            if val != env.current_seq[idx]:
                action_id = idx * num_colors + val
                mask[action_id] = True

    return np.asarray(mask, dtype=np.bool_).reshape(1, n_actions)
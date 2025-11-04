from collections import Counter
from itertools import combinations, product

def histogram(seq):
    c = Counter(seq)
    return (c[0], c[1], c[2], c[3])

def find_conflicts(seq, length=5):
    conflicts = []
    n = len(seq)
    substrings = []
    for i in range(n - length + 1):
        sub = seq[i:i+length]
        hist = histogram(sub)
        substrings.append((i, sub, hist))
    for (i1, s1, h1), (i2, s2, h2) in combinations(substrings, 2):
        if h1 == h2:
            conflicts.append({
                'pos1': i1,
                'substr1': s1,
                'pos2': i2,
                'substr2': s2,
                'histogram': h1,
                'length': length
            })
    return conflicts

def recover_sequence(modified_seq, max_value=3, substring_length=5, max_modify=3):
    conflicts = find_conflicts(modified_seq, length=substring_length)
    if not conflicts:
        return modified_seq, [], [], 0

    # 限定只考虑冲突区域涉及的位置
    candidate_positions = set()
    for c in conflicts:
        candidate_positions.update(range(c['pos1'], c['pos1'] + c['length']))
        candidate_positions.update(range(c['pos2'], c['pos2'] + c['length']))

    candidate_positions = sorted(candidate_positions)
    trial_count = 0

    for k in range(1, max_modify + 1):
        for positions in combinations(candidate_positions, k):
            original_values = [modified_seq[pos] for pos in positions]
            value_options = [[v for v in range(max_value + 1) if v != ov] for ov in original_values]
            for new_values in product(*value_options):
                trial_seq = modified_seq.copy()
                for pos, new_val in zip(positions, new_values):
                    trial_seq[pos] = new_val
                trial_count += 1
                if not find_conflicts(trial_seq, length=substring_length):
                    return trial_seq, positions, new_values, trial_count

    return None, None, None, trial_count

def recover_from_single_sequence(sequence, m=5, c=4, max_modify=3):
    print("🔍 尝试恢复序列：", sequence)
    recovered_seq, recovered_positions, recovered_values, trial_count = recover_sequence(
        sequence, max_value=c-1, substring_length=m, max_modify=max_modify
    )
    if recovered_seq: 
        print("✅ 恢复成功！")
        print("恢复后的序列：", recovered_seq)
        print("修改位置：", list(recovered_positions))
        print("修改值：", list(recovered_values))
        print("尝试次数：", trial_count)
    else:
        print("❌ 恢复失败。尝试次数：", trial_count)
example_sequence = [1, 2, 2, 1, 1, 2, 0, 0, 2, 0, 0, 1, 0, 0, 1]
recover_from_single_sequence(example_sequence, m=5, c=3, max_modify=15)
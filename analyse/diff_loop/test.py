import numpy as np

def calculate_significant(file_path, length, cut):
    significants = []
    with open(file_path, "r") as f:
        datas = f.readlines()
        for data in datas[1:]:
            data = data.split()
            x, y = int(int(data[1])/1000), int(int(data[3])/1000)
            q_value = float(data[6])
            if abs(x-y)>length or abs(x-y)<5 or q_value>cut:
                continue
            significants.append((x, y, q_value))
    return significants

def test(path, length):
    cut = 1
    significants = calculate_significant(path, length, cut)
    significants = np.array(significants)
    nums = significants.shape[0]
    while True:
        cut /= 10
        _q_values = significants[:, 2] < cut
        _nums = np.sum(_q_values)
        if abs(nums - _nums) / _nums < 0.001:
            return significants[np.where(_q_values)[0], :]
        nums = _nums

length = 12
hr_chr1_path = '/data1/lmh_data/MMSR_complete/analyse/GM12878/analyse/experiment_diff_loop/loop/hr/chr1_1000b/outputs/FitHiC.spline_pass1.res1000.significances.txt'
enhanced_chr1_path = '/data1/lmh_data/MMSR_complete/analyse/GM12878/analyse/experiment_diff_loop/loop/enhanced/chr1_1000b/outputs/FitHiC.spline_pass1.res1000.significances.txt'

print(len(test(hr_chr1_path, length)))
# print(len(test(enhanced_chr1_path, length)))
import os
import sys
import argparse
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from tools import merge_matrix, hic_norm, get_logger


parser = argparse.ArgumentParser(description="PCC Script")
parser.add_argument("--cell_line", default="GM12878", type=str, required=True)
opt = parser.parse_args()

cell_line = opt.cell_line
analyse_path = '/data1/lmh_data/MMSR_complete/analyse/'
experiment_path = os.path.join(analyse_path, cell_line, 'analyse', 'experiment_2')
log_path = os.path.join(experiment_path, 'exp.log')
fig_folder_path = os.path.join(experiment_path, 'fig')

logger = get_logger(log_path)
for chr in range(1, 23):
    chr_file_name = 'chr{}_1000b.npz'.format(chr)
    hr_path = os.path.join(analyse_path, cell_line, 'use_data', 'hr', chr_file_name)
    replaced_path = os.path.join(analyse_path, cell_line, 'use_data', 'replaced', chr_file_name)
    result_path = os.path.join(analyse_path, cell_line, 'validation', chr_file_name)

    hr = np.load(hr_path)['hic']
    replaced = np.load(replaced_path)['hic']
    result = np.load(result_path)['out']

    hr = hic_norm(hr)
    replaced = hic_norm(replaced)
    result[result>255] = 255
    hr = hr.astype(np.uint8)
    replaced = replaced.astype(np.uint8)
    result = result.astype(np.uint8)

    hr = merge_matrix(hr)
    hr = np.triu(hr).T + np.triu(hr)
    replaced = merge_matrix(replaced)
    replaced = np.triu(replaced).T + np.triu(replaced)
    result = merge_matrix(result)
    result = np.triu(result).T + np.triu(result)

    # 1kb PCC
    distance_all = [[0, chr]]
    dic_norm = {}
    length = 100 # 100kb
    for i in range(length):
        dic_norm[i]=[[], [], []]
    for i in range(len(distance_all)):
        for j in range(-length+1, length, 1):
            dis = distance_all[i][0] - j
            dic_norm[abs(dis)][0]+=hr.diagonal(offset=j).tolist()
            dic_norm[abs(dis)][1]+=replaced.diagonal(offset=j).tolist()
            dic_norm[abs(dis)][2]+=result.diagonal(offset=j).tolist()

    complete_y, result_y = [], []
    for i in range(length):
        complete_y.append(scipy.stats.pearsonr(dic_norm[i][0], dic_norm[i][1])[0])
        result_y.append(scipy.stats.pearsonr(dic_norm[i][0], dic_norm[i][2])[0])

    import seaborn as sns
    sns.set_style("whitegrid") 
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), sharey=True)
    axes.grid(False)
    axes.plot(list(range(100)), complete_y, color=(166/255, 206/255, 227/255), label='complete')
    axes.plot(list(range(100)), result_y, color=(51/255, 160/255, 44/255), label='enhanced')

    plt.ylim(0, 1)
    plt.xlabel('Genomic distance(1kb)')
    plt.ylabel('Pearson')
    plt.xlim(0, length)
    plt.legend()
    plt.savefig(os.path.join(fig_folder_path, 'chr{}_1000b.pdf'.format(chr)))

    logger.info('complete: {}'.format(complete_y))
    logger.info('result: {}'.format(result_y))
    for i in range(length):
        for j in range(5):
            if result_y[i+j] > complete_y[i+j]:
                break
        if j == 4:
            logger.info('交叉点: {}'.format(i))
            break
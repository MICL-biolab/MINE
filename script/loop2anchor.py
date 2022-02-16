import numpy as np

# input_file = '/home/lmh/work/repos/MINE/analyse/fig3/tss/temp/HepG2_ATAC_H3K27ac_H3K4me3_2_100_all_enhanced_sig.npy'
# output_file = '/data1/lmh_data/MINE/GM12878_ATAC_H3K27ac_H3K4me3/analyse/HepG2_ATAC_H3K27ac_H3K4me3/experiment/SDOC/HepG2_ATAC_H3K27ac_H3K4me3_2_100_enhanced_anchor.bed'

# input_file = '/home/lmh/work/repos/MINE/analyse/fig3/represses/tss/temp/HepG2_H3K9me3_H3K27me3_2_100_all_enhanced_sig.npy'
# output_file = '/data1/lmh_data/MINE/GM12878_H3K9me3_H3K27me3/analyse/HepG2_H3K9me3_H3K27me3/experiment/SDOC/HepG2_H3K9me3_H3K27me3_2_100_enhanced_anchor.bed'

input_file = '/home/lmh/work/repos/MINE/analyse/fig3/represses/tss/temp/HeLa_H3K9me3_H3K27me3_2_100_all_enhanced_sig.npy'
output_file = '/data1/lmh_data/MINE/GM12878_H3K9me3_H3K27me3/analyse/HeLa_H3K9me3_H3K27me3/experiment/SDOC/HeLa_H3K9me3_H3K27me3_2_100_enhanced_anchor.bed'

active_sigs = np.load(input_file, allow_pickle=True).item()
all_active_sig = set()
for chr in range(1, 23):
    for active_sig in active_sigs[chr]:
        bin1, bin2 = int(active_sig[0]*1000), int(active_sig[1]*1000)
        str_chr = 'chr{}'.format(chr)
        all_active_sig.add((str_chr, bin1, bin1+1000))
        all_active_sig.add((str_chr, bin2, bin2+1000))

with open(output_file, "w") as f:
    for active_sig in all_active_sig:
        f.write('{}\t{}\t{}\n'.format(active_sig[0], active_sig[1], active_sig[2]))
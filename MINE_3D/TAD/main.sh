hic_path='/data1/lmh_data/MINE/source/HeLa/4DNFICEGAHRC.hic'
analyse_anchor_path='/data1/lmh_data/MINE/GM12878_ATAC_H3K27ac_H3K4me3/analyse/HeLa_ATAC_H3K27ac_H3K4me3/experiment/SDOC/HeLa_ATAC_H3K27ac_H3K4me3_2_100_enhanced_anchor.bed'
out_path='/data1/lmh_data/MINE/GM12878_ATAC_H3K27ac_H3K4me3/analyse/HeLa_ATAC_H3K27ac_H3K4me3/experiment/'
resolution=10000
cell_line='HeLa'

juicer_tools_path='/home/lmh/work/repos/tools/juicer_tools_1.22.01.jar'
HiCDB_path='/home/lmh/work/repos/tools/HiCDB/'
SDOC_path='/home/lmh/work/repos/tools/SDOC/'

mkdir -p $out_path

# HiCDB
HiCDB_output=$out_path'HiCDB/txt/'
mkdir -p $HiCDB_output
for ((i=1; i<=22; i++))
do
        java -jar $juicer_tools_path dump observed VC $hic_path $i $i BP $resolution $HiCDB_output'/chr'$i'.matrix'
done
matlab -r "addpath(genpath('$HiCDB_path'));HiCDB({'$HiCDB_output'}, $resolution, 'hg38', 'ref', 'hg38');exit;"

# SDOC
SDOC_work_path=$out_path'SDOC/'
mkdir -p $SDOC_work_path'txt'
mkdir -p $SDOC_work_path'TADs'
for ((i=1; i<=22; i++))
do
        java -jar $juicer_tools_path dump observed VC -d $hic_path $i $i BP $resolution $SDOC_work_path'txt/'$cell_line'_chr'$i
done
python ./CDB2TADS.py -i $HiCDB_output'CDB.txt' -o $SDOC_work_path'TADs' -c $cell_line
cd $SDOC_path
python ./get_SDOC.py $cell_line $SDOC_work_path'TADs' $SDOC_work_path'txt' $resolution $SDOC_work_path'result' $analyse_anchor_path
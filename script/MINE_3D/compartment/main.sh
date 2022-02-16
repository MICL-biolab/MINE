hic_path='/data1/lmh_data/MINE/source/HeLa/4DNFICEGAHRC.hic'
analyse_anchor_path='/data1/lmh_data/MINE/GM12878_ATAC_H3K27ac_H3K4me3/analyse/HeLa_ATAC_H3K27ac_H3K4me3/experiment/SDOC/HeLa_ATAC_H3K27ac_H3K4me3_2_100_enhanced_anchor.bed'
out_path='/data1/lmh_data/MINE/GM12878_ATAC_H3K27ac_H3K4me3/analyse/HeLa_ATAC_H3K27ac_H3K4me3/experiment/'
TAD_resolution=5000000
reconstruction_resolution=50000
cell_line='HeLa'

juicer_tools_path='/home/lmh/work/repos/tools/juicer_tools_1.22.01.jar'
SDOC_path='/home/lmh/work/repos/tools/SDOC/'

mkdir -p $out_path

# HiCDB
eigen_output=$out_path'eigen/txt/'
mkdir -p $eigen_output
for ((i=1; i<=22; i++))
do
        java -jar $juicer_tools_path eigenvector VC $hic_path $i BP $TAD_resolution $eigen_output'/chr'$i'.eigen.txt'
done

# SDOC
SDOC_work_path=$out_path'SDOC_compartment/'
mkdir -p $SDOC_work_path'txt'
mkdir -p $SDOC_work_path'TADs'
for ((i=1; i<=22; i++))
do
        java -jar $juicer_tools_path dump observed VC -d $hic_path $i $i BP $reconstruction_resolution $SDOC_work_path'txt/'$cell_line'_chr'$i
done
python ./eigen2TADS.py -i $eigen_output -o $SDOC_work_path'TADs' -c $cell_line -r $TAD_resolution
cd $SDOC_path
python ./get_SDOC.py $cell_line $SDOC_work_path'TADs' $SDOC_work_path'txt' $reconstruction_resolution $SDOC_work_path'result' $analyse_anchor_path
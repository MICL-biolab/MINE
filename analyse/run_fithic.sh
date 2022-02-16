folder_path='/data1/lmh_data/MINE/GM12878_H3K9me3_H3K27me3/analyse/HeLa_H3K9me3_H3K27me3/experiment/loop/enhanced/'
for ((i=1; i<=22; i++))
do
        cd $folder_path'chr'$i'_1000b'
        pwd

        rm -rf outputs_2_100
        fithic -f fragments.gz -i interactions.gz -o outputs_2_100 -r 1000 -U 100000 -L 2000
        gzip -d outputs_2_100/FitHiC.spline_pass1.res1000.significances.txt.gz
done
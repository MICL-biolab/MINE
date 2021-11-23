folder_path='/data1/lmh_data/MINE/GM12878_ATAC_H3K27ac_H3K4me3/analyse/GM12878_ATAC_H3K27ac_H3K4me3/experiment/loop/hr/'
for ((i=19; i<=22; i++))
do
        cd $folder_path'chr'$i'_1000b'
        pwd

        rm -rf outputs_2_14
        fithic -f fragments.gz -i interactions.gz -o outputs_2_14 -r 1000 -U 14000 -L 2000
        gzip -d outputs_2_14/FitHiC.spline_pass1.res1000.significances.txt.gz

        # rm -rf outputs_12_500
        # fithic -f fragments.gz -i interactions.gz -o outputs_12_500 -r 1000 -U 500000 -L 12000
        # gzip -d outputs_12_500/FitHiC.spline_pass1.res1000.significances.txt.gz

        rm -rf outputs_14_50
        fithic -f fragments.gz -i interactions.gz -o outputs_14_50 -r 1000 -U 50000 -L 14000
        gzip -d outputs_14_50/FitHiC.spline_pass1.res1000.significances.txt.gz
        
        # rm -rf outputs_50_100
        # fithic -f fragments.gz -i interactions.gz -o outputs_50_100 -r 1000 -U 100000 -L 50000
        # gzip -d outputs_50_100/FitHiC.spline_pass1.res1000.significances.txt.gz

        # rm -rf outputs_2_100
        # fithic -f fragments.gz -i interactions.gz -o outputs_2_100 -r 1000 -U 100000 -L 2000
        # gzip -d outputs_2_100/FitHiC.spline_pass1.res1000.significances.txt.gz

        # rm -rf outputs_2_300
        # fithic -f fragments.gz -i interactions.gz -o outputs_2_300 -r 1000 -U 300000 -L 2000
        # gzip -d outputs_2_300/FitHiC.spline_pass1.res1000.significances.txt.gz

        # rm -rf outputs_2_500
        # fithic -f fragments.gz -i interactions.gz -o outputs_2_500 -r 1000 -U 500000 -L 2000
        # gzip -d outputs_2_500/FitHiC.spline_pass1.res1000.significances.txt.gz
done
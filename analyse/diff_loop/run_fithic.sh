folder_path='/data1/lmh_data/MMSR_complete/analyse/GM12878/analyse/experiment_diff_loop/loop/test/enhanced/'
for ((i=1; i<=22; i++))
do
        cd $folder_path'chr'$i'_1000b'
        pwd

        # fithic -f fragments.gz -i interactions.gz -o outputs -r 1000 -U 12000 -L 2000

        # rm -rf outputs_12_500
        # fithic -f fragments.gz -i interactions.gz -o outputs_12_500 -r 1000 -U 500000 -L 12000
        # gzip -d outputs_12_500/FitHiC.spline_pass1.res1000.significances.txt.gz

        # rm -rf outputs_2_100
        # fithic -f fragments.gz -i interactions.gz -o outputs_2_100 -r 1000 -U 100000 -L 2000
        # gzip -d outputs_2_100/FitHiC.spline_pass1.res1000.significances.txt.gz

        # rm -rf outputs_12_50
        # fithic -f fragments.gz -i interactions.gz -o outputs_12_50 -r 1000 -U 50000 -L 12000
        # gzip -d outputs_12_50/FitHiC.spline_pass1.res1000.significances.txt.gz
        
        # rm -rf outputs_50_100
        # fithic -f fragments.gz -i interactions.gz -o outputs_50_100 -r 1000 -U 100000 -L 50000
        # gzip -d outputs_50_100/FitHiC.spline_pass1.res1000.significances.txt.gz

        intersections=(18 14 14 9 15 7 10 16 14 11 14 8 16 10 19 18 12 11 4 13 14 31)
        rm -rf outputs
        fithic -f fragments.gz -i interactions.gz -o outputs -r 1000 -U ${intersections[i-1]}000 -L 2000
        gzip -d outputs/FitHiC.spline_pass1.res1000.significances.txt.gz
done
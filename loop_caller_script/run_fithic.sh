folder_path='/data1/lmh_data/MINE/additional/robustness/4DNFI7J8BQ4P/experiment/loop/enhanced/'
for ((i=1; i<=22; i++))
do
        cd $folder_path'chr'$i'_1000b'
        pwd

        rm -rf outputs_2_100
        fithic -f fragments.gz -i interactions.gz -o outputs_2_100 -r 1000 -U 100000 -L 2000
        gzip -d outputs_2_100/FitHiC.spline_pass1.res1000.significances.txt.gz
done
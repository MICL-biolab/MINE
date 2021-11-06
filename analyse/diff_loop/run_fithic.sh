folder_path='/data1/lmh_data/MMSR_complete/analyse/GM12878/analyse/experiment_diff_loop/loop/hr/'
for ((i=1; i<=22; i++))
do
        cd $folder_path'chr'$i'_1000b'
        pwd
        rm -rf outputs
        fithic -f fragments.gz -i interactions.gz -o outputs -r 1000 -U 12000 -L 2000
        gzip -d outputs/FitHiC.spline_pass1.res1000.significances.txt.gz
done
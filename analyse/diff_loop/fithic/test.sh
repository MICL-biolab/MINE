hic_path='/data1/lmh_data/MMSR_complete/source/hic/4DNFI1UEG1HD.hic'
juicer_path='/home/lmh/work/tools/juicer_tools_1.22.01.jar'
for ((i=1; i<=22; i++))
do
        txt_path='/data1/lmh_data/MMSR_complete/analyse/GM12878/analyse/experiment_diff_loop/loop/new_hr/txt/chr'$i'_1000b.txt'
        java -jar $juicer_path dump observed NONE $hic_path $i $i BP 1000 $txt_path
        folder='/data1/lmh_data/MMSR_complete/analyse/GM12878/analyse/experiment_diff_loop/loop/new_hr/fithic/chr'$i'_1000b'
        mkdir -p $folder
        OUT=$folder'/interactions.gz'
        cat $txt_path | awk -v chr1=$i -v chr2=$i '{ printf "%s\t%s\t%s\t%s\t%s\n", chr1,$1,chr2,$2,$3}' | gzip > $OUT
done
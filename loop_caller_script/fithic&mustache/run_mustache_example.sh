folder_path='/data1/lmh_data/MINE/additional/robustness/4DNFI7J8BQ4P/experiment/loop/enhanced/'
for ((i=1; i<=22; i++))
do
        cd $folder_path'chr'$i'_1000b'
        pwd

        cp interactions.gz tmp.gz
        gunzip tmp.gz
        mustache -f ./tmp -r 1kb -ch $i -o ./mustache_output.tsv -pt 0.5 -p 40
        rm tmp
done
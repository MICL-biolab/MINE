for((i=10;i<=20;i++));
do   
    # echo $(expr $i \* 3 + 1);
    python correlation.py -b /data1/lmh_data/MMSR_complete/analyse/hr_$i.npz -a /data1/lmh_data/MMSR_complete/analyse/validation_$i.npz
done  

# python correlation.py -b /data1/lmh_data/MMSR_complete/analyse/hr_19.npz -a /data1/lmh_data/MMSR_complete/analyse/validation_19.npz
### USE
#### Prepare the environment
1. Need to prepare juicer_tools and cuda10.1 environment in advance
2. conda create -n MMSR python=3.6
3. pip install -r requirements.txt
#### Generating data for training
##### Hi-C
1. .hic -> .txt(by juicer): java -jar /path/to/juicer_tools.jar dump observed VC /path/to/hic int int BP 1000 /path/to/txt
2. .txt -> .npz: python txt2npy.py -i /folder/to/txt -o /folder/to/npz -r 1000
3. Generate training data(Hi-C): python generate_train_data.py -i /folder/to/npz -o /folder/to/train -s 400 -f 2000
##### Epi
1. .bigWig -> .npz: python analysis_epi.py -i /path/to/bigWig -o /folder/to/epi -r 1000
2. Combine multiple epi data to generate correlation matrix: python epi_concat.py -i /folder/to/epis -o /folder/to/train/epi -r 1000 -s 400 -f 2000
#### Train
1. train: CUDA_VISIBLE_DEVICES=1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=5 train_model.py -i /folder/to/train -o /folder/to/checkpoint
2. validate: python validate.py --train_folder /folder/to/train --model /path/to/model --results /folder/to/result
### Predict & Analyse
1. We trained a model and put it in the data folder, you can use the validate.py in the data folder to predict the data
2. The analysis processing steps and results are in the analyse folder

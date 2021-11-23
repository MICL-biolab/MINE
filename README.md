# USE
## Prepare the environment
1. Need to prepare juicer_tools and cuda10.1 environment in advance
2. conda create -n MINE python=3.6
3. pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
4. pip install -r requirements.txt
## Generating data for training
### Hi-C
1. .hic -> .txt(by juicer)
```
java -jar /path/to/juicer_tools.jar dump observed VC /path/to/hic int int BP 1000 /path/to/txt
```
2. .txt -> .npz
```
python txt2npy.py -i /folder/to/txt -o /folder/to/npz -r 1000
```
3. Generate training data(Hi-C)
```
python generate_train_data.py -i /folder/to/npz -o /folder/to/train -s 400 -f 2000
```
### Epi
1. .bigWig -> .npz
```
python analysis_epi.py -i /path/to/bigWig -o /folder/to/epi -r 1000
```
2. Combine multiple epi data to generate correlation matrix
```
python epi_concat.py -i /folder/to/epis -o /folder/to/train/epi -r 1000 -s 400 -f 2000
```
### Annotation
1. .bigBed -> .npz
```
python generate_train_annotation_data.py -i /path/to/bigBed -o /folder/to/train/annotation
```
## Train
1. train
```
python train_model.py -i /folder/to/train -o /folder/to/checkpoint
```
2. validate
```
python validate.py --train_folder /folder/to/train --model /path/to/model --results /folder/to/result
```
# Predict & Analyse
1. We trained a model and put it in the data folder, you can use the validate.py in the data folder to predict the data
2. The analysis processing steps and results are in the analyse folder

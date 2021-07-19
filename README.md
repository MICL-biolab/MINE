### use
#### 准备环境
1. 提前准备 juicer_tools 工具 & cuda10.1 环境
2. 创建环境: conda create -n MMSR python=3.6
#### 数据处理
1. 通过.hic文件生成.txt(by juicer): java -jar /path/to/juicer_tools.jar dump observed VC /path/to/hic int int BP 1000 /path/to/txt
2. 通过.txt文件生成.npz: python txt2npy.py -i /folder/to/txt -o /folder/to/npz -r 1000
3. 生成hic训练数据: python generate_train_data.py -i /folder/to/npz -o /folder/to/train -s 400 -f 2000
4. 解析.bigWig文件生成epi数据: python analysis_epi.py -i /path/to/bigWig -o /folder/to/epi -r 1000
5. 将多个epi数据进行结合，生成相关性矩阵: python epi_concat.py -i /folder/to/epis -o /folder/to/train/epi -r 1000 -s 400 -f 2000
#### 数据训练
1. 训练: python train_model.py -i /folder/to/train -o /folder/to/checkpoint
2. 预测: python validate.py --train_folder /folder/to/train --model /path/to/model --results /folder/to/result
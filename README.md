# MMSR(Multi-Modal-Super-Resolution)
### file tree
.
├── dataset.py  数据集/验证集加载
├── generate_hic.py  生成hic矩阵
├── generate_train_data.py  生成训练所用数据
├── hicplus_model.py
├── model.py
├── multi_modal_model.py
├── README.md
├── train_model.py  训练模型
└── validate.py  验证模型
### use
1. python generate_hic.py -i /together/micl/liminghong/hic_data/unpack/GM12878_combined/1kb_resolution_intrachromosomal -o /together/micl/liminghong/hic_data/npz/test/GM12878_combined -hr 1kb
2. python generate_train_data.py -i /together/micl/liminghong/hic_data/npz/GM12878_combined -o /together/micl/liminghong/hic_data/train
3. python train_model.py
4. python validate.py
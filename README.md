# Prompt-Adapter: Prompt based Training-free Adaption of CLIP for Few-shot Classification


### Step 1: Installation
Create a conda environment and install dependencies:
```bash
conda create -y -n torch180 python=3.8
conda activate torch180
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

pip install -r requirements.txt

```

### Step 2: Dataset
All datasets are set in A5000 server. You just need to create a soft link:
```bash
cd prompt_tipadapter
ln -s /data/jason/data/coopdata data/
```
### Step 2-2: Prompt Download
The prompt is in the A5000 server, just copy it under the file 'prompt_tipadapter'
```bash
/home/jason/mvlpt/prompt_tensor_init.tar
```



### Step 3: Change  Configs

The running configurations can be modified in `configs/dataset.yaml`, including shot numbers, visual encoders, and hyperparamters. 

For our evauation of 1shot, 2shots, 4shots, 8shots, 16shots, YOU NEED to change the shots first and then running the follow script.

Note that the default `load_cache` and `load_pre_feat` are `False` for the first running, which will store the cache model and val/test features in `configs/dataset/`. For later running, they can be set as `True` for faster hyperparamters tuning.


### Step 4: Running
For ImageNet dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --config configs/imagenet.yaml
```
For other 10 datasets:
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/dataset.yaml

For example:
CUDA_VISIBLE_DEVICES=2 python main.py --config configs/oxford_pets.yaml
CUDA_VISIBLE_DEVICES=2 python main.py --config configs/stanford_cars.yaml
CUDA_VISIBLE_DEVICES=3 python main.py --config configs/caltech101.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/oxford_flowers.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/fgvc.yaml
CUDA_VISIBLE_DEVICES=3 python main.py --config configs/food101.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/sun397.yaml
CUDA_VISIBLE_DEVICES=2 python main.py --config configs/ucf101.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/dtd.yaml
CUDA_VISIBLE_DEVICES=2 python main.py --config configs/eurosat.yaml
```
### Step 5: Draw Pictures
```bash
python draw_curves.py
```



## Acknowledgement
This repo benefits from [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/Dassl.pytorch) and [CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter). Thanks for their wonderful works.


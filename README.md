# Prompt Tuning based Adapter for Vision-Language Model Adaption


### Step 1: Installation
Create a conda environment and install dependencies:
```bash
conda create -y -n torch180 python=3.8
conda activate torch180
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

pip install -r requirements.txt

```

### Step 2: Dataset
Follow [DATASETS.md](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/DATASET.md) to install the datasets used in the paper. Or run the following script(11 datasets, include ImageNet): 
```bash
bash scripts/data.sh
```


### Step 3: Prompt Download
Download the pretrained prompt from the [link](https://drive.google.com/file/d/1bfCXO9iE3ys3__xnOrC6bHAVXVcFXkyW/view?usp=share_link)
And decompress it under the folder `prompt_adapter/prompt_tensor_init`. 
```bash
tar -xvf prompt_tensor_init.tar
```


### Step 4: Change  Configs

The running configurations can be modified in `configs/dataset.yaml`, including shot numbers, visual encoders, and hyperparamters. 

For our evauation of 1shot, 2shots, 4shots, 8shots, 16shots, 20shots, YOU NEED to change the shots first and then running the follow script.

Note that the default `load_cache` and `load_pre_feat` are `False` for the first running, which will store the cache model and val/test features in `configs/dataset/`. For later running, they can be set as `True` for faster hyperparamters tuning.


### Step 5: Running
For ImageNet dataset:
```bash
python main_imagenet.py --config configs/imagenet.yaml
```
For other 10 datasets:
```bash
python main.py --config configs/oxford_pets.yaml
```



## Acknowledgement
This repo benefits from [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter) and [CoOp](https://github.com/KaiyangZhou/Dassl.pytorch). Thanks for their wonderful works.


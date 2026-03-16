# SS-BiMamba: Spectral-Spatial Bi-Directional Mamba for HSI Classification

------



## Getting Started

------

### 1 Environment Setup

------

create conda environment.

```powershell
conda create --name mamba python=3.10 -y
conda activate mamba
conda install cudatoolkit=11.8 -c nvidia
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging
git clone https://github.com/hustvl/Vim.git
cd Vim
pip install -r vim/vim_requirements.txt
pip install causal_conv1d==1.2.1
# pip install causal_conv1d==1.2.1 --no-build-isolation
pip install mamba-ssm==2.0.4
```

### 2 Download Dataset

------

Our work is evaluated on three pulic hyperspectral dataset:

```
paviaU, Indian_pines, Houston2013
```

### 3 Model Training and Inference

------

To train Hypermamba for classification on those datasets, you should change`config.data` for different dataset in code file`main.py ` use the following commands for model training.

```python
python main.py
```


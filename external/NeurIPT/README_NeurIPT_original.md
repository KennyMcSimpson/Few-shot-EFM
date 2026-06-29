<div align="center">

# NeurIPT


_Foundation Model for Neural Interfaces_


<!-- [![Paper](https://img.shields.io/badge/arXiv-2412.07236-red)](https://arxiv.org/abs/2412.07236)
[![Paper](https://img.shields.io/badge/Paper-ICLR-008B8B)](https://openreview.net/forum?id=NPNUHgHF2w)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/weighting666/CBraMod)
![GitHub Repo stars](https://img.shields.io/github/stars/wjq-learning/CBraMod) -->

</div>


<div align="center">
<img src="figure/NeurIPT_logo.png" style="width: 15%;" />
</div>

<p align="center">
    🔍&nbsp;<a href="#-about">About</a>
    | 🔨&nbsp;<a href="#-requirements">Requirements</a>
    | 💡&nbsp;<a href="#-how-to-pre-processing">How to Pre-processing</a>
    | 🚀&nbsp;<a href="#-how-to-pretrain">How to Pretrain</a>
    | 🚢&nbsp;<a href="#-how-to-finetune">How to Finetune</a>
    | 🤖&nbsp;<a href="#-custom-modify">Custom Modify</a>
    | ⭐&nbsp;<a href="#-acknowledgement">Acknowledgement</a>
</p>


<!-- 🔥 NEWS: The paper "_CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding_" has been accepted by ICLR 2025! -->

## 🔍 About
We propose **NeurIPT**, a foundation model advancing scalable and generalizable EEG decoding across diverse BCI tasks:
<!-- The preprint version of our paper is available at [arXiv](https://arxiv.org/abs/2412.07236). 
The camera-ready version of the paper will be available at [OpenReview](https://openreview.net/forum?id=NPNUHgHF2w). -->
<div align="center">
<img src="figure/main1.png" style="width:100%;" />
</div>

**Figure1.** Overview of our **NeurIPT**, which comprises **A**mplitude-**A**ware **M**asked **P**retraining (**AAMP**), **3D Electrode Embedding**, **P**rogressive **M**ixture-**o**f-**E**xperts (**PMoE**), and **I**ntra-**I**nter **L**obe **P**ooling (**IILP**) for fine-tuning.


<div align="center">
<img src="figure/main2.png" style="width:100%;" />
</div>

**Figure2.** **Left:** IILP leverages regional brain features during fine-tuning. **Right:** Visualization of attention scores from the temporal attention module and analysis of Pearson correlation between class logits and channel perturbation using Gaussian multiplicative noise.


## 🔨 Requirements
- [Python](https://www.python.org/downloads/) == 3.9.19
- [PyTorch](https://pytorch.org/get-started/locally/) == 2.3.0
- [eimops](https://einops.rocks/) == 0.8.0
- [mne](https://mne.tools/) == 1.4.2

❗️ Please strictly follow to the *mne* version, as other versions are not compatible with the *cnt* file for the SEED dataset.

Install other requirements:
```commandline
pip install -r requirements.txt
``` 

## 💡 How to Pre-processing
Please Followed by:
```commandline
./preprocessing/pretrain/
./preprocessing/downtream/
```

## 🚀  How to Pretrain
You can pretrain NeurIPT on our pretraining dataset or your custom pretraining dataset using the following code:
```commandline
bash ./scripts/run_pretrained.sh
```
<!-- We have released a pretrained checkpoint on [Hugginface🤗](https://huggingface.co/weighting666/CBraMod). -->

## 🚢  How to Finetune
You can finetune NeurIPT on our selected downstream datasets using the following code:
```commandline
bash ./scripts/run_DATASET.sh
```
Change the **DATASET** to the corresponding names: 

["BCICIV-2A", "MentalArithmetic", "Mumtaz", "P300", "SEED-V", "Sleep-EDFx", "TUAB", "TUEV"]


## 🤖 Custom Modify

| Parameter name | Description of parameter |
| --- | --- |
| data           | The dataset name                                             |
| data_root_path      | The root path of the data file   |
| checkpoints      | Location to store model checkpoints        |
| in_len | Length of input sequence |
| out_len | Length of output sequence |
| data_dim | Number of channels of the BCI data, i.e. $D$ in the paper |
| d_model | Dimension of hidden states, i.e. $d_{model}$ in the paper (defaults to 768) |
| d_ff | Dimension of MLP in MSA (defaults to 768) |
| n_heads | Num of heads in MSA (defaults to 8) |
| e_layers | Num of encoder layers, i.e. $N$ in the paper (defaults to 6) |
| dropout | The probability of dropout (defaults to 0.1) |
| num_workers | The num_works of Data loader (defaults to 8) |
| batch_size | The batch size for training and testing |
| train_epochs | Train epochs (defaults to 20) |
| patience | Early stopping patience (defaults to 3) |
| learning_rate | The initial learning rate for the optimizer (defaults to 4e-3) |
| part | The IILP part (defaults: functional, options: Hemispheres, Sagittal, Coronal) |
| freeze_encoder | Only adjust linear cls head at beginning (defaults to False) |
| freeze_epochs | Epochs to freeze the encoder (defaults to 3) |
| use_wandb | Whether to use wandb to log or not (defaults to True) |
| wandb_name | Name of the wandb item  |
| use_amp | Use automatic mixed precision (defaults to False) |
| amp_type | choose the dtype of automatic mixed precision (fp16 or bf16) |
| flash_attn | Enable FlashAttention2 if specified (Mutually exclusive with mem_efficient) |
| mem_efficient | Enable Memory-Efficient Attention if specified (Mutually exclusive with flash_atten) |
| DDP | Use Distributed Data Parallel (DDP) Framework (Mutually exclusive with FSDP) |
| FSDP | Use Fully Sharded Data Parallel (FSDP) Framework (Mutually exclusive with DDP) |
| compile | Compile before running (Mutually exclusive with max_compile) |
| max_compile | Max-autotune compile (Mutually exclusive with compile) |



<!-- ## 🔗 Citation
If you're using this repository in your research or applications, please cite using the following BibTeX:
```bibtex
@inproceedings{wang2025cbramod,
    title={{CB}raMod: A Criss-Cross Brain Foundation Model for {EEG} Decoding},
    author={Jiquan Wang and Sha Zhao and Zhiling Luo and Yangxuan Zhou and Haiteng Jiang and Shijian Li and Tao Li and Gang Pan},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=NPNUHgHF2w}
}
```

## ⭐ Star History
<div align="center">
    <a href="https://star-history.com/#wjq-learning/CBraMod&Date">
        <img src="https://api.star-history.com/svg?repos=wjq-learning/CBraMod&type=Date" style="width: 80%;" />
    </a>
</div> -->


## ⭐ Acknowledgement
We appreciate the following works for their valuable code:

https://github.com/Thinklab-SJTU/Crossformer

https://github.com/935963004/LaBraM

https://github.com/wjq-learning/CBraMod

https://github.com/BINE022/EEGPT
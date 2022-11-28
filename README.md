# 【AAAI'2023】Revisiting Classifier: Transferring Vision-Language Models for Video Recognition
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transferring-textual-knowledge-for-visual/action-classification-on-kinetics-400)](https://paperswithcode.com/sota/action-classification-on-kinetics-400?p=transferring-textual-knowledge-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transferring-textual-knowledge-for-visual/action-recognition-in-videos-on-activitynet)](https://paperswithcode.com/sota/action-recognition-in-videos-on-activitynet?p=transferring-textual-knowledge-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transferring-textual-knowledge-for-visual/action-recognition-in-videos-on-ucf101)](https://paperswithcode.com/sota/action-recognition-in-videos-on-ucf101?p=transferring-textual-knowledge-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transferring-textual-knowledge-for-visual/zero-shot-action-recognition-on-kinetics)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-kinetics?p=transferring-textual-knowledge-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transferring-textual-knowledge-for-visual/zero-shot-action-recognition-on-activitynet)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-activitynet?p=transferring-textual-knowledge-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transferring-textual-knowledge-for-visual/zero-shot-action-recognition-on-ucf101)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-ucf101?p=transferring-textual-knowledge-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transferring-textual-knowledge-for-visual/zero-shot-action-recognition-on-hmdb51)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-hmdb51?p=transferring-textual-knowledge-for-visual)


This is the official implementation of the paper [Revisiting Classifier: Transferring Vision-Language Models for Video Recognition](https://arxiv.org/abs/2207.01297). 




## Updates
- [] Models: The trained models & logs on Kinetics-400.
- [] Config: All the configs (general/few-shot/zero-shot video recognition) on Kinetics-400 & 600, ActivityNet, UCF, and HMDB.
- [] Code: Zero-shot Evaluation: Half-classes evaluation and Full-classes evaluation.
- [] Code: Distributed training for InfoNCE and Compatible with CE.
- [x] **11/28/2022** Code: Multi-Machine Multi-GPU Distributed Training
- [x] **11/28/2022** Code: Single-Machine Multi-GPU Distributed Training, Distributed testing.
- [x] **11/19/2022** Our paper has been accepted by AAAI-2023.
- [x] **07/01/2022** Our [initial Arxiv paper](https://arxiv.org/abs/2207.01297v1) is released.


## Overview
TODO


## Prerequisites
The code is built with following libraries:

- [PyTorch](https://pytorch.org/) >= 1.8
- RandAugment
- pprint
- tqdm
- dotmap
- yaml
- csv
- Optional: decord (for on-the-fly video training)
- Optional: torchnet (for mAP evaluation on ActivityNet)

## Data Preparation
TODO



## Training

### General Video Recognition
TODO

### Few-shot Video Recognition
TODO


## Testing

### General/Few-shot Video Recognition
TODO

### Zero-shot Evaluation
TODO



## Model Zoo
### Kinetics-400
TODO

### ActivityNet
TODO

### UCF-101
TODO

### HMDB-51
TODO



## Bibtex
If you find this repository useful, please consider citing our paper 😄:

```
@article{wu2022transferring,
  title={Revisiting Classifier: Transferring Vision-Language Models for Video Recognition},
  author={Wu, Wenhao and Sun, Zhun and Ouyang, Wanli},
  booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
  year={2022}
}
```




## Acknowledgement

This repository is built based on [ActionCLIP](https://github.com/sallymmx/actionclip) and [CLIP](https://github.com/openai/CLIP). Sincere thanks to their wonderful works.


## Contact
For any question, please file an issue or contact [Wenhao Wu](https://whwu95.github.io/)


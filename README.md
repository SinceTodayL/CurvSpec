## CurvSpec: Adaptive Multi-Curvature Learning for Partial Relevant Video Retrieval

This repository contains the official implementation of:

> [CurvSpec: Adaptive Multi-Curvature Learning for Partial Relevant Video Retrieval](https://arxiv.org/abs/2609.36815)

### Table of Contents

- [CurvSpec: Adaptive Multi-Curvature Learning for Partial Relevant Video Retrieval](#curvspec-adaptive-multi-curvature-learning-for-partial-relevant-video-retrieval)
  - [Table of Contents](#table-of-contents)
  - [Preparation](#preparation)
    - [Requirements](#requirements)
    - [Datasets](#datasets)
  - [Run](#run)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)


### Preparation

```bash
git clone https://github.com/SinceTodayL/CurvSpec.git
cd CurvSpec
```

#### Requirements

The codebase is packaged with `pyproject.toml`.

```bash
pip install -e .
```

The main dependencies include PyTorch, h5py, NumPy, PyYAML, scikit-learn, SciPy, matplotlib, and tqdm. 
We recommend using Python >= 3.10 and CUDA >= 11.8 with a matching PyTorch build.

#### Datasets

CurvSpec expects pre-extracted video features and RoBERTa text features for ActivityNet Captions, Charades-STA, and TVR. The dataset root passed to `--data-root` should be the directory that directly contains `activitynet`, `charades`, and `tvr`.

Prepared datasets are available on [Google Drive](https://drive.google.com/drive/folders/11dRUeXmsWU25VMVmeuHc9nffzmZhPJEj?usp=sharing), provided by [MS-SL](https://github.com/HuiGuanLab/ms-sl).

For example, if your data is stored as:

```text
dataset/
└── netdisk/
    ├── activitynet/
    ├── charades/
    └── tvr/
```

then use:

```bash
--data-root /dataset/netdisk
```

### Run

All commands below use one GPU per process.


Train on TVR:

```bash
curvspec train --dataset tvr  --data-root /path/to/dataset/netdisk
```

Train on ActivityNet Captions:

```bash
curvspec train --dataset act  --data-root /path/to/dataset/netdisk
```

Train on Charades-STA:

```bash
curvspec train --dataset cha  --data-root /path/to/dataset/netdisk
```

Outputs are saved to:

```text
outputs/<dataset_name>/CurvSpec/
```



### Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{liu2026curvspec,
  title={CurvSpec: Adaptive Multi-Curvature Learning for Partial Relevant Video Retrieval},
  author={Liu, Zhen and Li, Letian and Wang, Jinpeng and Xie, Shuzhao and Huang, Yuzhi and Jiang, Jingyan and Wang, Zhi},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026},
  url={https://arxiv.org/abs/2609.36815}
}
```

### Acknowledgements

This repository builds on prior research and open-source implementations for partial relevant video retrieval and hyperbolic representation learning. We thank the authors of these works for making their code and resources available.

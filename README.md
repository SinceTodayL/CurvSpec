# CurvSpec: Adaptive Multi-Curvature Learning for Partial Relevant Video Retrieval

## Run

```bash
pip install -e .
curvspec train --dataset tvr --gpu 0 --data-root /path/to/dataset/netdisk
curvspec train --dataset act --gpu 0 --data-root /path/to/dataset/netdisk
curvspec train --dataset cha --gpu 0 --data-root /path/to/dataset/netdisk
curvspec evaluate --dataset tvr --gpu 0 --data-root /path/to/dataset/netdisk --checkpoint /path/to/best.ckpt
```

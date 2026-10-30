# CoFTransTANet

Official PyTorch implementation of **CoFTransTANet: A Coarse-to-Fine
Transformer-Based Network for Camouflaged Person Segmentation with Dynamic
Texture Enhancement**.

CoFTransTANet combines a multi-scale transformer encoder, dynamic texture
modelling, and a coarse-to-fine prediction pipeline for pixel-level
segmentation of camouflaged persons.

## Repository structure

```text
CoFTransTANet/
├── lib/                 # Network and backbone definitions
├── utils/               # Data loading and training utilities
├── train.py             # Training and validation
├── test.py              # Prediction generation
├── evaluate.py          # S-measure, weighted F-measure, E-measure, and MAE
└── requirements.txt
```

## Requirements

- Python 3.8 or later
- PyTorch 1.11 or later
- CUDA-capable GPU recommended

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The ImageNet-pretrained Res2Net-50 backbone is downloaded automatically when
training starts for the first time.

## Dataset preparation

The experiments use the dataset introduced with DDCN and the COD10K-Human
subset derived from the official Human category in COD10K. Obtain the original
datasets from their respective published sources and arrange each split as:

```text
Dataset/
├── train/
│   ├── imgs/
│   └── masks/
└── test/
    ├── imgs/
    └── masks/
```

Image and mask filenames must correspond after sorting. Dataset images and
annotations are not redistributed in this repository.

## Training

```bash
python train.py \
  --train_root ./Dataset/train \
  --val_root ./Dataset/test \
  --save_path ./checkpoints/coftranstanet \
  --gpu_id 0
```

Important defaults are 384 x 384 input resolution, batch size 8, 300 epochs,
and an initial learning rate of 1e-5. Override them with the command-line
arguments shown by `python train.py --help`.

## Testing

```bash
python test.py \
  --checkpoint ./checkpoints/coftranstanet/Net_epoch_best.pth \
  --data_root ./Dataset/test \
  --save_root ./results \
  --gpu_id 0
```

Predicted masks are saved under `results/<dataset-name>/`.

## Evaluation

```bash
python evaluate.py \
  --mask_root ./Dataset/test/masks \
  --pred_root ./results/test
```

The evaluation script reports S-measure, weighted F-measure, adaptive/mean/max
E-measure, adaptive/mean/max F-measure, and MAE.

## Pretrained model

The trained CoFTransTANet checkpoint will be provided here. Large checkpoint
files should be distributed through a GitHub Release or an external archival
repository rather than committed directly to Git.

## Acknowledgements

This implementation uses the Res2Net-50 backbone and adapts components from
the Fully Convolutional Transformer (FCT). The evaluation implementation is
based on `py_sod_metrics`. Please cite the corresponding publications and
respect their licences when using this repository.

## Citation

Citation information will be updated after publication.

## Licence

The project is released under the MIT License. See [LICENSE](LICENSE).


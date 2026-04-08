# CoordFusion

---

## Requirements

- Python 3.8+
- Key libraries:
- PyTorch with CUDA support
- `segmentation_models_pytorch`
- `loguru`, `tqdm`, `tensorboard`

Install dependencies:

```bash
uv pip install -r requirements.txt
```

---

## Dataset

Download the [Sen1Floods11 dataset](https://github.com/cloudtostreet/Sen1Floods11) and place it at:

```
./datasets/sen1floods11_v1.1/
```

or run existing script from parents directory

```
python ./datasets/download_sen1.py
```

---

## Usage

### Run Baseline Models

Trains all baseline models (UNet, UNet3+, DeepLabV3, Prithvi variants, EvaNet, etc.) for comparison:

```bash
python3 multimodal_trainer.py \
  --run_baseline \
  --loss_func tversky \
  --epochs 300 \
  --batch_size 6 \
  --grad_accumulation 2 \
  --mixed_precision
```

### Run Proposed Model

Trains the proposed `DSUNet_Coord_SE` model:

```bash
python3 multimodal_trainer.py \
  --loss_func tversky \
  --epochs 300 \
  --batch_size 6 \
  --grad_accumulation 2 \
  --mixed_precision
```

---

## Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data_path` | `str` | `./datasets/sen1floods11_v1.1` | Path to dataset directory |
| `--version` | `str` | `Multimodal` | Experiment version label |
| `--batch_size` | `int` | `6` | Batch size for training |
| `--epochs` | `int` | `300` | Number of training epochs |
| `--learning_rate` | `float` | `5e-4` | Initial learning rate |
| `--test_interval` | `int` | `1` | Validate every N epochs |
| `--loss_func` | `str` | `tversky` | Loss function: `bce`, `diceloss`, `dl2`, `wce`, `focal`, `lovasz`, `tversky`, `evaloss` |
| `--finetune_ratio` | `float` | `1.0` | Fine-tune epoch ratio for Prithvi/ Pretrained Foundation models |
| `--run_baseline` | flag | `False` | Run all baseline models instead of proposed model |
| `--full_seed` | flag | `False` | Run 5-seed experiment `[124, 42, 12, 114, 28]` |
| `--mixed_precision` | flag | `True` | Use bfloat16 mixed precision training |
| `--grad_accumulation` | int | `2` | Enable gradient accumulation |

---

## Baseline Models

| Model | Backbone | Input |
|---|---|---|
| `UNet_Sentinel2` | UNet | Sentinel-2 (6ch) |
| `UNet3+_Sentinel2` | UNet3+ | Sentinel-2 (6ch) |
| `DualStream_UNet_Sentinel1_2` | Dual-Stream UNet | Sentinel-1 + Sentinel-2 (8ch) |
| `Prithvi_UNet_Sentinel2` | Prithvi + UNet | Sentinel-2 (6ch) |
| `Prithvi_Segmenter_Sentinel2` | Prithvi Segmenter | Sentinel-2 (6ch) |
| `DeeplabV3_Resnet50_Sentinel2` | DeepLabV3 + ResNet-50 | Sentinel-2 (6ch) |
| `DeeplabV3_MobilenetV2Large_Sentinel2` | DeepLabV3 + MobileNetV3-Large | Sentinel-2 (6ch) |
| `DualSwinTransUNet_Sentinel2` | Swin TransUNet | Sentinel-2 (6ch) |
| `PrithviCAFE_Sentinel2` | PrithviCAFE | Sentinel-2 (6ch) |
| `EvaNet_Sentinel2` | EvaNet | Sentinel-2 (6ch) |

## Proposed Model

| Model | Description | Input |
|---|---|---|
| `DSUNet_Coord_SE` | Dual-stream UNet with CoordConv skip attention and Squeeze-and-Excitation end attention | Sentinel-1 + Sentinel 2 (8ch) + DEM + JRCPW (2ch) 

---

## Outputs

- **TensorBoard logs**: `./logs/<run_name>/`
- **Model checkpoints**: `./logs/<run_name>/<model_name>/models/checkpoint_epoch_N.pt`
- **Final model weights**: `./logs/<run_name>/<model_name>/models/model_final.pt`
- **Per-seed results**: `./logs/<run_dir>/multimodal_e<epochs>_<loss>.json`
- **Aggregated results** (multi-seed): `./logs/multimodal_e<epochs>_<loss>_all_seeds.json`

View training progress:

```bash
tensorboard --logdir ./logs
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| `IOU_floods` | Intersection over Union for flood class |
| `IOU_non_floods` | Intersection over Union for non-flood class |
| `Avg_IOU` | Mean IoU across both classes |
| `ACC_floods` | Recall for flood class |
| `ACC_non_floods` | Recall for non-flood class |
| `Avg_ACC` | Mean accuracy across both classes |

Evaluation is performed on two held-out splits: **Test Set** and **Bolivia Set** (out-of-distribution generalization).
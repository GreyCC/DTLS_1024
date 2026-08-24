# Domain Transfer in Latent Space (DTLS) for Image Super-Resolution

Official code for **“Domain Transfer in Latent Space (DTLS) Wins on Image Super-Resolution — A Non-Denoising Model,” 2026, Under Revision in Transaction of Consumer Electronics.** 

> **Project status:** This repository is research code. The commands below describe the intended FFHQ workflow. Please read [Important implementation notes](#important-implementation-notes) before launching a long training run.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Inference](#inference)
- [Outputs](#outputs)
- [Important implementation notes](#important-implementation-notes)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Requirements

- Linux (the code has been developed and tested as research code on Linux)
- Python 3.8+ recommended
- An NVIDIA GPU with a CUDA-enabled PyTorch installation
- Sufficient GPU memory for the selected batch size and image resolution
- [Weights & Biases](https://wandb.ai/) account, because `train.py` calls `wandb.init()` and logs training images

The repository does not currently include a `requirements.txt` or environment file. Install a compatible PyTorch/torchvision pair for your CUDA version, then install the Python packages imported by the training and evaluation scripts:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the CUDA-compatible torch and torchvision versions for your system.
# See https://pytorch.org/get-started/locally/ for the appropriate command.
python -m pip install torch torchvision
python -m pip install einops imgaug numpy Pillow tqdm wandb lmdb opencv-python
```

Verify the main dependencies and CUDA visibility:

```bash
python - <<'PY'
import torch
import torchvision
import einops
import imgaug
import wandb

print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

Log in to Weights & Biases before training, unless you have configured another W&B authentication method:

```bash
wandb login
```

## Installation

Clone the repository and enter its root directory:

```bash
git clone https://github.com/GreyCC/DTLS_1024.git
cd DTLS_1024
```

Run all commands from this directory so that local imports such as `models.py`, `operation.py`, and `diffaug.py` resolve correctly.

## Dataset

### FFHQ training data

The experiments use the [FFHQ dataset](https://github.com/NVlabs/ffhq). A project-provided download is available here:

<https://drive.google.com/file/d/1WvlAIvuochQn_L_f9p3OdFdTiSLlnnhv/view?usp=drive_link>

Download and extract the dataset, then place it under the repository or provide its absolute path. The training dataset should contain image files (`.jpg`, `.jpeg`, or `.png`) in the dataset directory or one of its subdirectories. For example:

```text
DTLS_1024/
├── train.py
├── eval.py
├── models.py
└── FFHQ1024/
    ├── 00000.png
    ├── 00001.png
    └── ...
```

The generic dataset loader recursively searches for supported image extensions. Make sure the directory contains at least one readable image and that images are RGB (or can be converted to RGB by your local dataset preparation step).

### Evaluation images

The original project uses a folder named `NLQ_Faces` for natural low-resolution face images. If you have this folder, place it in the repository root:

```text
DTLS_1024/
└── NLQ_Faces/
    ├── face_0001.png
    ├── face_0002.png
    └── ...
```

If the folder is not included in your checkout, create it and add your own low-resolution inputs. Supported input extensions are `.jpg`, `.jpeg`, and `.png`.

## Training

The documented experiment command is:

```bash
python train.py \
  --path FFHQ1024 \
  --cuda 0 \
  --name DTLS_1024_FFHQ
```

Equivalent one-line form:

```bash
python train.py --path FFHQ1024 --cuda 0 --name DTLS_1024_FFHQ
```

### Training arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--path` | `/hdda/Datasets/Face_super_resolution/images1024x1024` | Path to the training dataset. See the implementation note below. |
| `--output_path` | `./` | Root directory for experiment outputs. |
| `--cuda` | `0` | CUDA device index, for example `1` for the second GPU. |
| `--name` | `DTLS_xlpsr` | Experiment name. |
| `--iter` | `100000` | Number of training iterations. |
| `--start_iter` | `0` | Starting iteration when resuming. |
| `--batch_size` | `8` | Batch size. Reduce it if you run out of GPU memory. |
| `--im_size` | `256` | Training image size in the current script. |
| `--ckpt` | `None` | Checkpoint path for resuming training. |
| `--workers` | `0` | Number of DataLoader workers. |
| `--save_interval` | `1000` | Interval used for training previews; model checkpoints are saved at the script’s checkpoint interval. |

A run named `DTLS_1024_FFHQ` is intended to produce:

```text
./train_results/DTLS_1024_FFHQ/
├── args.txt
├── images/
│   ├── <iteration>.jpg
│   ├── domains_<iteration>.jpg
│   └── training_<iteration>.jpg
├── models/
│   └── <iteration>.pth
└── *.py                  # copied source files
```

Each model checkpoint is a PyTorch dictionary with `enc` and `dec` state dictionaries. Training can take from approximately one to three days depending on the GPU, batch size, and number of iterations.

### Resuming training

Pass a checkpoint with `--ckpt` and set `--start_iter` consistently with that checkpoint, for example:

```bash
python train.py \
  --path FFHQ1024 \
  --cuda 0 \
  --name DTLS_1024_FFHQ_resume \
  --ckpt train_results/DTLS_1024_FFHQ/models/300000.pth \
  --start_iter 300000
```

## Inference

Run inference with a trained checkpoint:

```bash
python eval.py \
  --path NLQ_Faces \
  --cuda 0 \
  --output_path DTLS_1024_NLR_test \
  --ckpt train_results/DTLS_1024_FFHQ/models/300000.pth
```

The results are written to:

```text
eval/DTLS_1024_NLR_test/
```

### Inference arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--path` | `world_real_lr_iii` | Folder containing input images. |
| `--output_path` | `DTLS_realLR_512_32_iv` | Output folder name below `eval/`. |
| `--cuda` | `0` | CUDA device index. |
| `--batch_size` | `1` | Inference batch size. |
| `--im_size` | `1024` | Input/model image resolution expected by this evaluation script. |
| `--ckpt` | Repository-specific path | Trained checkpoint containing `enc` and `dec`. |
| `--workers` | `1` | Number of DataLoader workers. |
| `--samples` | `4` | Number of samples in the script interface. |
| `--input_folder` | `None` | Reserved for an alternative input path; the corresponding branch is currently commented out in `eval.py`. |

The evaluation script currently preserves each input filename when saving its result. It also upsamples the generated result to 512 pixels before saving; inspect `eval.py` if you need a different output resolution.

## Outputs

- Training checkpoints: `train_results/<experiment-name>/models/`
- Training previews: `train_results/<experiment-name>/images/`
- Saved training arguments: `train_results/<experiment-name>/args.txt`
- Evaluation results: `eval/<output-path>/`

Checkpoints are not included in the source repository. Download or train a checkpoint before running evaluation, and ensure the checkpoint path is correct relative to the repository root.

## Important implementation notes

The current checked-in `train.py` contains research/development code that does **not** yet use the command-line dataset path in the active training loader. Inside `train()`, the active loader is:

```python
PlateDataset("/hdda/Datasets/xlpsr/custom_dataset")
```

Consequently, `--path FFHQ1024` is parsed but is currently ignored by the active training path. `PlateDataset` also expects an `images/` subdirectory and applies plate-specific preprocessing. To train directly on FFHQ with the generic recursive `Dataset` class, the active loader must be changed to something equivalent to:

```python
dataset = Dataset(data_root, im_size)
```

This README intentionally does not modify source code. If you want a turnkey public release, this is the first code change to make, followed by adding a pinned dependency file and a CPU/device check. Likewise, the direct arbitrary-input-folder branch in `eval.py` is commented out; the currently active path is `--path`.

Before a multi-day run, perform a short smoke test after resolving the loader issue:

```bash
python train.py --path FFHQ1024 --cuda 0 --name smoke_test --iter 2 --batch_size 1 --workers 0
```

## Troubleshooting

### `ModuleNotFoundError`

Install the missing package in the active virtual environment. The most common dependencies are `torch`, `torchvision`, `einops`, `imgaug`, `Pillow`, `tqdm`, `wandb`, `lmdb`, and `opencv-python`.

### CUDA errors or out-of-memory errors

- Confirm the selected GPU with `--cuda`.
- Reduce `--batch_size`.
- Use fewer DataLoader workers with `--workers 0`.
- Confirm that the installed PyTorch build supports the installed NVIDIA driver.

### No images found

Confirm that `--path` is correct, that it contains `.jpg`, `.jpeg`, or `.png` files, and that you are running the command from the repository root. An empty dataset causes the sampler to fail.

### Checkpoint loading fails

Use a checkpoint generated by this repository. It must contain `enc` and `dec` keys, and the model architecture/resolution configuration must match the checkpoint.

### Weights & Biases prompts for login

Run `wandb login`, or configure W&B according to its documentation before starting training. Training calls `wandb.init(project="DTLS_challenge", name=<experiment-name>)`.

## Citation

If you use this code or method in your research, please cite:

```bibtex
@article{hui2025dtls,
  title   = {Domain Transfer in Latent Space (DTLS) Wins on Image Super-Resolution - A Non-Denoising Model},
  author  = {Hui, Chun-Chuen and Siu, Wan-Chi and Law, Ngai-Fong},
  year    = {2026}
}
```

## Authors

- Chun-Chuen Hui
- Wan-Chi Siu
- Ngai-Fong Law

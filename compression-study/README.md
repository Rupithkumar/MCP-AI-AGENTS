# Model Compression Study: what shrinks a neural network, and what does it cost?

Takes one trained image classifier (ResNet-18-sized, CIFAR-10) and compresses it five ways, then measures **accuracy, size on
disk, CPU speed and GPU speed** for every variant, so the trade-offs are numbers instead of folklore.

| Technique | What it does | Where it runs |
|---|---|---|
| fp16 / bf16 | 16-bit weights and maths | GPU |
| int8 static quantization | 8-bit weights **and** activations, calibrated on sample data | CPU |
| Magnitude pruning (50/80/90%) | zero the smallest weights, retrain briefly | both (but see below) |
| Smaller architecture | a half-width student (about a quarter of the parameters), trained alone | both |
| Knowledge distillation | the same student, taught by the big model's soft predictions | both |
| Distillation + int8 | the two combined | CPU |

The question it answers honestly: *which of these actually make the model cheaper to run, and which only look smaller on paper?*
(Unstructured pruning is the classic example: the file compresses, but a normal dense kernel does the same work. The study measures that
instead of asserting it.)

## Status: read this first

The pipeline is finished and tested (12 tests, including a full mini-study on a CPU that also checks the resume cache). **No real run
has been made.** The sandbox this was built in has no GPU and cannot download CIFAR-10, so there are **no real results in this package**,
and every figure you will quote comes from running it on your machine. The only numbers I saw were from a 6-epoch run on tiny fake data
used to check the plumbing; they mean nothing and are not included.

## Run it (Windows, PowerShell, VS Code)

```powershell
# 1. environment
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. PyTorch with GPU support (RTX 50-series needs a CUDA 12.8+ build; if this fails use the selector at pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3. tests (about a minute)
python -m pytest

# 4. quick trial, 2 epochs on 5000 images (downloads CIFAR-10, 163 MB, the first time)
python -m src.study --epochs 2 --subset 5000

# 5. the real run (checkpoints are cached in outputs/, so an interrupted run resumes)
python -m src.study --epochs 30
```

Output: `results/REPORT.md` (table, chart, findings computed from the measured rows), `results/chart.png`, `results/results.json`.
No GPU? Add `--cpu` for the pipeline, but expect training to be slow; a CPU run is fine for checking the code, not for the final numbers.

## How it works

```
src/models.py     ResNet for 32x32 images, width is a parameter; built so int8 conversion works
src/data.py       CIFAR-10 loaded without torchvision; whole dataset and augmentation on the GPU (no Windows worker problems)
src/train.py      training (bf16, one-cycle schedule) and the distillation loss
src/compress.py   pruning, static int8 quantization, half precision
src/benchmark.py  size, gzip size, latency (median of repeats, GPU synchronised) and accuracy
src/study.py      runs every variant, caches checkpoints, writes the results
src/report.py     REPORT.md and chart
```

## Choices worth knowing about

- **Latency is measured, not estimated**, using the median of repeated runs with warm-up, and the GPU is synchronised so timings are real.
- **gzip size is reported next to raw size** because zeros cost nothing once compressed but still cost memory and compute when running.
- **Findings in the report are generated from the rows**, so the text cannot disagree with the table.
- **No torchvision** to avoid version mismatches with a CUDA 12.8 build of PyTorch.

## Limitations (say these out loud when presenting)

- One dataset, one architecture, **one run per variant**. There are no error bars, so differences under about a point are not reliable. Repeating with different seeds would be the first improvement.
- Latency depends on the machine and what else is running; compare variants only within one run. int8 speed-ups depend on the CPU having fast int8 instructions.
- Post-training int8 uses PyTorch's eager-mode quantization, which PyTorch is moving to the torchao library. It works in current releases but may need porting later.
- Structured pruning (removing whole channels so the layer genuinely shrinks) is not covered.
- Image classification only. Compressing large language models needs different tools.

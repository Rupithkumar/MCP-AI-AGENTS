# LoRA Ticket Classifier: fine-tuning vs prompting vs classic ML

Does fine-tuning a small open model beat just prompting it? And does either beat a 10-line scikit-learn baseline?
This project answers that on one task (support-ticket intent classification, 20 intents from Banking77) with the
same held-out test set for every system, and reports accuracy, macro-F1, invalid-answer rate and inference cost.

| System | What it is |
|---|---|
| `tfidf` | TF-IDF + logistic regression. Trains in seconds on a CPU. |
| `base_zero` | Qwen2.5-0.5B-Instruct, prompted with the label list, no training |
| `base_few` | Same model, label list plus one example per label |
| `lora` | Same model, LoRA fine-tuned, short prompt (it learned the labels, so the prompt does not list them) |
| `hosted` | Optional: a large hosted model (e.g. Groq) few-shot, on a subset |

## Status: read this first

The code, 12 tests (loss masking, parsing, metrics, data splits, a tiny-model train/save/load/generate cycle, report) pass on a
CPU with a tiny random model. **No real training run and no real results exist yet.** The numbers come from running this on
your GPU. Two things I could not verify from where this was built: the Hugging Face dataset and model names
(`mteb/banking77`, `PolyAI/banking77`, `Qwen/Qwen2.5-0.5B-Instruct`) and the PyTorch CUDA build for an RTX 5050.
If a name fails, the error says what to do; `--data-dir` accepts your own `train.csv` / `test.csv` (columns `text,label`).

## Run it (Windows, PowerShell, VS Code terminal)

```powershell
# 1. environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. dependencies
pip install -r requirements.txt

# 3. PyTorch with GPU support. RTX 50-series needs a CUDA 12.8+ build:
pip install torch --index-url https://download.pytorch.org/whl/cu128
#    if that fails, use the selector at pytorch.org (Windows, Pip, newest CUDA)

# 4. confirm the GPU works BEFORE training
python -m src.check_gpu

# 5. data (downloads Banking77, keeps 20 intents, holds out a test set)
python -m src.prepare_data --num-labels 20

# 6. baselines, no training needed
python -m src.evaluate --systems tfidf,base_zero,base_few

# 7. fine-tune (about 1 GB model; a few minutes on a modern GPU)
python -m src.train

# 8. evaluate the fine-tuned model, then build the report
python -m src.evaluate --systems lora
python -m src.report          # writes results/REPORT.md

# optional: a big hosted model on 200 examples (set your own key; never commit it)
$env:LLM_BASE_URL="https://api.groq.com/openai/v1"; $env:LLM_MODEL="openai/gpt-oss-120b"; $env:LLM_API_KEY="<your key>"
python -m src.evaluate --systems hosted --limit 200 --delay 2
python -m src.report
```

Quick trial first: `python -m src.train --epochs 1 --limit-train 500`. Out of GPU memory: lower `--batch-size`, add `--grad-ckpt`.
Out of time: `--model Qwen/Qwen2.5-1.5B-Instruct` for a stronger base if the card has the memory.

## Design choices worth knowing

- **Loss only on the answer.** Prompt tokens are labelled -100, so the model learns the label, not the instruction text. Tested.
- **Generative classification, strict scoring.** The model must output a label. Anything else counts as wrong and is reported as the invalid rate; it is never mapped to the closest label.
- **Fair comparison.** One untouched test set; the hosted model's smaller subset is handled by a same-subset table, so it is not compared against a different sample.
- **Fine-tuning's real advantage is cost.** Compare "Prompt tokens" in the report: the tuned model needs a short prompt, the few-shot model needs the whole label list and examples on every call.
- **No QLoRA.** bitsandbytes is unreliable on Windows and new GPUs; bf16 LoRA on a 0.5B model fits easily.
- **Hosted errors** (rate limits, timeouts) are counted separately and excluded from the score, never scored as wrong labels.

## Project layout

```
src/data.py, prepare_data.py   load Banking77 (or CSV), pick labels, stratified train/val/test, few-shot examples
src/prompts.py                 three prompt styles
src/train.py                   LoRA fine-tuning with answer-only loss
src/predictors.py              the five systems
src/evaluate.py                score systems, one result file each
src/report.py                  results/REPORT.md
src/check_gpu.py               verify the GPU works
tests/                         12 tests on a tiny CPU model
```

## What to say about it

When you have real numbers, put them in the table above and be specific about what they show. Do not claim a result the
report does not contain. If the TF-IDF baseline wins on a clean 20-class task, that is a legitimate finding, not a failure.

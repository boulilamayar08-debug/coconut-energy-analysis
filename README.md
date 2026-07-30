# Coconut Training-vs-Inference Energy Analysis

A reproducibility study measuring the training-side energy cost of Coconut
(continuous-thought latent reasoning; Hao et al., 2024) against its inference-side
token/energy savings, compared to a standard Chain-of-Thought (CoT) baseline.
Companion code for *"Is Thinking Without a Language Sustainable? An Energy and
Carbon Analysis of Continuous-Thought Reasoning in AI."*

## Motivation

Prior work on Coconut and similar latent-reasoning methods reports inference-time
efficiency gains but does not account for the additional training cost of the
staged curriculum required to teach a model to reason without emitting text. This
repository implements a training-vs-inference energy break-even framework —
extending the training/inference cost logic of Luccioni et al. (2024) to the
CoT-vs-Coconut comparison specifically — and applies it empirically.

## What's here

- `scripts/` — standalone Python scripts for CPU/local runs (data conversion,
  training, evaluation, break-even calculation)
- `notebooks/coconut_scaled_1000.ipynb` — Colab-ready notebook (GPU, mixed
  precision, checkpointing/resume, persistent logging) for the 1,000-example run
- `results/` — measured energy logs, run metadata, and evaluation outputs from
  our runs

## Reproducing

**Local/CPU (small-scale pilot, ~150 examples):**
```bash
pip install -r requirements.txt
cd scripts
python convert_prosqa.py --input_path /path/to/prosqa_train.json --output_dir ../data --n_train 150 --n_eval 50
python train_baseline_cot.py --data_path ../data/train_baseline.jsonl --output_dir ../baseline_cot_model
python train_coconut.py --data_path ../data/train_coconut.jsonl --output_dir ../coconut_model
python eval_and_count_tokens.py --model_dir ../baseline_cot_model --data_path ../data/eval_baseline.jsonl
python eval_and_count_tokens.py --model_dir ../coconut_model --data_path ../data/eval_coconut.jsonl
python breakeven_calculator.py --baseline_dir ../baseline_cot_model --coconut_dir ../coconut_model --baseline_avg_tokens <X> --coconut_avg_tokens <Y>
```

**Colab/GPU (scaled, 1,000 examples):** upload `notebooks/coconut_scaled_1000.ipynb`
to Colab, set Runtime → T4 GPU, run cells top to bottom. Data is pulled directly
from the [official Coconut repository](https://github.com/facebookresearch/coconut).

## Dataset

ProsQA (Hao et al., 2024), sourced directly from
`facebookresearch/coconut/data/prosqa_{train,test}.json`. Not redistributed here;
scripts download it automatically.

## Results (150-example pilot run)

| Metric | Baseline CoT | Coconut |
|---|---|---|
| Training epochs / steps | 3 epochs (450 steps) | 3 stages × 1 epoch (450 steps) |
| Wall-clock training time | 3.0621 h | 2.1396 h |
| Training energy (estimated) | 0.076552 kWh | 0.053489 kWh |
| Task accuracy (50 held-out) | 0.00% | 0.00% |
| Avg. tokens/response | 37.7 | 36.6 |
| Est. energy/query | 0.0000273 kWh | 0.0000265 kWh |

At this scale, Coconut's staged curriculum consumed less total training energy
than standard CoT while also producing fewer tokens per response — no break-even
was required. See the paper for full discussion of why this should **not** be
read as a general claim about Coconut's efficiency; see Limitations below.

*(Scaled 1,000-example results to be added following the run in `notebooks/coconut_scaled_1000.ipynb`.)*

## Limitations

- Single random seed per condition (no variance estimate)
- Energy is estimated from device TDP × wall-clock time, not measured via live
  power sensors
- Small-scale (150–1,000 examples) relative to the original paper's full ProsQA
  training set (~17,886 examples)
- This is a simplified, single-file reimplementation of the Coconut mechanism,
  not the official `facebookresearch/coconut` training pipeline
- Latent-state substitution is applied during training only; at inference the
  `<|latent|>` token is generated as an ordinary discrete token rather than
  recursively re-injected as a hidden state

## Citation

See `CITATION.cff`.

## References

- Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., & Tian, Y. (2024).
  Training large language models to reason in a continuous latent space.
  arXiv:2412.06769.
- Luccioni, A. S., Jernite, Y., & Strubell, E. (2024). Power hungry processing:
  Watts driving the cost of AI deployment? FAccT '24.
- Husom, E. J. et al. (2025). Sustainable LLM inference for edge AI. arXiv:2504.03360.

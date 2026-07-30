# Results

## Pilot run (150 train / 50 eval, Colab CPU)

| Metric | Baseline CoT | Coconut |
|---|---|---|
| Training examples | 150 | 150 |
| Training epochs / steps | 3 epochs (450 steps) | 3 stages × 1 epoch (450 steps) |
| Hardware | Colab CPU (~25W est.) | Colab CPU (~25W est.) |
| Wall-clock training time | 3.0621 h | 2.1396 h |
| Training energy (estimated) | 0.076552 kWh | 0.053489 kWh |
| Task accuracy (50 held-out) | 0.00% | 0.00% |
| Avg. tokens/response | 37.7 | 36.6 |
| Est. energy/query (J/token=2.61) | 0.0000273 kWh | 0.0000265 kWh |

**Break-even:** not applicable — Coconut was cheaper on both training and inference energy at this scale (see `breakeven_calculator.py` output in `results/pilot_150/`).

Source files: `results/pilot_150/baseline_run_metadata.json`, `results/pilot_150/baseline_energy_log.json`, `results/pilot_150/baseline_eval_results.json`, `results/pilot_150/coconut_run_metadata.json`, `results/pilot_150/coconut_energy_log.json`, `results/pilot_150/coconut_eval_results.json`

---

## Scaled run (1,000 train / 200 eval, Colab T4 GPU)

| Metric | Baseline CoT | Coconut |
|---|---|---|
| Training examples | 1000 | 1000 |
| Training epochs / steps | 3 epochs | 3 stages × 1 epoch |
| Hardware | Colab T4 GPU (70W TDP) | Colab T4 GPU (70W TDP) |
| Wall-clock training time | 0.0669 h | 0.0755 h |
| Training energy (estimated) | 0.004683 kWh | 0.005286 kWh |
| Task accuracy (200 held-out) | 0.00% | 0.00% |
| Avg. tokens/response | 52.1 | 40.8 |
| Est. energy/query (J/token=2.61) | 0.0000378 kWh | 0.0000296 kWh |

**Break-even: 74 queries.** Unlike the pilot run, Coconut's staged curriculum here trained at slightly *higher* energy cost than baseline (0.005286 kWh vs. 0.004683 kWh, +0.000603 kWh) — but because Coconut generates markedly fewer tokens per response (40.8 vs. 52.1), that extra training cost is paid back after just 74 inference queries, after which Coconut is the lower-energy option for every subsequent query.

Source files: `results/scaled_1000/baseline_run_metadata.json`, `results/scaled_1000/baseline_energy_log.json`, `results/scaled_1000/baseline_eval_results.json`, `results/scaled_1000/coconut_run_metadata.json`, `results/scaled_1000/coconut_energy_log.json`, `results/scaled_1000/coconut_eval_results.json`

---

## How to update this table

1. Open each `run_metadata.json` — copy `elapsed_hours` and `energy_kwh`
2. Open each `eval_results.json` — copy `accuracy` and `avg_generated_tokens`
3. Replace the `*[fill in]*` placeholders above with those numbers
4. Re-run the break-even calculation (Cell 11 in the notebook, or `breakeven_calculator.py` locally) with the two `avg_generated_tokens` values, and paste the verdict (break-even query count, or "no break-even needed") into the Break-even line

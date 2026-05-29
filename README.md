# neyman-pearson-gap — Optimal Deadbands as Hypothesis Tests

Compute context-dependent optimal deadbands using the Neyman-Pearson lemma. The optimal gap minimizes the combined cost of false positives (responding to noise) and false negatives (missing real signals).

## What This Gives You

- **Cost-optimal thresholds** — balance false-positive and false-negative costs with configurable priors
- **Gaussian signal model** — noise and signal distributions with arbitrary means and standard deviations
- **Golden-section search** — efficiently finds the threshold that minimizes expected cost
- **Full diagnostics** — false positive rate, false negative rate, and total expected cost at the optimum

## Quick Start

```rust
use neyman_pearson_gap::{optimal_deadband, CostModel, SignalModel};

let model = SignalModel {
    noise_mean: 0.0,
    noise_std: 1.0,
    signal_mean: 3.0,
    signal_std: 1.0,
};

let cost = CostModel {
    fp_cost: 1.0,    // cost of false alarm
    fn_cost: 5.0,    // cost of missing a signal
    signal_prior: 0.3,
};

let gap = optimal_deadband(&model, &cost);
println!("Optimal threshold: [{:.3}, {:.3}]", gap.lower, gap.upper);
println!("Expected cost: {:.4}", gap.expected_cost);
println!("FP rate: {:.4}, FN rate: {:.4}", gap.fp_rate, gap.fn_rate);
```

## API Reference

| Type | Description |
|------|-------------|
| `CostModel` | Configurable costs for FP, FN, and signal prior probability |
| `SignalModel` | Gaussian noise and signal distribution parameters |
| `OptimalGap` | Result: lower/upper thresholds, expected cost, FP/FN rates |
| `optimal_deadband()` | Computes the cost-minimizing threshold via golden-section search |

## How It Fits

Used by [spectral-deadband](https://github.com/SuperInstance/spectral-deadband) to give the spectral gap a rigorous statistical foundation. The deadband isn't ad hoc — it's the Neyman-Pearson optimal decision boundary for distinguishing noise from signal.

## Installation

```toml
[dependencies]
neyman-pearson-gap = "0.1"
```

## License

MIT

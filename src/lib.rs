//! Neyman-Pearson gap: context-dependent optimal deadband as hypothesis test.
//! The optimal gap minimizes false-positive + false-negative cost.

/// Cost model for false positives vs false negatives
#[derive(Clone, Copy)]
pub struct CostModel {
    /// Cost of false positive (responding to noise)
    pub fp_cost: f64,
    /// Cost of false negative (missing a real signal)
    pub fn_cost: f64,
    /// Prior probability of signal being present
    pub signal_prior: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self { fp_cost: 1.0, fn_cost: 1.0, signal_prior: 0.5 }
    }
}

/// Signal distribution parameters (assumed Gaussian)
#[derive(Clone, Copy)]
pub struct SignalModel {
    pub noise_mean: f64,
    pub noise_std: f64,
    pub signal_mean: f64,
    pub signal_std: f64,
}

impl Default for SignalModel {
    fn default() -> Self {
        Self { noise_mean: 0.0, noise_std: 1.0, signal_mean: 3.0, signal_std: 1.0 }
    }
}

/// Result of optimal deadband computation
#[derive(Debug)]
pub struct OptimalGap {
    /// Optimal lower threshold
    pub lower: f64,
    /// Optimal upper threshold
    pub upper: f64,
    /// Expected total cost at this threshold
    pub expected_cost: f64,
    /// False positive rate at this threshold
    pub fp_rate: f64,
    /// False negative rate at this threshold
    pub fn_rate: f64,
}

/// Standard normal CDF approximation (Abramowitz & Stegun)
fn phi(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2).min(1.0).max(-1.0))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Compute the optimal Neyman-Pearson deadband
/// For a symmetric deadband around the midpoint of noise and signal distributions,
/// find the threshold that minimizes expected cost.
pub fn optimal_deadband(model: &SignalModel, cost: &CostModel) -> OptimalGap {
    let midpoint = (model.noise_mean + model.signal_mean) / 2.0;

    // Search for optimal threshold using golden section
    let lo = model.noise_mean - 3.0 * model.noise_std;
    let hi = model.signal_mean + 3.0 * model.signal_std;
    let phi_val = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut a = lo;
    let mut b = hi;
    let mut c = b - (b - a) / phi_val;
    let mut d = a + (b - a) / phi_val;

    for _ in 0..100 {
        if expected_cost(c, model, cost) < expected_cost(d, model, cost) {
            b = d;
        } else {
            a = c;
        }
        c = b - (b - a) / phi_val;
        d = a + (b - a) / phi_val;
    }
    let threshold = (a + b) / 2.0;

    // Compute deadband: threshold to (signal_mean - threshold) gap
    let fp_rate = 1.0 - phi((threshold - model.noise_mean) / model.noise_std);
    let fn_rate = phi((threshold - model.signal_mean) / model.signal_std);

    OptimalGap {
        lower: threshold,
        upper: 2.0 * midpoint - threshold,
        expected_cost: expected_cost(threshold, model, cost),
        fp_rate,
        fn_rate,
    }
}

fn expected_cost(threshold: f64, model: &SignalModel, cost: &CostModel) -> f64 {
    let fp_rate = 1.0 - phi((threshold - model.noise_mean) / model.noise_std);
    let fn_rate = phi((threshold - model.signal_mean) / model.signal_std);
    (1.0 - cost.signal_prior) * fp_rate * cost.fp_cost + cost.signal_prior * fn_rate * cost.fn_cost
}

/// Spectral deadband: compute the optimal spectral gap for a graph
/// given costs of over-connection and under-connection
pub fn spectral_deadband(
    eigenvalues: &[f64],
    overconnect_cost: f64,
    underconnect_cost: f64,
) -> f64 {
    if eigenvalues.len() < 2 { return 0.0; }
    let lambda_2 = eigenvalues[1];
    let lambda_n = *eigenvalues.last().unwrap();
    if lambda_n <= 0.0 { return 0.0; }
    // CR = λ₂/λₙ, optimal when weighted costs balance
    let cr = lambda_2 / lambda_n;
    // The "optimal" gap in the Neyman-Pearson sense:
    // minimize overconnect_cost * (1-CR) + underconnect_cost * CR
    // derivative = -overconnect_cost + underconnect_cost = 0 at balance
    // optimum CR = overconnect_cost / (overconnect_cost + underconnect_cost)
    let optimal_cr = overconnect_cost / (overconnect_cost + underconnect_cost);
    // Return the gap between actual and optimal
    (cr - optimal_cr).abs()
}

/// Likelihood ratio test: is observation x more likely signal or noise?
pub fn likelihood_ratio(x: f64, model: &SignalModel) -> f64 {
    let noise_ll = gaussian_log_pdf(x, model.noise_mean, model.noise_std);
    let signal_ll = gaussian_log_pdf(x, model.signal_mean, model.signal_std);
    (signal_ll - noise_ll).exp()
}

fn gaussian_log_pdf(x: f64, mean: f64, std: f64) -> f64 {
    let z = (x - mean) / std;
    -0.5 * z * z - std.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// ROC curve: sweep thresholds and compute (FPR, TPR) pairs
pub fn roc_curve(model: &SignalModel, n_points: usize) -> Vec<(f64, f64)> {
    let lo = model.noise_mean - 4.0 * model.noise_std;
    let hi = model.signal_mean + 4.0 * model.signal_std;
    (0..=n_points)
        .map(|i| {
            let t = lo + (hi - lo) * i as f64 / n_points as f64;
            let fpr = 1.0 - phi((t - model.noise_mean) / model.noise_std);
            let tpr = 1.0 - phi((t - model.signal_mean) / model.signal_std);
            (fpr, tpr)
        })
        .collect()
}

/// AUC (Area Under ROC Curve) via trapezoidal rule
pub fn auc(points: &[(f64, f64)]) -> f64 {
    if points.len() < 2 { return 0.0; }
    let mut area = 0.0;
    for w in points.windows(2) {
        area += (w[1].0 - w[0].0) * (w[1].1 + w[0].1) / 2.0;
    }
    area.abs()
}

/// Neyman-Pearson test: accept signal if LR > threshold
pub fn neyman_pearson_test(x: f64, model: &SignalModel, lr_threshold: f64) -> bool {
    likelihood_ratio(x, model) > lr_threshold
}

/// Multi-hypothesis deadband: N signal sources, find optimal thresholds for each
pub fn multi_source_deadband(models: &[SignalModel], costs: &[CostModel]) -> Vec<OptimalGap> {
    models.iter().zip(costs.iter())
        .map(|(m, c)| optimal_deadband(m, c))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimal_gap_balances_costs() {
        let model = SignalModel::default();
        let balanced = CostModel { fp_cost: 1.0, fn_cost: 1.0, signal_prior: 0.5 };
        let gap = optimal_deadband(&model, &balanced);
        // Threshold should be near midpoint (1.5) for balanced costs
        assert!((gap.lower - 1.5).abs() < 0.5, "Threshold {} far from midpoint", gap.lower);
    }

    #[test]
    fn asymmetric_costs_shift_threshold() {
        let model = SignalModel::default();
        let fp_cheap = CostModel { fp_cost: 0.1, fn_cost: 10.0, signal_prior: 0.5 };
        let fn_cheap = CostModel { fp_cost: 10.0, fn_cost: 0.1, signal_prior: 0.5 };
        let gap1 = optimal_deadband(&model, &fp_cheap);
        let gap2 = optimal_deadband(&model, &fn_cheap);
        // When FN is expensive, threshold moves lower (catch more signals)
        assert!(gap1.lower < gap2.lower, "FN-expensive {} should be < FP-expensive {}", gap1.lower, gap2.lower);
    }

    #[test]
    fn roc_auc_perfect_separation() {
        let model = SignalModel { noise_mean: 0.0, noise_std: 0.1, signal_mean: 10.0, signal_std: 0.1 };
        let roc = roc_curve(&model, 100);
        let area = auc(&roc);
        assert!(area > 0.99, "AUC = {}, expected ~1.0 for perfect separation", area);
    }

    #[test]
    fn roc_auc_overlapping() {
        let model = SignalModel { noise_mean: 0.0, noise_std: 1.0, signal_mean: 0.5, signal_std: 1.0 };
        let roc = roc_curve(&model, 100);
        let area = auc(&roc);
        // Overlapping distributions should give AUC between 0.5 and 1.0
        assert!(area > 0.5 && area < 0.95, "AUC = {} for overlapping", area);
    }

    #[test]
    fn likelihood_ratio_direction() {
        let model = SignalModel::default();
        let lr_noise = likelihood_ratio(model.noise_mean, &model);
        let lr_signal = likelihood_ratio(model.signal_mean, &model);
        assert!(lr_signal > lr_noise, "LR at signal mean should exceed LR at noise mean");
    }

    #[test]
    fn neyman_pearson_decision() {
        let model = SignalModel::default();
        // High threshold: only strong signals pass
        assert!(!neyman_pearson_test(1.0, &model, 100.0)); // near noise, rejected
        assert!(neyman_pearson_test(5.0, &model, 100.0));  // deep in signal, accepted
    }

    #[test]
    fn spectral_deadband_computes() {
        let eigs = vec![0.0, 0.5, 1.0, 2.0, 5.0];
        let gap = spectral_deadband(&eigs, 1.0, 1.0);
        // CR = 0.5/5.0 = 0.1, optimal CR = 0.5, gap = 0.4
        assert!((gap - 0.4).abs() < 0.01, "Spectral gap = {}, expected 0.4", gap);
    }

    #[test]
    fn multi_source_works() {
        let models = vec![SignalModel::default(), SignalModel::default()];
        let costs = vec![CostModel::default(), CostModel::default()];
        let gaps = multi_source_deadband(&models, &costs);
        assert_eq!(gaps.len(), 2);
        assert!(gaps[0].expected_cost > 0.0);
    }

    #[test]
    fn phi_bounds() {
        assert!((phi(0.0) - 0.5).abs() < 0.01);
        assert!(phi(-3.0) < 0.01);
        assert!(phi(3.0) > 0.99);
    }
}

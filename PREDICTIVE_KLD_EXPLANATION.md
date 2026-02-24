# Predictive KLD for Bayesian Optimal Experimental Design

## Summary of Changes

I've implemented **predictive KL divergence** (also known as BALD - Bayesian Active Learning by Disagreement) as an alternative acquisition function to the existing parameter-space KLD in `HSA_Cam_B_017_pyro.py`.

## Critique of Your Original Roadmap

Your original plan had a **fundamental conceptual issue**:

### What You Proposed:
1. Generate "prior predictive" of the "old" model
2. Draw hypothetical samples
3. Condition model on new data
4. Compare prior pred. dist. vs posterior pred. dist.

### The Problems:
1. **Terminology confusion**: Your "old" model is already a **posterior** (conditioned on data), not a prior
2. **Wrong comparison**: You can't compare prior vs posterior predictives when both are conditioned on different datasets
3. **Missing test set**: You need to define WHERE you want to measure prediction changes

## The Correct Approach: Predictive KLD (BALD)

### Mathematical Formulation

For a candidate experimental condition **x***, the predictive information gain is:

$$U(x^*) = \mathbb{E}_{y^* \sim p(y^*|x^*, \mathcal{D}_{\text{old}})} \left[ \text{KL}\left( p(y_{\text{test}}|x_{\text{test}}, \mathcal{D}_{\text{new}}) \,||\, p(y_{\text{test}}|x_{\text{test}}, \mathcal{D}_{\text{old}}) \right) \right]$$

Where:
- $\mathcal{D}_{\text{old}}$: Current dataset
- $\mathcal{D}_{\text{new}} = \mathcal{D}_{\text{old}} \cup \{(x^*, y^*)\}$: Augmented dataset
- $y^*$: Hypothetical observation at candidate location
- $x_{\text{test}}$: Test grid where we measure prediction changes

### Intuition

This measures: **"If I observe data at x*, how much will my model's predictions change at test locations?"**

- High value → Observing at x* will significantly change predictions → **Good experimental choice**
- Low value → Observing at x* won't change predictions much → **Redundant experiment**

### Key Differences from Parameter KLD

| Aspect | Parameter KLD | Predictive KLD (BALD) |
|--------|--------------|---------------------|
| **What it measures** | Changes in parameter beliefs | Changes in predictions |
| **Mathematical form** | $\text{KL}(p(\theta\|\mathcal{D}_{\text{new}}) \|\| p(\theta\|\mathcal{D}_{\text{old}}))$ | $\mathbb{E}_{y^*}[\text{KL}(p(y_{\text{test}}\|\mathcal{D}_{\text{new}}) \|\| p(y_{\text{test}}\|\mathcal{D}_{\text{old}}))]$ |
| **Focus** | Parameter uncertainty reduction | Prediction uncertainty reduction |
| **Applicability** | When parameters are interpretable | When predictions matter most |
| **Requires** | Compatible parameter spaces | Test set definition |

## Implementation Details

### New Functions

1. **`predictive_KL_divergence()`**: Sequential version
   - Computes predictive KLD for a single candidate
   - Generates hypothetical observations at candidate
   - Updates posterior for each hypothetical
   - Compares predictions at test grid

2. **`mp_predictive_KL_divergence()`**: Parallel version
   - Parallelizes across hypothetical observations
   - Much faster for large n_hyp_samples

3. **`_single_predictive_kl_worker()`**: Worker function
   - Top-level function for multiprocessing
   - Handles single hypothetical observation

4. **`generate_test_grid()`**: Test set generator
   - Creates Latin Hypercube sample in experimental space
   - Ensures good coverage of operating conditions
   - Default: 20 test points

### New Command-Line Arguments

```bash
python HSA_Cam_B_017_pyro.py \
    --acquisition predictive \  # Use predictive KLD instead of parameter KLD
    --ntestpoints 20 \           # Number of test points for measuring prediction changes
    --workers 12 \
    --KLsamples 12 \
    --iterations 16
```

### Usage Example

```python
# Generate test grid (where to measure prediction changes)
x_test = generate_test_grid(
    n_points=20,
    T_range=(500.0, 600.0),
    p_range=(2.0, 20.0),
    c2h_range=(0.5, 2.0),
    f_range=(10.0, 1000.0),
    c2i_range=(0.1, 1.0),
)

# Compute predictive KLD for a candidate
klds = mp_predictive_KL_divergence(
    x_old=proposals,          # Current experimental conditions
    y_old=r_obs,             # Current observations
    x_new=candidate,         # Candidate to evaluate
    x_test=x_test,           # Test grid
    idata_old=idata,         # Current posterior
    n_hyp_samples=12,        # Hypothetical observations to average
    n_pred_samples=500,      # Samples for predictive distributions
    n_cores=12,              # Parallel workers
    use_vi=True,             # Use VI for speed
)

# Average across hypothetical obs and test points
mean_kld = klds.mean().item()
```

## Algorithm Workflow

### For Each Candidate x*:

1. **Generate hypothetical observations** at x*:
   ```
   y*_1, y*_2, ..., y*_N ~ p(y*|x*, θ), θ ~ p(θ|D_old)
   ```

2. **For each hypothetical y*_i**:
   
   a. **Update posterior**:
      ```
      p(θ|D_old, x*, y*_i)
      ```
   
   b. **Generate baseline predictions** at test grid:
      ```
      y_test^(old) ~ p(y_test|x_test, D_old)
      ```
   
   c. **Generate updated predictions**:
      ```
      y_test^(new) ~ p(y_test|x_test, D_old, x*, y*_i)
      ```
   
   d. **Compute KLD at each test point** j:
      ```
      KL_j^(i) = KL(p_new(y_test,j) || p_old(y_test,j))
      ```

3. **Average**:
   ```
   U(x*) = (1/N) Σ_i Σ_j KL_j^(i)
   ```

## When to Use Which Acquisition Function?

### Use **Parameter KLD** when:
- ✅ You care about identifying specific parameter values
- ✅ Parameters have physical/chemical meaning
- ✅ Model is not too complex (parameters are identifiable)
- ✅ You want to reduce parameter uncertainty directly

### Use **Predictive KLD (BALD)** when:
- ✅ You care about prediction accuracy more than parameter values
- ✅ Model might be overparameterized
- ✅ You want experiments that improve predictions across operating space
- ✅ You have specific regions of interest (test set)
- ✅ Parameters are highly correlated (parameter KLD might be misleading)

### For Your Application (Kinetics):
I recommend **trying both**! 
- **Parameter KLD**: Good if you want to pin down activation energies, adsorption constants, etc.
- **Predictive KLD**: Good if you want accurate rate predictions across all conditions

## Computational Considerations

### Predictive KLD is MORE expensive than Parameter KLD:

**Parameter KLD**:
- Complexity: O(n_hyp × n_params)
- Typical: ~12 hyp samples × 9 parameters = 108 KL computations

**Predictive KLD**:
- Complexity: O(n_hyp × n_test × n_pred_samples)
- Typical: ~12 hyp samples × 20 test points × 500 pred samples = much more expensive!

### Optimization Tips:

1. **Use VI instead of MCMC**: Set `use_vi=True` (10x faster)
2. **Reduce test points**: Start with `--ntestpoints 10`
3. **Reduce prediction samples**: 500 is reasonable, don't go higher
4. **Parallelize**: Use all available cores with `--workers`
5. **Strategic test grid**: Place test points in regions you care about

## Theoretical Background

### Why This Works

The predictive KLD approximates the **Expected Information Gain** about predictions:

$$I(y_{\text{test}}; y^* | x^*, x_{\text{test}}, \mathcal{D}_{\text{old}})$$

This is the mutual information between:
- Future test predictions $y_{\text{test}}$
- Hypothetical observation $y^*$ at candidate $x^*$

### Connection to BALD

This is equivalent to **Bayesian Active Learning by Disagreement**:
- Experiments that cause maximum "disagreement" between posterior predictives
- Disagreement quantified by KL divergence
- Averages over epistemic uncertainty (hypothetical outcomes)

### Advantages Over Other Methods

Compared to other acquisition functions:

| Method | Pros | Cons |
|--------|------|------|
| **Variance reduction** | Fast, simple | Ignores information content |
| **Parameter KLD** | Principled, parameter-focused | May not improve predictions |
| **Predictive KLD (this)** | Prediction-focused, principled | More expensive |
| **Mutual information** | Optimal in theory | Hard to compute exactly |

## References

1. **BALD**: Houlsby et al. (2011) "Bayesian Active Learning by Disagreement"
2. **Bayesian OED**: Chaloner & Verdinelli (1995) "Bayesian Experimental Design: A Review"
3. **Predictive Information**: MacKay (1992) "Information-Based Objective Functions"

## Next Steps

1. **Test both acquisition functions** on your problem
2. **Compare results**: Does predictive KLD find different experiments?
3. **Analyze test grid sensitivity**: How does choice of test points affect results?
4. **Computational profiling**: Measure actual runtimes
5. **Scientific validation**: Which gives better parameter estimates / predictions?

## Code Integration

The changes are **backward compatible**. Your existing code will still work with parameter KLD (default). To switch to predictive KLD:

```bash
# Old behavior (parameter KLD)
python HSA_Cam_B_017_pyro.py --workers 12 --KLsamples 12

# New behavior (predictive KLD)
python HSA_Cam_B_017_pyro.py --workers 12 --KLsamples 12 --acquisition predictive
```

The main optimization loops (`sp_find_opt_candidate`, `mp_find_opt_candidate`) remain unchanged - they just call different KLD functions based on the `--acquisition` flag.

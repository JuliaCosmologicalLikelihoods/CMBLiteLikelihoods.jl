# CMBLiteLikelihoods.jl

[![Build Status](https://github.com/JuliaCosmologicalLikelihoods/CMBLiteLikelihoods.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaCosmologicalLikelihoods/CMBLiteLikelihoods.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`CMBLiteLikelihoods.jl` is a high-performance, native Julia implementation of lightweight CMB and CMB lensing likelihoods. It features first-class support for **just-in-time (JIT) compilation** via [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) (XLA compilation targeting CPU, GPU, and TPU backends) and **reverse-mode automatic differentiation (AD)** using [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl).

---

## Features

* **High-Performance CMB Likelihoods**:
  * `PlanckLowTT`: Low-$\ell$ TT Planck likelihood.
  * `SROLL2`: Low-$\ell$ EE Planck likelihood (SROLL2).
  * `CamSpecLite`: High-$\ell$ Planck TT/TE/EE likelihood.
  * `ACTDR6CMBOnly`: High-$\ell$ ACT DR6 CMB-only likelihood.
  * `SPT3GD1Lite`: High-$\ell$ SPT-3G TT/TE/EE likelihood.
* **Combined Likelihood**: 
  * `joint_chi2`: A convenience function combining all individual likelihoods with custom scale cuts (e.g., Planck up to $\ell_{\text{TT}}=1500$, $\ell_{\text{TE}}=1000$, $\ell_{\text{EE}}=600$, with ACT and SPT-3G treated as independent above those cuts).
* **CMB Lensing with Active Runtime Corrections**:
  * `ACTPlanckSPTLensing`: Combined ACT, Planck, and SPT-3G lensing likelihood.
  * Supports **active runtime corrections** (normalization matrices and $N_1$ corrections) computed via pre-cached derivative expansion matrices, matching the reference Python implementation.
* **Hardware Acceleration & JIT Compilation**: Fully compatible with `Reactant.jl` to compile execution graphs to XLA.
* **Adjoint Automatic Differentiation**: Compatible with `DifferentiationInterface.jl` (via `ForwardDiff` and `FiniteDiff`) and direct `Enzyme` reverse-mode gradients on XLA compiled graphs.

---

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaCosmologicalLikelihoods/CMBLiteLikelihoods.jl")
```

---

## Usage

### 1. Evaluating CMB Lensing with Active Corrections

```julia
using CMBLiteLikelihoods

# Initialize likelihood with active CMB corrections
like = ACTPlanckSPTLensing(load_corrections=true)

# Dummy cosmological spectra (length 4001, matching ell = 0:4000)
ell = 0:4000
cl_pp = [l <= 1 ? 0.0 : 1e-2 / ((l + 10.0) * (l + 11.0))^3 for l in ell]
cl_tt = [l <= 1 ? 0.0 : 2000.0 * exp(-(l - 200.0)^2 / 20000.0) for l in ell]
cl_ee = [l <= 1 ? 0.0 : 0.05 * exp(-(l - 200.0)^2 / 20000.0) for l in ell]
cl_te = [l <= 1 ? 0.0 : 100.0 * exp(-(l - 300.0)^2 / 25000.0) for l in ell]
cl_bb = zeros(4001)

# Compute chi-squared value
χ² = chi2(like, cl_pp, cl_tt, cl_ee, cl_te, cl_bb)
```

### 2. JIT Compilation with Reactant

```julia
using Reactant

# Create Reactant input parameters
p_val = [1.0, 1.0, 1.0, 1.0]
p_reactant = Reactant.to_rarray((p_val...,); track_numbers=true)

# Define objective function
function loss(p)
    cl_pp_scaled = cl_pp .* p[1]
    cl_tt_scaled = cl_tt .* p[2]
    cl_ee_scaled = cl_ee .* p[3]
    cl_te_scaled = cl_te .* p[4]
    return chi2(like, cl_pp_scaled, cl_tt_scaled, cl_ee_scaled, cl_te_scaled, cl_bb)
end

# Compile the execution graph to XLA
compiled_loss = Reactant.compile(loss, (p_reactant,))

# Run compiled version on XLA (CPU/GPU/TPU)
val = compiled_loss(p_reactant)
```

### 3. Reverse-Mode Differentiation with Enzyme

```julia
using Enzyme

# Compile gradient calculation using Enzyme on XLA
g_enzyme = Reactant.compile(p -> Enzyme.gradient(Reverse, loss, p)[1], (p_reactant,))

# Compute gradient vector (near-zero allocations)
grad = g_enzyme(p_reactant)
```

---

## Performance

The table below shows CPU benchmark timings for the binned `ACTPlanckSPTLensing` likelihood with active CMB corrections ($N_1$ and normalization corrections):

| Operation | Implementation | Time (ms) | Memory Allocations | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Evaluation** | Plain Julia | $26.89\text{ ms}$ | $1.95\text{ MiB}$ (270 allocs) | *Reference* |
| | Reactant (CPU XLA) | **$3.17\text{ ms}$** | **$400\text{ B}$** (14 allocs) | **$8.48\times$** |
| **Gradient** | Plain Julia (`ForwardDiff`) | $262.62\text{ ms}$ | $9.34\text{ MiB}$ (278 allocs) | *Reference* |
| | Reactant + `Enzyme` | **$6.42\text{ ms}$** | **$688\text{ B}$** (26 allocs) | **$40.94\times$** |

### Key Performance Benefits:
1. **$41\times$ Gradient Speedup**: Adjoint/reverse-mode differentiation via `Enzyme` on the compiled execution graph is over 40 times faster than forward-mode differentiation (`ForwardDiff`) in plain Julia.
2. **Zero Allocation Overhead**: Memory allocation drops from megabytes to **bytes**, preventing garbage collection slowdowns in nested parameter sampling routines (MCMC, nested sampling, etc.).
3. **Accelerator Native**: Automatic translation to CPU, GPU, or TPU kernels via the XLA compiler backend.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
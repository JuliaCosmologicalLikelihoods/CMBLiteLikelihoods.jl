# CMBLiteLikelihoods.jl

[![Build Status](https://github.com/JuliaCosmologicalLikelihoods/CMBLiteLikelihoods.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaCosmologicalLikelihoods/CMBLiteLikelihoods.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`CMBLiteLikelihoods.jl` is a high-performance, native Julia implementation of lightweight CMB likelihoods (foreground-marginalized, CMB-only). It features first-class support for **just-in-time (JIT) compilation** via [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) (targeting XLA CPU/GPU/TPU backends) and **automatic differentiation (AD)** using backends like [Zygote.jl](https://github.com/FluxML/Zygote.jl), [Mooncake.jl](https://github.com/compmonks/Mooncake.jl), and [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl).

---

## Features

* **High-Performance CMB Likelihoods**:
  * `PlanckLowTT`: Low-$\ell$ TT Planck likelihood (converted to purely functional patterns to support Zygote AD).
  * `SROLL2`: Low-$\ell$ EE Planck likelihood (SROLL2).
  * `CamSpecLite`: High-$\ell$ Planck TT/TE/EE likelihood.
  * `ACTDR6CMBOnly`: High-$\ell$ ACT DR6 CMB-only likelihood.
  * `SPT3GD1Lite`: High-$\ell$ SPT-3G TT/TE/EE likelihood.
* **Combined Likelihood**: 
  * `joint_chi2`: A convenience function combining all individual likelihoods with custom scale cuts (Planck up to $\ell_{\text{TT}}=1500$, $\ell_{\text{TE}}=1000$, $\ell_{\text{EE}}=600$, with ACT and SPT-3G treated as independent above those cuts).
* **Hardware Acceleration & JIT Compilation**: Fully compatible with `Reactant.jl` to compile execution graphs to XLA.
* **Full AD Compatibility**: Seamlessly integrated with `DifferentiationInterface.jl` to allow gradient evaluations via Zygote, Mooncake, ForwardDiff, FiniteDiff, and Reactant + Enzyme.

---

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaCosmologicalLikelihoods/CMBLiteLikelihoods.jl")
```

---

## Usage

### 1. Evaluating Combined Joint CMB Likelihood

```julia
using CMBLiteLikelihoods

# Define dummy spectra NamedTuple starting at ℓ = 0
ell = 0:3000
cl_tt = [l <= 1 ? 0.0 : 2000.0 * exp(-(l - 200.0)^2 / 20000.0) for l in ell]
cl_ee = [l <= 1 ? 0.0 : 0.05 * exp(-(l - 200.0)^2 / 20000.0) for l in ell]
cl_te = [l <= 1 ? 0.0 : 100.0 * exp(-(l - 300.0)^2 / 25000.0) for l in ell]
Dls = (TT=cl_tt, EE=cl_ee, TE=cl_te)

# Define nuisance parameters
params = (
    A_planck=1.0, calTE=1.0, calEE=1.0, 
    A_act=1.0, P_act=1.0, 
    Tcal=1.0, Ecal=1.0
)

# Compute joint chi-squared value
χ² = joint_chi2(Dls, params)
```

### 2. JIT Compilation with Reactant

```julia
using Reactant

# Create Reactant input parameters
p_val = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
p_reactant = Reactant.to_rarray((p_val...,); track_numbers=true)

# Define objective function
function loss(p)
    scaled_params = (
        A_planck=p[1], calTE=p[2], calEE=p[3], 
        A_act=p[4], P_act=p[5], 
        Tcal=p[6], Ecal=p[7]
    )
    return joint_chi2(Dls, scaled_params)
end

# Compile the execution graph to XLA
compiled_loss = Reactant.compile(loss, (p_reactant,))

# Run compiled version on XLA (CPU/GPU/TPU)
val = compiled_loss(p_reactant)
```

### 3. Gradient Evaluation using DifferentiationInterface (Zygote & Mooncake)

You can compute gradients using any AD backend wrapped by `DifferentiationInterface.jl`.

```julia
using DifferentiationInterface
using Zygote
using Mooncake

# Define a function mapping parameters vector to the joint chi-squared value
function f_loss(p)
    scaled_params = (
        A_planck=p[1], calTE=p[2], calEE=p[3], 
        A_act=p[4], P_act=p[5], 
        Tcal=p[6], Ecal=p[7]
    )
    return joint_chi2(Dls, scaled_params)
end

p_vec = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Compute gradient using Zygote
grad_zygote = DifferentiationInterface.gradient(f_loss, AutoZygote(), p_vec)

# Compute gradient using Mooncake
grad_mooncake = DifferentiationInterface.gradient(f_loss, AutoMooncake(), p_vec)
```

### 4. Reverse-Mode Differentiation with Reactant + Enzyme

For maximum performance on hardware accelerators, compile the gradient calculation to XLA using Reactant + Enzyme:

```julia
using Enzyme

# Compile gradient calculation using Enzyme on XLA
g_enzyme = Reactant.compile(p -> Enzyme.gradient(Reverse, loss, p)[1], (p_reactant,))

# Compute gradient vector (near-zero allocations)
grad = g_enzyme(p_reactant)
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
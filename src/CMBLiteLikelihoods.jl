"""
    CMBLiteLikelihoods

Native Julia implementation of lite CMB likelihoods (foreground-marginalized,
CMB-only) for cosmological parameter estimation.

Supported likelihoods:
- `CamSpecLite` — Planck PR4/NPIPE CamSpec lite (TT/TE/EE, integer-ℓ)
- `ACTDR6CMBOnly` — ACT DR6 CMB-only lite (TT/TE/EE, windowed)
- `SPT3GD1Lite` — SPT-3G D1 TnE lite (TT/TE/EE, aberration + windowed)
- `PlanckLowTT` — Planck 2018 low-ℓ TT (Gaussianized Gibbs)
- `SROLL2` — Planck 2018 low-ℓ EE SROLL2 (table lookup)

All likelihoods expect theory spectra as D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) in μK²,
starting at ℓ = 0 (i.e. `Dls.TT[1]` corresponds to ℓ = 0).
"""
module CMBLiteLikelihoods

import AbstractCosmologicalEmulators
using LinearAlgebra
using NPZ
using Artifacts

# Common utilities
include("gaussian.jl")
include("aberration.jl")

# High-ℓ likelihoods
include("camspec.jl")
include("act.jl")
include("spt.jl")

# Low-ℓ likelihoods
include("lowtt.jl")
include("sroll2.jl")


# Refs for pre-loaded likelihoods
const CAMSPEC_LITE = Ref{CamSpecLite}()
const ACT_DR6_CMBONLY = Ref{ACTDR6CMBOnly}()
const SPT3G_D1_LITE = Ref{SPT3GD1Lite}()
const PLANCK_LOW_TT = Ref{PlanckLowTT}()
const SROLL2_LITE = Ref{SROLL2}()

const CAMSPEC_LITE_SUB = Ref{CamSpecLite}()
const ACT_DR6_CMBONLY_SUB = Ref{ACTDR6CMBOnly}()
const SPT3G_D1_LITE_SUB = Ref{SPT3GD1Lite}()

function __init__()
    dir = joinpath(artifact"cmblite_data", "cmblite_data")
    CAMSPEC_LITE[] = CamSpecLite(joinpath(dir, "camspec_npipe_lite"))
    ACT_DR6_CMBONLY[] = ACTDR6CMBOnly(joinpath(dir, "act_dr6_cmbonly"))
    SPT3G_D1_LITE[] = SPT3GD1Lite(joinpath(dir, "spt3g_d1_tne_lite"))
    PLANCK_LOW_TT[] = PlanckLowTT(joinpath(dir, "planck_2018_lowl_TT"))
    SROLL2_LITE[] = SROLL2(joinpath(dir, "planck_2018_lowl_EE_sroll2"))

    # Subsetted versions for joint_chi2
    CAMSPEC_LITE_SUB[] = CamSpecLite(joinpath(dir, "camspec_npipe_lite"); tt_max=1500, te_max=1000, ee_max=600)
    ACT_DR6_CMBONLY_SUB[] = ACTDR6CMBOnly(joinpath(dir, "act_dr6_cmbonly"); tt_min=1501, te_min=1001, ee_min=601)
    SPT3G_D1_LITE_SUB[] = SPT3GD1Lite(joinpath(dir, "spt3g_d1_tne_lite"); tt_min=1501, te_min=1001, ee_min=601)
end

# Default constructors accessing the pre-loaded instances
CamSpecLite() = CAMSPEC_LITE[]
ACTDR6CMBOnly() = ACT_DR6_CMBONLY[]
SPT3GD1Lite() = SPT3G_D1_LITE[]
PlanckLowTT() = PLANCK_LOW_TT[]
SROLL2() = SROLL2_LITE[]

"""
    joint_chi2(Dls, params) -> Real

Compute a single joint χ² sum over all likelihood components with scale cuts:
- Low-ℓ: Planck Low-ℓ TT (`PlanckLowTT`) and SROLL2 Low-ℓ EE (`SROLL2`) on their full range.
- High-ℓ: Planck/CamSpec up to ℓ_max: 1500 (TT), 1000 (TE), 600 (EE).
- Above these thresholds, ACT and SPT are summed together (assumed independent).

# Arguments
- `Dls`: NamedTuple of theory spectra `(TT=..., TE=..., EE=...)` starting at ℓ=0.
- `params`: Nuisance parameters NamedTuple/struct (e.g. `(A_planck=1.0, calTE=1.0, calEE=1.0, A_act=1.0, P_act=1.0, Tcal=1.0, Ecal=1.0)`).
"""
function joint_chi2(Dls, params)
    # Sum low-ℓ χ²
    c_lowtt = chi2(PLANCK_LOW_TT[], Dls, params)
    c_sroll2 = chi2(SROLL2_LITE[], Dls, params)

    # Sum high-ℓ χ² (using subsetted versions)
    c_camspec = chi2(CAMSPEC_LITE_SUB[], Dls, params)
    c_act = chi2(ACT_DR6_CMBONLY_SUB[], Dls, params)
    c_spt = chi2(SPT3G_D1_LITE_SUB[], Dls, params)

    return c_lowtt + c_sroll2 + c_camspec + c_act + c_spt
end

# Exports
export CamSpecLite, ACTDR6CMBOnly, SPT3GD1Lite
export PlanckLowTT, SROLL2
export loglike, chi2, build_model_vector
export apply_aberration, gaussian_chi2
export joint_chi2

end # module CMBLiteLikelihoods

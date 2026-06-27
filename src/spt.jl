"""
    spt.jl — SPT-3G D1 TnE lite likelihood

Foreground-marginalized, CMB-only. Applies aberration correction before
calibration and window binning.

Calibration model:
    TT: D_ℓ / Tcal²
    TE: D_ℓ / (Tcal² × Ecal)
    EE: D_ℓ / (Tcal² × Ecal²)

Data vector layout: [TT (52 bins), TE (72 bins), EE (72 bins)] concatenated.
Window matrices are stored as (n_bin, n_ell_theory) on grid ℓ = 2..4095.
"""

"""
    SPT3GD1Lite

Pre-computed data structure for the SPT-3G D1 TnE lite Gaussian likelihood.

# Fields
- `data_vector::Vector{Float64}`: observed band powers (length 196)
- `cov_chol::Cholesky`: Cholesky factorisation of the covariance
- `windows::Vector{Matrix{Float64}}`: window matrices [TT, TE, EE]
- `theory_ell::Vector{Int}`: theory multipoles (2..4095)
- `aberration_coeff::Float64`: aberration coefficient AC = β⟨cosθ⟩
- `n_bins::Vector{Int}`: bins per spectrum [52, 72, 72]
"""
struct SPT3GD1Lite
    data_vector::Vector{Float64}
    cov_chol::Cholesky{Float64, Matrix{Float64}}
    windows::Vector{Matrix{Float64}}   # [TT, TE, EE]
    theory_ell::Vector{Int}             # 2..4095
    aberration_coeff::Float64           # -0.0004826
    n_bins::Vector{Int}                 # [52, 72, 72]
end

"""
    SPT3GD1Lite(data_dir::AbstractString)

Load the SPT-3G D1 TnE lite likelihood from a directory of `.npy` files.
"""
function SPT3GD1Lite(data_dir::AbstractString; tt_min::Integer=0, te_min::Integer=0, ee_min::Integer=0)
    data = npzread(joinpath(data_dir, "data_vector.npy"))
    cov = npzread(joinpath(data_dir, "covariance.npy"))
    theory_ell = npzread(joinpath(data_dir, "theory_ell.npy"))
    eff_ell = npzread(joinpath(data_dir, "effective_ell.npy"))

    # Bins per spectrum in original SPT: [52, 72, 72]
    # Bins are ordered: TT (1..52), TE (53..124), EE (125..196)
    spec_id = Vector{Int}(undef, 196)
    spec_id[1:52] .= 1
    spec_id[53:124] .= 2
    spec_id[125:196] .= 3

    # Keep indices
    keep_indices = Int[]
    for i in 1:length(data)
        ℓ = eff_ell[i]
        si = spec_id[i]
        if (si == 1 && ℓ >= tt_min) || (si == 2 && ℓ >= te_min) || (si == 3 && ℓ >= ee_min)
            push!(keep_indices, i)
        end
    end

    data_sub = data[keep_indices]
    cov_sub = cov[keep_indices, keep_indices]

    win_files = ["window_TT_lxl.npy", "window_TE_lxl.npy", "window_EE_lxl.npy"]
    windows = Matrix{Float64}[]
    
    # We can determine which bins of TT/TE/EE were kept
    offsets = [0, 52, 124]
    counts = [52, 72, 72]

    for i in 1:3
        offset = offsets[i]
        orig_indices = offset .+ (1:counts[i])
        kept_sub_indices = [idx - offset for idx in orig_indices if idx in keep_indices]

        W = npzread(joinpath(data_dir, win_files[i]))
        W_sub = W[kept_sub_indices, :]
        push!(windows, Float64.(W_sub))
    end

    n_bins = [size(W, 1) for W in windows]
    cov_chol = cholesky(Symmetric(cov_sub, :L))

    # Aberration coefficient (SPT-3G default: -0.0004826)
    AC = -0.0004826

    return SPT3GD1Lite(
        Float64.(data_sub),
        cov_chol,
        windows,
        Int.(theory_ell),
        AC,
        n_bins,
    )
end

"""
    build_model_vector(like::SPT3GD1Lite, Dls, params)

Build the model band-power vector.

Pipeline:
1. Apply aberration correction to each unbinned theory D_ℓ
2. Apply calibration (Tcal, Ecal)
3. Bin with window matrices

# Arguments
- `like`: `SPT3GD1Lite` struct
- `Dls`: `NamedTuple` with keys `:TT`, `:TE`, `:EE`, each starting at ℓ=0
- `params`: `NamedTuple` with keys `:Tcal`, `:Ecal`

# Returns
- `Vector{Float64}` model vector, same length as `like.data_vector`
"""
function build_model_vector(like::SPT3GD1Lite, Dls, params)
    ell = like.theory_ell
    AC = like.aberration_coeff
    Tcal = getproperty(params, :Tcal)
    Ecal = getproperty(params, :Ecal)

    # Calibration denominators
    Tcal2 = Tcal * Tcal
    cal = (Tcal2,                       # TT
           Tcal2 * Ecal,                # TE
           Tcal2 * Ecal * Ecal)         # EE

    # TT
    Dl_cut_tt = Dls.TT[ell .+ 1]
    Dl_ab_tt = apply_aberration(Dl_cut_tt, ell; AC=AC)
    binned_tt = like.windows[1] * (Dl_ab_tt ./ cal[1])

    # TE
    Dl_cut_te = Dls.TE[ell .+ 1]
    Dl_ab_te = apply_aberration(Dl_cut_te, ell; AC=AC)
    binned_te = like.windows[2] * (Dl_ab_te ./ cal[2])

    # EE
    Dl_cut_ee = Dls.EE[ell .+ 1]
    Dl_ab_ee = apply_aberration(Dl_cut_ee, ell; AC=AC)
    binned_ee = like.windows[3] * (Dl_ab_ee ./ cal[3])

    return vcat(binned_tt, binned_te, binned_ee)
end

"""
    loglike(like::SPT3GD1Lite, Dls, params) -> Float64

Compute the Gaussian log-likelihood: -0.5 × χ².
"""
function loglike(like::SPT3GD1Lite, Dls, params)
    model = build_model_vector(like, Dls, params)
    return gaussian_loglike(like.cov_chol, like.data_vector, model)
end

"""
    chi2(like::SPT3GD1Lite, Dls, params) -> Float64
"""
function chi2(like::SPT3GD1Lite, Dls, params)
    model = build_model_vector(like, Dls, params)
    return gaussian_chi2(like.cov_chol, like.data_vector .- model)
end

function Base.show(io::IO, like::SPT3GD1Lite)
    n = length(like.data_vector)
    println(io, "SPT3GD1Lite")
    println(io, "  Bins: TT=$(like.n_bins[1]), TE=$(like.n_bins[2]), EE=$(like.n_bins[3]) = $n total")
    println(io, "  Theory ℓ: $(like.theory_ell[1]) – $(like.theory_ell[end])")
    print(io,   "  Aberration coeff: $(like.aberration_coeff)")
end

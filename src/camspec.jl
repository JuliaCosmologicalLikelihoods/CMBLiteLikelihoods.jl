"""
    camspec.jl — Planck PR4/NPIPE CamSpec lite likelihood

Foreground-marginalized, CMB-only. No bandpower windows — the SACC windows
are identity selections at integer multipoles, so theory D_ℓ values are used
directly.

Calibration model:
    TT: D_ℓ / A_planck²
    TE: D_ℓ / (calTE × A_planck²)
    EE: D_ℓ / (calEE × A_planck²)

Data vector layout: [TT bins, TE bins, EE bins] concatenated.
"""

"""
    CamSpecLite

Pre-computed data structure for the CamSpec NPIPE lite Gaussian CMB likelihood.

# Fields
- `data_vector::Vector{Float64}`: observed band powers
- `cov_chol::Cholesky`: Cholesky factorisation of the covariance
- `ell::Vector{Int}`: integer multipoles, one per data point
- `spec_id::Vector{Int}`: spectrum index (1=TT, 2=TE, 3=EE) per data point
- `n_bins::Vector{Int}`: number of bins per spectrum [N_TT, N_TE, N_EE]
"""
struct CamSpecLite
    data_vector::Vector{Float64}
    cov_chol::Cholesky{Float64, Matrix{Float64}}
    ell::Vector{Int}
    spec_id::Vector{Int}       # 1=TT, 2=TE, 3=EE
    n_bins::Vector{Int}        # [N_TT, N_TE, N_EE]
    ell_tt::Vector{Int}
    ell_te::Vector{Int}
    ell_ee::Vector{Int}
end

"""
    CamSpecLite(data_dir::AbstractString)

Load the CamSpec NPIPE lite likelihood from a directory of `.npy` files.

Expected files:
- `data_vector.npy`
- `covariance.npy`
- `ell.npy`
- `spectrum.npy` (string array: "TT", "TE", "EE")
"""
function CamSpecLite(data_dir::AbstractString; tt_max::Integer=99999, te_max::Integer=99999, ee_max::Integer=99999)
    data = npzread(joinpath(data_dir, "data_vector.npy"))
    cov = npzread(joinpath(data_dir, "covariance.npy"))
    ell_raw = npzread(joinpath(data_dir, "ell.npy"))
    spec_raw = npzread(joinpath(data_dir, "spectrum_id.npy"))

    # spectrum_id: 0=TT, 1=TE, 2=EE → convert to 1-based
    spec_id = Int.(spec_raw) .+ 1

    # Filter indices based on scale cuts
    keep_indices = Int[]
    for i in 1:length(data)
        ℓ = ell_raw[i]
        si = spec_id[i]
        if (si == 1 && ℓ <= tt_max) || (si == 2 && ℓ <= te_max) || (si == 3 && ℓ <= ee_max)
            push!(keep_indices, i)
        end
    end

    data_sub = data[keep_indices]
    cov_sub = cov[keep_indices, keep_indices]
    ell_sub = ell_raw[keep_indices]
    spec_id_sub = spec_id[keep_indices]

    # Precompute indices for vectorized build_model_vector
    idx_tt = findall(==(1), spec_id_sub)
    idx_te = findall(==(2), spec_id_sub)
    idx_ee = findall(==(3), spec_id_sub)

    ell_tt = ell_sub[idx_tt] .+ 1
    ell_te = ell_sub[idx_te] .+ 1
    ell_ee = ell_sub[idx_ee] .+ 1

    # Count bins per spectrum
    n_bins = [count(==(i), spec_id_sub) for i in 1:3]

    cov_chol = cholesky(Symmetric(cov_sub, :L))

    return CamSpecLite(
        Float64.(data_sub),
        cov_chol,
        Int.(ell_sub),
        spec_id_sub,
        n_bins,
        Int.(ell_tt),
        Int.(ell_te),
        Int.(ell_ee),
    )
end

"""
    build_model_vector(like::CamSpecLite, Dls, params)

Build the model band-power vector from theory D_ℓ and nuisance parameters.

# Arguments
- `like`: `CamSpecLite` struct
- `Dls`: `NamedTuple` with keys `:TT`, `:TE`, `:EE`, each starting at ℓ=0
- `params`: `NamedTuple` with keys `:A_planck`, `:calTE`, `:calEE`

# Returns
- `Vector` model vector, same length as `like.data_vector`
"""
function build_model_vector(like::CamSpecLite, Dls, params)
    A = params.A_planck^2
    calTE = getproperty(params, :calTE)
    calEE = getproperty(params, :calEE)

    model_tt = Dls.TT[like.ell_tt] ./ A
    model_te = Dls.TE[like.ell_te] ./ (calTE * A)
    model_ee = Dls.EE[like.ell_ee] ./ (calEE * A)

    return vcat(model_tt, model_te, model_ee)
end

"""
    loglike(like::CamSpecLite, Dls, params) -> Float64

Compute the Gaussian log-likelihood: -0.5 × χ².
"""
function loglike(like::CamSpecLite, Dls, params)
    model = build_model_vector(like, Dls, params)
    return gaussian_loglike(like.cov_chol, like.data_vector, model)
end

"""
    chi2(like::CamSpecLite, Dls, params) -> Float64

Compute χ² = δᵀ C⁻¹ δ.
"""
function chi2(like::CamSpecLite, Dls, params)
    model = build_model_vector(like, Dls, params)
    return gaussian_chi2(like.cov_chol, like.data_vector .- model)
end

function Base.show(io::IO, like::CamSpecLite)
    n = length(like.data_vector)
    println(io, "CamSpecLite")
    println(io, "  Bins: TT=$(like.n_bins[1]), TE=$(like.n_bins[2]), EE=$(like.n_bins[3]) = $n total")
    print(io,   "  ℓ range: $(minimum(like.ell)) – $(maximum(like.ell))")
end

"""
    act.jl — ACT DR6 CMB-only lite likelihood

Foreground-marginalized, CMB-only. Uses nontrivial SACC bandpower windows
to bin theory D_ℓ into band powers.

Calibration model:
    TT: D_ℓ / A_act²
    TE: D_ℓ / (A_act² × P_act)
    EE: D_ℓ / (A_act² × P_act²)

Data vector layout: [TT (45 bins), TE (45 bins), EE (45 bins)] concatenated.
Window matrices are stored as (n_bin, n_ell_theory).
"""

"""
    ACTDR6CMBOnly

Pre-computed data structure for the ACT DR6 CMB-only lite Gaussian likelihood.

# Fields
- `data_vector::Vector{Float64}`: observed band powers (length 135)
- `cov_chol::Cholesky`: Cholesky factorisation of the covariance
- `windows::Vector{Matrix{Float64}}`: window matrices [TT, TE, EE], each (nbin, nell)
- `window_ell::Vector{Vector{Int}}`: theory ℓ values for each window
- `n_bins::Vector{Int}`: bins per spectrum [45, 45, 45]
"""
struct ACTDR6CMBOnly
    data_vector::Vector{Float64}
    cov_chol::Cholesky{Float64, Matrix{Float64}}
    windows::Vector{Matrix{Float64}}   # [TT, TE, EE]
    window_ell::Vector{Vector{Int}}     # [TT, TE, EE]
    n_bins::Vector{Int}                 # [45, 45, 45]
end

"""
    ACTDR6CMBOnly(data_dir::AbstractString)

Load the ACT DR6 CMB-only lite likelihood from a directory of `.npy` files.
"""
function ACTDR6CMBOnly(data_dir::AbstractString; tt_min::Integer=0, te_min::Integer=0, ee_min::Integer=0)
    data = npzread(joinpath(data_dir, "data_vector.npy"))
    cov = npzread(joinpath(data_dir, "covariance.npy"))
    ell_bin = npzread(joinpath(data_dir, "ell.npy"))
    spec_raw = npzread(joinpath(data_dir, "spectrum_id.npy"))
    spec_id = Int.(spec_raw) .+ 1

    # Keep indices
    keep_indices = Int[]
    for i in 1:length(data)
        ℓ = ell_bin[i]
        si = spec_id[i]
        if (si == 1 && ℓ >= tt_min) || (si == 2 && ℓ >= te_min) || (si == 3 && ℓ >= ee_min)
            push!(keep_indices, i)
        end
    end

    data_sub = data[keep_indices]
    cov_sub = cov[keep_indices, keep_indices]

    # Window files and their ell grids
    win_files = ["window_TT_0_45.npy", "window_TE_45_90.npy", "window_EE_90_135.npy"]
    ell_files = ["window_ell_TT_0_45.npy", "window_ell_TE_45_90.npy", "window_ell_EE_90_135.npy"]

    windows = Matrix{Float64}[]
    window_ell = Vector{Int}[]
    for i in 1:3
        # ACT bins in original data are TT (1..45), TE (46..90), EE (91..135)
        offset = (i - 1) * 45
        orig_indices = offset .+ (1:45)
        kept_sub_indices = [idx - offset for idx in orig_indices if idx in keep_indices]

        W = npzread(joinpath(data_dir, win_files[i]))
        W_sub = W[kept_sub_indices, :]
        push!(windows, Float64.(W_sub))

        we = npzread(joinpath(data_dir, ell_files[i]))
        push!(window_ell, Int.(we))
    end

    n_bins = [size(W, 1) for W in windows]
    cov_chol = cholesky(Symmetric(cov_sub, :L))

    return ACTDR6CMBOnly(
        Float64.(data_sub),
        cov_chol,
        windows,
        window_ell,
        n_bins,
    )
end

"""
    build_model_vector(like::ACTDR6CMBOnly, Dls, params)

Build the model band-power vector.

# Arguments
- `like`: `ACTDR6CMBOnly` struct
- `Dls`: `NamedTuple` with keys `:TT`, `:TE`, `:EE`, each starting at ℓ=0
- `params`: `NamedTuple` with keys `:A_act`, `:P_act`

# Returns
- `Vector{Float64}` model vector, same length as `like.data_vector`
"""
function build_model_vector(like::ACTDR6CMBOnly, Dls, params)
    A2 = params.A_act^2
    P = getproperty(params, :P_act)

    # Calibration denominators
    cal = (A2, A2 * P, A2 * P * P)  # TT, TE, EE

    # Vectorized computation for TT, TE, EE
    dat_tt = Dls.TT[like.window_ell[1] .+ 1] ./ cal[1]
    binned_tt = like.windows[1] * dat_tt

    dat_te = Dls.TE[like.window_ell[2] .+ 1] ./ cal[2]
    binned_te = like.windows[2] * dat_te

    dat_ee = Dls.EE[like.window_ell[3] .+ 1] ./ cal[3]
    binned_ee = like.windows[3] * dat_ee

    return vcat(binned_tt, binned_te, binned_ee)
end

"""
    loglike(like::ACTDR6CMBOnly, Dls, params) -> Float64

Compute the Gaussian log-likelihood: -0.5 × χ².
"""
function loglike(like::ACTDR6CMBOnly, Dls, params)
    model = build_model_vector(like, Dls, params)
    return gaussian_loglike(like.cov_chol, like.data_vector, model)
end

"""
    chi2(like::ACTDR6CMBOnly, Dls, params) -> Float64
"""
function chi2(like::ACTDR6CMBOnly, Dls, params)
    model = build_model_vector(like, Dls, params)
    return gaussian_chi2(like.cov_chol, like.data_vector .- model)
end

function Base.show(io::IO, like::ACTDR6CMBOnly)
    n = length(like.data_vector)
    println(io, "ACTDR6CMBOnly")
    println(io, "  Bins: TT=$(like.n_bins[1]), TE=$(like.n_bins[2]), EE=$(like.n_bins[3]) = $n total")
    println(io, "  Window ℓ range: $(minimum(like.window_ell[1])) – $(maximum(like.window_ell[1]))")
    print(io,   "  Data vector length: $n")
end

"""
    lowtt.jl — Planck 2018 low-ℓ TT likelihood (Gaussianized Gibbs)

Uses per-multipole cubic splines to transform D_ℓ → x_ℓ, then
evaluates a Gaussian in x-space with a Jacobian correction.

Algorithm:
    For each ℓ = 2..29:
        x_ℓ = cubic_spline(D_ℓ / A_planck²)
        log|dx/dD_ℓ| via cubic spline derivative

    log L = Σ log|dx/dD_ℓ|
            - 0.5 × (x - μ)ᵀ Σ_x⁻¹ (x - μ)
            - offset

Reject if any D_ℓ is outside prior_bounds.
Uses natural cubic splines (C² continuous, second derivative zero at boundaries).
"""

"""
    PlanckLowTT

Pre-computed data for the Planck low-ℓ TT Gaussianized Gibbs likelihood
with cubic spline interpolation.

# Fields
- `ell::Vector{Int}`: multipoles 2..29
- `x_splines::Vector{Tuple{Vector{Float64},Vector{Float64}}}`: (y_vals, x_grid) for D_ℓ → x_ℓ transform, one per ℓ
- `deriv_splines::Vector{Tuple{Vector{Float64},Vector{Float64}}}`: (y_vals, x_grid) for dx/dD_ℓ derivative, one per ℓ
- `mean_x::Vector{Float64}`: mean in x-space, shape (28,)
- `prec_x::Matrix{Float64}`: precision matrix in x-space, shape (28, 28)
- `prior_bounds::Matrix{Float64}`: valid D_ℓ range [lo, hi] per ℓ, shape (28, 2)
- `offset::Float64`: normalization offset
"""
struct PlanckLowTT
    ell::Vector{Int}
    x_splines::Vector{CubicSpline}
    deriv_splines::Vector{CubicSpline}
    mean_x::Vector{Float64}
    prec_x::Matrix{Float64}
    prior_bounds::Matrix{Float64}
    offset::Float64
end

"""
    PlanckLowTT(data_dir::AbstractString)

Load the Planck low-ℓ TT likelihood from a directory of `.npy` files.
Constructs cubic splines for smooth D_ℓ → x_ℓ transformation.
"""
function PlanckLowTT(data_dir::AbstractString)
    ell = npzread(joinpath(data_dir, "ell.npy"))
    cl_grid = npzread(joinpath(data_dir, "spline_cl_grid_Dell.npy"))
    x_grid = npzread(joinpath(data_dir, "spline_x_grid.npy"))
    mean_x = npzread(joinpath(data_dir, "gaussianized_mean_x.npy"))
    prec_x = npzread(joinpath(data_dir, "gaussianized_precision_x.npy"))
    prior_bounds = npzread(joinpath(data_dir, "prior_bounds_Dell.npy"))
    offset = npzread(joinpath(data_dir, "normalization_offset.npy"))

    n_ell = length(ell)

    # Store spline data as CubicSpline structs
    x_splines = Vector{CubicSpline}(undef, n_ell)
    deriv_splines = Vector{CubicSpline}(undef, n_ell)

    for i in 1:n_ell
        cl_col = collect(cl_grid[:, i])
        x_col = collect(x_grid[:, i])

        # Spline data: D_ℓ → x_ℓ  (u=y_vals=x, t=x_grid=D_ℓ)
        x_splines[i] = CubicSpline(x_col, cl_col)

        # Derivative via finite differences
        dx = diff(x_col) ./ diff(cl_col)
        mid_cl = (cl_col[1:end-1] .+ cl_col[2:end]) ./ 2
        deriv_splines[i] = CubicSpline(dx, mid_cl)
    end

    return PlanckLowTT(
        Int.(ell),
        x_splines,
        deriv_splines,
        Float64.(mean_x),
        Float64.(prec_x),
        Float64.(prior_bounds),
        Float64(offset),
    )
end

"""
    loglike(like::PlanckLowTT, Dls, params) -> Float64

Compute the Planck low-ℓ TT log-likelihood via cubic spline interpolation.

# Arguments
- `like`: `PlanckLowTT` struct
- `Dls`: `NamedTuple` with key `:TT`, starting at ℓ=0
- `params`: `NamedTuple` with key `:A_planck`
"""
function loglike(like::PlanckLowTT, Dls, params)
    A_planck = getproperty(params, :A_planck)
    n = length(like.ell)

    res = map(Tuple(1:n)) do i
        ℓ = like.ell[i]
        d = Dls.TT[ℓ + 1] / (A_planck * A_planck)

        # Check prior bounds and clamp input for spline evaluation
        lower = like.prior_bounds[i, 1]
        upper = like.prior_bounds[i, 2]
        in_bounds = (d >= lower) & (d <= upper)
        d_clamped = clamp(d, lower, upper)

        # Evaluate cubic spline: D_ℓ → x_ℓ
        x_val = evaluate(like.x_splines[i], d_clamped)
        # Evaluate derivative spline for Jacobian
        dxdd = evaluate(like.deriv_splines[i], d_clamped)

        logjac_val = ifelse(in_bounds, log(abs(dxdd)), -1e20)
        x_val_final = ifelse(in_bounds, x_val, zero(x_val))
        return (logjac_val, x_val_final)
    end

    logjac = sum(r -> r[1], res)
    x = [map(r -> r[2], res)...]

    δ = x .- like.mean_x
    chi2_x = dot(δ, like.prec_x * δ)

    return logjac - 0.5 * chi2_x - like.offset
end

"""
    chi2(like::PlanckLowTT, Dls, params) -> Float64

Returns -2 × loglike for consistency with other likelihoods.
Note: for low-ℓ TT this includes the Jacobian term, so it is not
a pure quadratic form.
"""
function chi2(like::PlanckLowTT, Dls, params)
    ll = loglike(like, Dls, params)
    return ifelse(ll == -Inf, Inf, -2.0 * ll)
end

function Base.show(io::IO, like::PlanckLowTT)
    println(io, "PlanckLowTT (cubic spline smoothed)")
    println(io, "  ℓ range: $(like.ell[1]) – $(like.ell[end])")
    print(io,   "  Offset: $(like.offset)")
end

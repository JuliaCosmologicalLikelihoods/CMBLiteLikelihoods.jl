"""
    sroll2.jl — Planck 2018 low-ℓ EE SROLL2 likelihood

Smooth likelihood via cubic spline interpolation of the probability table.
For each ℓ = 2..29, the probability is interpolated from the table using
natural cubic splines (C² continuous, differentiable).

    log L = Σ_ℓ cubic_spline(D_ℓ / A_planck²)

where the spline interpolates the probability table for each ℓ.
"""

"""
    SROLL2

Pre-computed data for the Planck low-ℓ EE SROLL2 likelihood with cubic spline smoothing.

# Fields
- `ell::Vector{Int}`: multipoles 2..29
- `prob_splines::Vector{Tuple{Vector{Float64},Vector{Float64}}}`: (prob_vals, cl_grid) per ℓ
- `cl_grid::Vector{Float64}`: D_ℓ grid (0, stepEE, 2×stepEE, ...)
- `step_ee::Float64`: grid spacing (0.0001 μK²)
- `dell_max::Float64`: maximum D_ℓ value in the table (cl_grid[end])
"""
struct SROLL2
    ell::Vector{Int}
    prob_splines::Vector{CubicSpline}
    cl_grid::Vector{Float64}
    step_ee::Float64
    dell_max::Float64
end

"""
    SROLL2(data_dir::AbstractString)

Load the Planck low-ℓ EE SROLL2 likelihood from a directory of `.npy` files.
Constructs cubic splines for smooth interpolation of the probability table.
"""
function SROLL2(data_dir::AbstractString)
    ell = npzread(joinpath(data_dir, "ell.npy"))
    prob_table = npzread(joinpath(data_dir, "prob_table.npy"))
    cl_grid = npzread(joinpath(data_dir, "cl_grid_Dell.npy"))

    step_ee = length(cl_grid) > 1 ? cl_grid[2] - cl_grid[1] : 0.0001
    dell_max = cl_grid[end]

    # Store spline data as CubicSpline structs
    n_ell = length(ell)
    prob_splines = Vector{CubicSpline}(undef, n_ell)
    for i in 1:n_ell
        prob_splines[i] = CubicSpline(collect(prob_table[:, i]), collect(cl_grid))
    end

    return SROLL2(
        Int.(ell),
        prob_splines,
        Float64.(cl_grid),
        Float64(step_ee),
        Float64(dell_max),
    )
end

"""
    loglike(like::SROLL2, Dls, params) -> Float64

Compute the Planck low-ℓ EE SROLL2 log-likelihood via cubic spline interpolation.

# Arguments
- `like`: `SROLL2` struct
- `Dls`: `NamedTuple` with key `:EE`, starting at ℓ=0
- `params`: `NamedTuple` with key `:A_planck`

Returns `-Inf` if any D_ℓ / A_planck² falls outside the table range.
"""
function loglike(like::SROLL2, Dls, params)
    A_planck = getproperty(params, :A_planck)
    T = promote_type(eltype(Dls.EE), typeof(A_planck))
    logl = zero(T)

    @inbounds for i in 1:length(like.ell)
        ℓ = like.ell[i]
        d = Dls.EE[ℓ + 1] / (A_planck * A_planck)

        # Check bounds and clamp input for spline evaluation
        in_bounds = (d >= 0.0) & (d <= like.dell_max)
        d_clamped = clamp(d, 0.0, like.dell_max)

        # Interpolate probability via cubic spline
        val = evaluate(like.prob_splines[i], d_clamped)
        logl += ifelse(in_bounds, val, -1e20)
    end

    return logl
end

"""
    chi2(like::SROLL2, Dls, params) -> Float64

Returns -2 × loglike for consistency.
"""
function chi2(like::SROLL2, Dls, params)
    ll = loglike(like, Dls, params)
    return ifelse(ll == -Inf, Inf, -2.0 * ll)
end

function Base.show(io::IO, like::SROLL2)
    println(io, "SROLL2 (Planck low-ℓ EE, cubic spline smoothed)")
    println(io, "  ℓ range: $(like.ell[1]) – $(like.ell[end])")
    println(io, "  Splines: $(length(like.prob_splines))")
    print(io,   "  Step: $(like.step_ee) μK²")
end

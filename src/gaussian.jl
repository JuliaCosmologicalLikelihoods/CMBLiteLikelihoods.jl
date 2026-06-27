"""
    gaussian.jl

Common Gaussian likelihood infrastructure shared by all high-ℓ CMB lite
likelihoods.

The Gaussian log-likelihood is:

    log L = -0.5 × δᵀ C⁻¹ δ

where δ = data - model, computed via Cholesky solve for numerical stability.
No normalization constant is included, matching the Python convention.
"""

"""
    gaussian_chi2(cov_chol, δ) -> Real

Compute χ² = δᵀ C⁻¹ δ using the pre-computed Cholesky factorisation.

# Arguments
- `cov_chol`: lower Cholesky factor `L` of the covariance matrix C = L Lᵀ
- `δ`: residual vector (data - model)

# Returns
- χ² value (scalar)
"""
function gaussian_chi2(cov_chol::Cholesky, δ::AbstractVector)
    y = cov_chol.L \ δ
    return dot(y, y)
end

"""
    gaussian_loglike(cov_chol, data, model) -> Real

Compute the Gaussian log-likelihood = -0.5 × χ².

No normalization constant (no -N/2 log(2π) or -0.5 logdet C).
This matches the Python (candl/ACT/CamSpec) convention where
loglike = -0.5 * chi2.
"""
function gaussian_loglike(cov_chol::Cholesky, data::AbstractVector, model::AbstractVector)
    δ = data .- model
    return -0.5 * gaussian_chi2(cov_chol, δ)
end

"""
    CubicSpline

Pre-computed cubic spline representation to avoid solving tridiagonal linear systems
at every likelihood evaluation.
"""
struct CubicSpline
    u::Vector{Float64}
    t::Vector{Float64}
    h::Vector{Float64}
    z::Vector{Float64}
end

function CubicSpline(u::AbstractVector, t::AbstractVector)
    h, z = AbstractCosmologicalEmulators._cubic_spline_coefficients(u, t)
    return CubicSpline(Vector{Float64}(u), Vector{Float64}(t), Vector{Float64}(h), Vector{Float64}(z))
end

"""
    evaluate(spl::CubicSpline, tq::Real)

Evaluate the precomputed cubic spline at a scalar query point.
"""
function evaluate(spl::CubicSpline, tq)
    return AbstractCosmologicalEmulators._cubic_spline_eval(spl.u, spl.t, spl.h, spl.z, tq)
end


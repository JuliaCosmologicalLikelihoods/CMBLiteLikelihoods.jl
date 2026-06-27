"""
    aberration.jl

Aberration correction for SPT-3G, following Jeong et al. 2013 (arXiv:1309.2285),
Eq. 23, as implemented in candl's `AberrationCorrection`.

The additive correction in D_ℓ space is:

    ΔD_ℓ = -AC × ℓ × ℓ(ℓ+1)/(2π) × ∂C_ℓ/∂ℓ

where AC = β⟨cos θ⟩ is the aberration coefficient (e.g. -0.0004826 for SPT-3G).

The derivative ∂C_ℓ/∂ℓ is computed via `jnp.gradient`-equivalent finite
differences on a unit-spaced grid, applied separately per spectrum (TT/TE/EE)
to avoid cross-spectrum derivative spikes.
"""

"""
    _gradient_unit_spacing(y::AbstractVector) -> Vector

Numerical gradient on a unit-spaced grid, matching `np.gradient` behaviour:
- Interior: central differences `(y[i+1] - y[i-1]) / 2`
- Endpoints: one-sided differences
"""
function _gradient_unit_spacing(y::AbstractVector)
    n = length(y)
    dy1 = y[2] - y[1]
    dyn = y[n] - y[n-1]
    if n > 2
        dy_mid = 0.5 .* (y[3:n] .- y[1:n-2])
        return vcat(dy1, dy_mid, dyn)
    else
        return vcat(dy1, dyn)
    end
end

"""
    apply_aberration(Dl::AbstractVector, ell::AbstractVector; AC=-0.0004826)

Apply the aberration correction to a single D_ℓ spectrum.

Returns the corrected D_ℓ = D_ℓ + ΔD_ℓ.

# Arguments
- `Dl`: theory D_ℓ values on the `ell` grid
- `ell`: theory multipoles (must be unit-spaced integers)
- `AC`: aberration coefficient AC = β⟨cos θ⟩

# Returns
- Corrected D_ℓ vector (same length as input)
"""
function apply_aberration(Dl::AbstractVector, ell::AbstractVector; AC=-0.0004826)
    T = promote_type(eltype(Dl), eltype(ell), typeof(AC))
    # Convert D_ℓ → C_ℓ
    Cl = Dl .* (2π) ./ (ell .* (ell .+ 1))
    # Derivative in C_ℓ space
    dCl_dell = _gradient_unit_spacing(Cl)
    # Convert correction back to D_ℓ space
    corr = (-AC) .* ell .* (ell .* (ell .+ 1)) ./ (2π) .* dCl_dell
    return Dl .+ corr
end

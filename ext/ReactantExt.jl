module ReactantExt

using CMBLiteLikelihoods
using Reactant
using LinearAlgebra

using AbstractCosmologicalEmulators

# Overload gaussian_chi2 for Reactant tracked/concrete arrays to bypass Cholesky triangular solve
function CMBLiteLikelihoods.gaussian_chi2(cov_chol::Cholesky, δ::Reactant.TracedRArray{T, 1}) where T
    inv_C = inv(Matrix(cov_chol))
    return dot(δ, inv_C * δ)
end

function CMBLiteLikelihoods.gaussian_chi2(cov_chol::Cholesky, δ::Reactant.ConcreteRArray{T, 1}) where T
    inv_C = inv(Matrix(cov_chol))
    return dot(δ, inv_C * δ)
end


function CMBLiteLikelihoods.evaluate(spl::CMBLiteLikelihoods.CubicSpline, tq::Union{Reactant.TracedRNumber, Reactant.ConcretePJRTNumber})
    tq_vec = vcat(tq)
    u_traced = tq * 0 .+ spl.u
    t_traced = tq * 0 .+ spl.t
    h_traced = tq * 0 .+ spl.h
    z_traced = tq * 0 .+ spl.z
    res_vec = AbstractCosmologicalEmulators._cubic_spline_eval(u_traced, t_traced, h_traced, z_traced, tq_vec)
    return sum(res_vec)
end

end # module

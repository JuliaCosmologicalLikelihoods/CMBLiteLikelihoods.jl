using CMBLiteLikelihoods
using Reactant
using Enzyme
using DifferentiationInterface
import ForwardDiff
using LinearAlgebra
using BenchmarkTools
using Printf

println("=" ^ 80)
println("CMBLiteLikelihoods Lensing Performance Benchmark Suite")
println("=" ^ 80)

# Setup inputs
like = ACTPlanckSPTLensing()
ell = 0:4000
cl_pp_base = [l <= 1 ? 0.0 : 1e-2 / ((l + 10.0) * (l + 11.0))^3 for l in ell]

# Plain Julia function to benchmark
function f_lens_vector(p)
    cl_pp = cl_pp_base .* p[1]
    return chi2(like, cl_pp)
end

# Reactant compiled version needs a tuple or array
function f_lens_tuple(p)
    cl_pp = cl_pp_base .* p[1]
    return chi2(like, cl_pp)
end

p_val = [1.0]
p_tuple = (1.0,)

# Warm up
println("Warming up JIT compiler and functions...")
val_julia = f_lens_vector(p_val)
grad_julia = DifferentiationInterface.gradient(f_lens_vector, AutoForwardDiff(), p_val)

p_reactant = Reactant.to_rarray(p_tuple; track_numbers=true)
c_lens = Reactant.compile(f_lens_tuple, (p_reactant,))
val_reactant = c_lens(p_reactant)

g_enzyme_reactant = Reactant.compile(p -> Enzyme.gradient(Reverse, f_lens_tuple, p)[1], (p_reactant,))
grad_reactant = g_enzyme_reactant(p_reactant)

println("Verification:")
println("  - Julia chi2: $val_julia")
println("  - Reactant chi2: $(Float64(val_reactant))")
println("  - Gradients match: $(Float64(grad_reactant[1]) ≈ grad_julia[1])")

# Run benchmarks
println("\nBenchmarking plain Julia evaluation...")
t_eval_julia = @belapsed $f_lens_vector($p_val)

println("Benchmarking Reactant evaluation...")
t_eval_reactant = @belapsed $c_lens($p_reactant)

println("Benchmarking ForwardDiff gradient...")
t_grad_julia = @belapsed DifferentiationInterface.gradient($f_lens_vector, AutoForwardDiff(), $p_val)

println("Benchmarking Reactant + Enzyme gradient...")
t_grad_reactant = @belapsed $g_enzyme_reactant($p_reactant)

@printf("\nLensing Likelihood Benchmark Results:\n")
@printf("  Plain Julia Evaluation:          %10.3f ms\n", t_eval_julia * 1000)
@printf("  Reactant-compiled Evaluation:     %10.3f ms\n", t_eval_reactant * 1000)
@printf("  Evaluation Speedup:               %10.2fx\n", t_eval_julia / t_eval_reactant)
println("-" ^ 80)
@printf("  Plain Julia ForwardDiff Gradient: %10.3f ms\n", t_grad_julia * 1000)
@printf("  Reactant+Enzyme Gradient:         %10.3f ms\n", t_grad_reactant * 1000)
@printf("  Gradient Speedup:                 %10.2fx\n", t_grad_julia / t_grad_reactant)
println("=" ^ 80)

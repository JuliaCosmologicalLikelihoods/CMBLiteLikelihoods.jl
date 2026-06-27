using CMBLiteLikelihoods
using Reactant
using Enzyme
using DifferentiationInterface
import ForwardDiff
using LinearAlgebra
using BenchmarkTools
using Printf

println("=" ^ 80)
println("CMBLiteLikelihoods Performance Benchmark Suite")
println("=" ^ 80)

# 1. Prepare dummy spectra of length 9001
TT = [3000.0 * exp(-(l - 200)^2 / 20000) for l in 0:9000]
EE = [0.05 * exp(-(l - 200)^2 / 20000) for l in 0:9000]
TE = [100.0 * exp(-(l - 300)^2 / 25000) for l in 0:9000]
Dls = (TT=TT, TE=TE, EE=EE)

# Nuisance parameters
p_tuple = (1.001, 0.999, 1.002, 1.003, 1.002, 1.001, 1.002)
p_vec = collect(p_tuple)

function f_joint_vector(p)
    params = (
        A_planck = p[1],
        calTE = p[2],
        calEE = p[3],
        A_act = p[4],
        P_act = p[5],
        Tcal = p[6],
        Ecal = p[7]
    )
    return joint_chi2(Dls, params)
end

function f_joint_tuple(p)
    params = (
        A_planck = p[1],
        calTE = p[2],
        calEE = p[3],
        A_act = p[4],
        P_act = p[5],
        Tcal = p[6],
        Ecal = p[7]
    )
    return joint_chi2(Dls, params)
end

# Ensure precompilation / JIT warm-up
println("Performing warm-ups / triggering compilation...")
val_julia = f_joint_vector(p_vec)
grad_julia = DifferentiationInterface.gradient(f_joint_vector, AutoForwardDiff(), p_vec)

p_reactant = Reactant.to_rarray(p_tuple; track_numbers=true)
c_joint = Reactant.compile(f_joint_tuple, (p_reactant,))
val_reactant = c_joint(p_reactant)

g_enzyme_reactant = Reactant.compile(p -> Enzyme.gradient(Reverse, f_joint_tuple, p)[1], (p_reactant,))
grad_reactant = g_enzyme_reactant(p_reactant)

# Check correctness
println("Verification:")
println("  - Julia logl value: $val_julia")
println("  - Reactant logl value: $(Float64(val_reactant))")
println("  - Values match: $(val_julia ≈ Float64(val_reactant))")
grad_reactant_vec = [Float64(g) for g in grad_reactant]
println("  - Gradients match: $(grad_reactant_vec ≈ grad_julia)")
println("-" ^ 80)

# Benchmark Execution
println("Benchmarking plain Julia joint_chi2 evaluation...")
t_eval_julia = @belapsed $f_joint_vector($p_vec)

println("Benchmarking Reactant-compiled joint_chi2 evaluation...")
t_eval_reactant = @belapsed $c_joint($p_reactant)

println("Benchmarking plain Julia ForwardDiff gradient computation...")
t_grad_julia = @belapsed DifferentiationInterface.gradient($f_joint_vector, AutoForwardDiff(), $p_vec)

println("Benchmarking Reactant-compiled Enzyme gradient computation...")
t_grad_reactant = @belapsed $g_enzyme_reactant($p_reactant)

println("=" ^ 80)
println("Benchmark Results Summary:")
println("=" ^ 80)
@printf("  Plain Julia Evaluation:          %10.3f ms\n", t_eval_julia * 1000)
@printf("  Reactant-compiled Evaluation:     %10.3f ms\n", t_eval_reactant * 1000)
@printf("  Evaluation Speedup:               %10.2fx\n", t_eval_julia / t_eval_reactant)
println("-" ^ 80)
@printf("  Plain Julia ForwardDiff Gradient: %10.3f ms\n", t_grad_julia * 1000)
@printf("  Reactant+Enzyme Gradient:         %10.3f ms\n", t_grad_reactant * 1000)
@printf("  Gradient Speedup:                 %10.2fx\n", t_grad_julia / t_grad_reactant)
println("=" ^ 80)

using BenchmarkTools
using CMBLiteLikelihoods
using NPZ
using LinearAlgebra

const EXPORTS = joinpath(@__DIR__, "..", "..", "cmbliteplay", "exports")

function delim_readdlm(path::AbstractString)
    lines = readlines(path)
    rows = filter(line -> !startswith(line, "#"), filter(!isempty, strip.(lines)))
    ncol = length(split(rows[1]))
    mat = Matrix{Float64}(undef, length(rows), ncol)
    for (i, line) in enumerate(rows)
        vals = split(line)
        for (j, v) in enumerate(vals)
            mat[i, j] = parse(Float64, v)
        end
    end
    return mat
end

function pad_spec(raw)
    ell_min = Int(raw[1, 1])
    TT = zeros(Float64, ell_min); TE = zeros(Float64, ell_min); EE = zeros(Float64, ell_min)
    append!(TT, raw[:, 2]); append!(TE, raw[:, 3]); append!(EE, raw[:, 4])
    return (TT=TT, TE=TE, EE=EE)
end

# Load test spectra
act_spec = delim_readdlm(joinpath(EXPORTS, "..", "_sources", "candl_data",
    "candl_data", "tests", "ACT_DR6_TTTEEE_test_spec.txt"))
spt_spec = delim_readdlm(joinpath(EXPORTS, "..", "_sources", "spt_candl_data",
    "spt_candl_data", "tests", "SPT3G_D1_TnE_test_spec.txt"))

Dls_act = pad_spec(act_spec)
Dls_spt = pad_spec(spt_spec)

# === CamSpec ===
println("=" ^ 60)
println("CamSpec NPIPE lite")
println("=" ^ 60)
like_cs = CamSpecLite(joinpath(EXPORTS, "camspec_npipe_lite"))
params_cs = (A_planck=1.0, calTE=1.0, calEE=1.0)
println("loglike = $(loglike(like_cs, Dls_act, params_cs))")
display(@benchmark loglike($like_cs, $Dls_act, $params_cs))
println("\n")

# === ACT DR6 ===
println("=" ^ 60)
println("ACT DR6 CMB-only")
println("=" ^ 60)
like_act = ACTDR6CMBOnly(joinpath(EXPORTS, "act_dr6_cmbonly"))
params_act = (A_act=1.0, P_act=1.0)
println("loglike = $(loglike(like_act, Dls_act, params_act))")
display(@benchmark loglike($like_act, $Dls_act, $params_act))
println("\n")

# === SPT-3G ===
println("=" ^ 60)
println("SPT-3G D1 TnE lite")
println("=" ^ 60)
like_spt = SPT3GD1Lite(joinpath(EXPORTS, "spt3g_d1_tne_lite"))
params_spt = (Tcal=1.001, Ecal=0.999)
println("loglike = $(loglike(like_spt, Dls_spt, params_spt))")
display(@benchmark loglike($like_spt, $Dls_spt, $params_spt))
println("\n")

# === Planck low-TT ===
println("=" ^ 60)
println("Planck low-l TT")
println("=" ^ 60)
like_lowtt = PlanckLowTT(joinpath(EXPORTS, "planck_2018_lowl_TT"))
mu_sigma = npzread(joinpath(EXPORTS, "planck_2018_lowl_TT", "mu_sigma_Dell.npy"))
Dls_lowtt = (TT=zeros(Float64, 30),)
for i in 1:length(like_lowtt.ell)
    Dls_lowtt.TT[like_lowtt.ell[i] + 1] = mu_sigma[i]
end
params_lowtt = (A_planck=1.0,)
println("loglike = $(loglike(like_lowtt, Dls_lowtt, params_lowtt))")
display(@benchmark loglike($like_lowtt, $Dls_lowtt, $params_lowtt))
println("\n")

# === SROLL2 ===
println("=" ^ 60)
println("SROLL2 (Planck low-l EE)")
println("=" ^ 60)
like_sroll2 = SROLL2(joinpath(EXPORTS, "planck_2018_lowl_EE_sroll2"))
Dls_sroll2 = (EE=zeros(Float64, 30),)
for i in 1:length(like_sroll2.ell)
    ℓ = like_sroll2.ell[i]
    Dls_sroll2.EE[ℓ + 1] = Dls_spt.EE[ℓ + 1]
end
params_sroll2 = (A_planck=1.0,)
println("loglike = $(loglike(like_sroll2, Dls_sroll2, params_sroll2))")
display(@benchmark loglike($like_sroll2, $Dls_sroll2, $params_sroll2))
println()

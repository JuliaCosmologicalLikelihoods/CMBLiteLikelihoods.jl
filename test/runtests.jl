using Test
using CMBLiteLikelihoods
using LinearAlgebra
using NPZ
using DifferentiationInterface
import ForwardDiff
import FiniteDiff
using Reactant
using Enzyme
using Zygote
using Mooncake
using Artifacts


const HAVE_CORRECTIONS = begin
    artifacts_toml = joinpath(pkgdir(CMBLiteLikelihoods), "Artifacts.toml")
    meta = Artifacts.artifact_meta("cmblite_data", artifacts_toml)
    art_path = Artifacts.artifact_path(Base.SHA1(meta["git-tree-sha1"]))
    local_dev_path = joinpath(pkgdir(CMBLiteLikelihoods), "..", "cmbliteplay", "_sources", "spt_act_likelihood", "act_dr6_spt_lenslike", "data", "v1.2", "like_corrs")
    isdir(joinpath(art_path, "cmblite_data", "act_planck_spt3g_lensing", "like_corrs")) || isdir(local_dev_path)
end

const EXPORTS = joinpath(@__DIR__, "..", "..", "cmbliteplay", "exports")

# Helper: read whitespace-delimited file
function delim_readdlm(path::AbstractString)
    lines = readlines(path)
    rows = filter(!startswith.("#"), filter(!isempty, strip.(lines)))
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

# -----------------------------------------------------------------------
# Section 1: Data loading and shape tests
# -----------------------------------------------------------------------

@testset "CamSpec data loading" begin
    dir = joinpath(EXPORTS, "camspec_npipe_lite")
    if !isdir(dir)
        @warn "CamSpec data not found at $dir"
        @test_skip 1
    else
        like = CamSpecLite(dir)
        @test length(like.data_vector) == 6413
        @test size(like.cov_chol.L, 1) == 6413
        @test like.n_bins == [2471, 1971, 1971]
        @test minimum(like.ell) == 30
        @test maximum(like.ell) == 2500
    end
end

@testset "ACT data loading" begin
    dir = joinpath(EXPORTS, "act_dr6_cmbonly")
    if !isdir(dir)
        @warn "ACT data not found at $dir"
        @test_skip 1
    else
        like = ACTDR6CMBOnly(dir)
        @test length(like.data_vector) == 135
        @test like.n_bins == [45, 45, 45]
        @test size(like.windows[1], 1) == 45
        @test minimum(like.window_ell[1]) == 2
        @test maximum(like.window_ell[1]) == 8501
    end
end

@testset "SPT data loading" begin
    dir = joinpath(EXPORTS, "spt3g_d1_tne_lite")
    if !isdir(dir)
        @warn "SPT data not found at $dir"
        @test_skip 1
    else
        like = SPT3GD1Lite(dir)
        @test length(like.data_vector) == 196
        @test like.n_bins == [52, 72, 72]
        @test size(like.windows[1], 1) == 52
        @test like.theory_ell[1] == 2
        @test like.theory_ell[end] == 4095
        @test like.aberration_coeff ≈ -0.0004826
    end
end

@testset "Planck low-TT data loading" begin
    dir = joinpath(EXPORTS, "planck_2018_lowl_TT")
    if !isdir(dir)
        @warn "LowTT data not found at $dir"
        @test_skip 1
    else
        like = PlanckLowTT(dir)
        @test length(like.ell) == 28
        @test like.ell[1] == 2
        @test like.ell[end] == 29
        @test length(like.x_splines) == 28
        @test length(like.deriv_splines) == 28
        @test size(like.prec_x) == (28, 28)
    end
end

@testset "SROLL2 data loading" begin
    dir = joinpath(EXPORTS, "planck_2018_lowl_EE_sroll2")
    if !isdir(dir)
        @warn "SROLL2 data not found at $dir"
        @test_skip 1
    else
        like = SROLL2(dir)
        @test length(like.ell) == 28
        @test length(like.prob_splines) == 28
        @test like.step_ee ≈ 0.0001
    end
end

# -----------------------------------------------------------------------
# Section 2: Utility function tests
# -----------------------------------------------------------------------

@testset "Aberration correction" begin
    # Flat D_ℓ → C_ℓ = D_ℓ × 2π/[ℓ(ℓ+1)] is NOT flat, so derivative ≠ 0.
    # But correction should be small and finite.
    ell = collect(2.0:100.0)
    Dl_flat = ones(Float64, length(ell))
    Dl_corr = apply_aberration(Dl_flat, ell; AC=-0.0004826)
    @test all(isfinite, Dl_corr)
    # Correction should be small relative to the spectrum value
    @test all(abs.(Dl_corr .- Dl_flat) .< 0.01)

    # Linear spectrum: derivative is constant
    Dl_lin = 2.0 .* ell
    Dl_corr2 = apply_aberration(Dl_lin, ell; AC=-0.0004826)
    @test all(isfinite, Dl_corr2)
    # Should be slightly different from input
    @test !all(Dl_corr2 .≈ Dl_lin)
end

@testset "Gaussian chi2" begin
    # Identity covariance: chi2 = sum of squared residuals
    C = Matrix{Float64}(I, 3, 3)
    chol = cholesky(Symmetric(C))
    δ = [1.0, 2.0, 3.0]
    @test gaussian_chi2(chol, δ) ≈ 14.0  # 1+4+9
end

# -----------------------------------------------------------------------
# Section 3: SPT reference test (candl test spectrum)
# -----------------------------------------------------------------------

@testset "SPT-3G reference (candl test spectrum)" begin
    dir = joinpath(EXPORTS, "spt3g_d1_tne_lite")
    test_spec = joinpath(EXPORTS, "..", "_sources", "spt_candl_data",
                         "spt_candl_data", "tests", "SPT3G_D1_TnE_test_spec.txt")

    if !isdir(dir) || !isfile(test_spec)
        @warn "SPT data or test spectrum not found, skipping reference test"
        @test_skip 1
    else
        # Load test spectrum (starts at ℓ=2, pad to start at ℓ=0)
        raw = delim_readdlm(test_spec)
        # Columns: ell TT TE EE BB pp kk
        ell_min_data = Int(raw[1, 1])  # should be 2
        TT = zeros(Float64, ell_min_data)
        TE = zeros(Float64, ell_min_data)
        EE = zeros(Float64, ell_min_data)
        append!(TT, raw[:, 2])
        append!(TE, raw[:, 3])
        append!(EE, raw[:, 4])
        Dls = (TT=TT, TE=TE, EE=EE)

        params = (Tcal=1.001, Ecal=0.999)

        like = SPT3GD1Lite(dir)
        ll = loglike(like, Dls, params)

        # Expected value with priors cleared (matching clear_internal_priors=True)
        # chi2 = 166.09132499594295
        # loglike = -83.04566249797148
        @test ll ≈ -83.04566249797148 rtol=1e-4
    end
end
# -----------------------------------------------------------------------
# Section 3.5: High-ℓ reference tests (candl test spectrum)
# -----------------------------------------------------------------------

@testset "ACT DR6 CMB-only reference (candl test spectrum)" begin
    dir = joinpath(EXPORTS, "act_dr6_cmbonly")
    test_spec = joinpath(EXPORTS, "..", "_sources", "candl_data",
                         "candl_data", "tests", "ACT_DR6_TTTEEE_test_spec.txt")

    if !isdir(dir) || !isfile(test_spec)
        @warn "ACT data or test spectrum not found, skipping reference test"
        @test_skip 1
    else
        # Load test spectrum (starts at ℓ=2, pad to start at ℓ=0)
        raw = delim_readdlm(test_spec)
        ell_min_data = Int(raw[1, 1])  # should be 2
        TT = zeros(Float64, ell_min_data)
        TE = zeros(Float64, ell_min_data)
        EE = zeros(Float64, ell_min_data)
        append!(TT, raw[:, 2])
        append!(TE, raw[:, 3])
        append!(EE, raw[:, 4])
        Dls = (TT=TT, TE=TE, EE=EE)

        params = (A_act=1.0, P_act=1.0)

        like = ACTDR6CMBOnly(dir)
        ll = loglike(like, Dls, params)

        # Expected value from candl test spectrum with A_act=1, P_act=1
        # chi2 = 177.563540193143751
        # loglike = -88.781770096571876
        @test ll ≈ -88.781770096571876 rtol=1e-4
    end
end

@testset "CamSpec NPIPE lite reference (candl test spectrum)" begin
    dir = joinpath(EXPORTS, "camspec_npipe_lite")
    test_spec = joinpath(EXPORTS, "..", "_sources", "candl_data",
                         "candl_data", "tests", "ACT_DR6_TTTEEE_test_spec.txt")

    if !isdir(dir) || !isfile(test_spec)
        @warn "CamSpec data or test spectrum not found, skipping reference test"
        @test_skip 1
    else
        # Load test spectrum (starts at ℓ=2, pad to start at ℓ=0)
        raw = delim_readdlm(test_spec)
        ell_min_data = Int(raw[1, 1])  # should be 2
        TT = zeros(Float64, ell_min_data)
        TE = zeros(Float64, ell_min_data)
        EE = zeros(Float64, ell_min_data)
        append!(TT, raw[:, 2])
        append!(TE, raw[:, 3])
        append!(EE, raw[:, 4])
        Dls = (TT=TT, TE=TE, EE=EE)

        params = (A_planck=1.0, calTE=1.0, calEE=1.0)

        like = CamSpecLite(dir)
        ll = loglike(like, Dls, params)

        # Expected value from candl test spectrum with A_planck=1, calTE=1, calEE=1
        # chi2 = 6788.989977757713859
        # loglike = -3394.494988878856930
        @test ll ≈ -3394.4949888788569 rtol=1e-4
    end
end

@testset "Planck low-TT reference value" begin
    dir = joinpath(EXPORTS, "planck_2018_lowl_TT")
    if !isdir(dir)
        @warn "LowTT data not found, skipping reference value test"
        @test_skip 1
    else
        like = PlanckLowTT(dir)
        mu_sigma = npzread(joinpath(dir, "mu_sigma_Dell.npy"))
        Dls = (TT=zeros(Float64, 30),)
        for i in 1:length(like.ell)
            Dls.TT[like.ell[i] + 1] = mu_sigma[i]
        end
        params = (A_planck=1.0,)
        ll = loglike(like, Dls, params)
        @test ll ≈ 4.351987902850851e-5 atol=1e-7
    end
end

@testset "SROLL2 reference value" begin
    dir = joinpath(EXPORTS, "planck_2018_lowl_EE_sroll2")
    test_spec = joinpath(EXPORTS, "..", "_sources", "spt_candl_data",
                         "spt_candl_data", "tests", "SPT3G_D1_TnE_test_spec.txt")
    if !isdir(dir) || !isfile(test_spec)
        @warn "SROLL2 data or test spectrum not found, skipping reference value test"
        @test_skip 1
    else
        like = SROLL2(dir)
        raw = delim_readdlm(test_spec)
        
        Dls = (EE=zeros(Float64, 30),)
        for i in 1:length(like.ell)
            ℓ = like.ell[i]
            row = findfirst(==(ℓ), raw[:, 1])
            Dls.EE[ℓ + 1] = raw[row, 4]
        end
        params = (A_planck=1.0,)
        ll = loglike(like, Dls, params)
        @test ll ≈ -195.04623884156962 rtol=1e-6
    end
end

@testset "CAMB spectrum regression tests" begin
    spec_file = joinpath(@__DIR__, "camb_test_spec.txt")
    ref_file = joinpath(@__DIR__, "camb_test_reference.jl")

    if !isfile(spec_file) || !isfile(ref_file)
        @warn "CAMB test data not found, skipping CAMB regression tests"
        @test_skip 1
    else
        # Load reference values
        include(ref_file)

        # Load spectrum
        raw_spec = delim_readdlm(spec_file)
        Dls = (
            TT = raw_spec[:, 2],
            TE = raw_spec[:, 3],
            EE = raw_spec[:, 4]
        )

        # A. CamSpec
        like_camspec = CamSpecLite()
        # Fiducial
        @test loglike(like_camspec, Dls, (A_planck=1.0, calTE=1.0, calEE=1.0)) ≈ REF_CAMSPEC_FID rtol=1e-5
        # Non-fiducial
        @test loglike(like_camspec, Dls, (A_planck=0.990, calTE=0.995, calEE=1.005)) ≈ REF_CAMSPEC_NONFID rtol=1e-5

        # B. ACT
        like_act = ACTDR6CMBOnly()
        # Fiducial
        @test loglike(like_act, Dls, (A_act=1.0, P_act=1.0)) ≈ REF_ACT_FID rtol=1e-5
        # Non-fiducial
        @test loglike(like_act, Dls, (A_act=0.985, P_act=1.015)) ≈ REF_ACT_NONFID rtol=1e-5

        # C. SPT
        like_spt = SPT3GD1Lite()
        # Fiducial
        @test loglike(like_spt, Dls, (Tcal=1.0, Ecal=1.0)) ≈ -REF_SPT_FID rtol=1e-5
        # Non-fiducial
        @test loglike(like_spt, Dls, (Tcal=0.992, Ecal=1.008)) ≈ -REF_SPT_NONFID rtol=1e-5

        # D. Planck Low-TT
        like_lowtt = PlanckLowTT()
        # Fiducial
        @test loglike(like_lowtt, Dls, (A_planck=1.0,)) ≈ REF_LOWTT_FID rtol=1e-5
        # Non-fiducial
        @test loglike(like_lowtt, Dls, (A_planck=0.990,)) ≈ REF_LOWTT_NONFID rtol=1e-5

        # E. SROLL2
        like_sroll2 = SROLL2()
        # Fiducial
        @test loglike(like_sroll2, Dls, (A_planck=1.0,)) ≈ REF_SROLL2_FID rtol=1e-5
        # Non-fiducial
        @test loglike(like_sroll2, Dls, (A_planck=0.990,)) ≈ REF_SROLL2_NONFID rtol=1e-5
    end
end

# -----------------------------------------------------------------------
# Section 4: Likelihood sanity checks
# -----------------------------------------------------------------------

@testset "Likelihood sanity checks" begin
    for (name, T, dir, params_func) in (
        ("CamSpec", CamSpecLite, joinpath(EXPORTS, "camspec_npipe_lite"),
            () -> (A_planck=1.0, calTE=1.0, calEE=1.0)),
        ("ACT", ACTDR6CMBOnly, joinpath(EXPORTS, "act_dr6_cmbonly"),
            () -> (A_act=1.0, P_act=1.0)),
    )
        @testset "$name" begin
            if !isdir(dir)
                @warn "$name data not found, skipping"
                @test_skip 1
            else
                like = T(dir)
                n = length(like.data_vector)

                # Build fake theory Dls (all zeros → model = 0 → δ = data)
                Dls_fake = (
                    TT = zeros(9001),
                    TE = zeros(9001),
                    EE = zeros(9001),
                )
                ll = loglike(like, Dls_fake, params_func())
                @test isfinite(ll)
                @test ll < 0.0

                # chi2 should be positive
                χ² = chi2(like, Dls_fake, params_func())
                @test χ² > 0.0
                @test χ² ≈ -2.0 * ll
            end
        end
    end
end
# Global definitions for AD and Reactant test suite
const GLOBAL_JOINT_TT = [3000.0 * exp(-(l - 200)^2 / 20000) for l in 0:9000]
const GLOBAL_JOINT_EE = [0.05 * exp(-(l - 200)^2 / 20000) for l in 0:9000]
const GLOBAL_JOINT_TE = [100.0 * exp(-(l - 300)^2 / 25000) for l in 0:9000]
const GLOBAL_JOINT_DLS = (TT=GLOBAL_JOINT_TT, TE=GLOBAL_JOINT_TE, EE=GLOBAL_JOINT_EE)

function f_joint_global(p)
    params = (
        A_planck = p[1],
        calTE = p[2],
        calEE = p[3],
        A_act = p[4],
        P_act = p[5],
        Tcal = p[6],
        Ecal = p[7]
    )
    return joint_chi2(GLOBAL_JOINT_DLS, params)
end

# Individual helper functions for each likelihood component
function f_indiv_plancklowtt(p)
    return chi2(PlanckLowTT(), GLOBAL_JOINT_DLS, (A_planck=p[1],))
end
function f_indiv_sroll2(p)
    return chi2(SROLL2(), GLOBAL_JOINT_DLS, (A_planck=p[1],))
end
function f_indiv_camspec(p)
    return chi2(CamSpecLite(), GLOBAL_JOINT_DLS, (A_planck=p[1], calTE=p[2], calEE=p[3]))
end
function f_indiv_act(p)
    return chi2(ACTDR6CMBOnly(), GLOBAL_JOINT_DLS, (A_act=p[1], P_act=p[2]))
end
function f_indiv_spt(p)
    return chi2(SPT3GD1Lite(), GLOBAL_JOINT_DLS, (Tcal=p[1], Ecal=p[2]))
end



@testset "AD, Joint Likelihood, and Reactant" begin
    p_vec = [1.001, 0.999, 1.002, 1.003, 1.002, 1.001, 1.002]

    val = f_joint_global(p_vec)
    @test isfinite(val)
    @test val > 0.0

    # 1. Differentiation tests using DifferentiationInterface
    grad_fd = DifferentiationInterface.gradient(f_joint_global, AutoForwardDiff(), p_vec)
    @test all(isfinite, grad_fd)

    grad_findiff = DifferentiationInterface.gradient(f_joint_global, AutoFiniteDiff(), p_vec)
    @test all(isfinite, grad_findiff)

    @test grad_fd ≈ grad_findiff rtol=1e-3 atol=1e-3

    # Zygote differentiation test
    @testset "Zygote AD" begin
        grad_zygote = DifferentiationInterface.gradient(f_joint_global, AutoZygote(), p_vec)
        @test all(isfinite, grad_zygote)
        @test grad_zygote ≈ grad_fd rtol=1e-3 atol=1e-3
    end

    # Mooncake differentiation test
    @testset "Mooncake AD" begin
        grad_mooncake = DifferentiationInterface.gradient(f_joint_global, AutoMooncake(), p_vec)
        @test all(isfinite, grad_mooncake)
        @test grad_mooncake ≈ grad_fd rtol=1e-3 atol=1e-3
    end

    # Test differentiation of each individual likelihood
    for (name, f_indiv, p_sub) in [
        ("PlanckLowTT", f_indiv_plancklowtt, [1.001]),
        ("SROLL2", f_indiv_sroll2, [1.001]),
        ("CamSpecLite", f_indiv_camspec, [1.001, 0.999, 1.002]),
        ("ACTDR6CMBOnly", f_indiv_act, [1.003, 1.002]),
        ("SPT3GD1Lite", f_indiv_spt, [1.001, 1.002])
    ]
        @testset "AD - $name" begin
            g_ad = DifferentiationInterface.gradient(f_indiv, AutoForwardDiff(), p_sub)
            g_num = DifferentiationInterface.gradient(f_indiv, AutoFiniteDiff(), p_sub)

            @test all(isfinite, g_ad)
            @test g_ad ≈ g_num rtol=1e-3 atol=1e-3
        end
    end

    # 2. Reactant compilation and output consistency
    @testset "Reactant Compilation" begin
        compiled_joint = Reactant.compile(f_joint_global, (p_vec,))
        reactant_val = compiled_joint(p_vec)
        @test reactant_val ≈ val
    end

    @testset "Reactant + Enzyme Differentiation" begin
        p_tuple = (p_vec...,)
        p_reactant = Reactant.to_rarray(p_tuple; track_numbers=true)
        g_enzyme_reactant = Reactant.compile(p -> Enzyme.gradient(Reverse, f_joint_global, p)[1], (p_reactant,))
        reactant_grad = g_enzyme_reactant(p_reactant)
        reactant_grad_vec = Float64[Float64(g) for g in reactant_grad]
        @test reactant_grad_vec ≈ grad_fd rtol=1e-4 atol=1e-4
    end


end

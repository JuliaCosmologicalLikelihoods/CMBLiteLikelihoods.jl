"""
    lensing.jl — ACT DR6 + Planck NPIPE + SPT-3G baseline lensing likelihood

Joint CMB lensing likelihood combining ACT, Planck, and SPT-3G baseline data.
Supports both lensing-only mode and dynamic runtime CMB corrections.
Input is the CMB lensing potential spectrum C_ℓ^ϕϕ (cl_pp) starting at ℓ = 0.
"""

# Helper function to read space-separated matrices from text files
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

# Cache parsed text matrices as .npy files to speed up subsequent loads
function load_matrix_cached(path::AbstractString)
    npy_path = replace(path, ".txt" => ".npy")
    if isfile(npy_path)
        return npzread(npy_path)
    else
        mat = delim_readdlm(path)
        try
            npzwrite(npy_path, mat)
        catch e
            @warn "Could not write cached .npy file: $e"
        end
        return mat
    end
end

# Standardize multipoles up to nlen = trim_lmax + lbuffer (typically 3000)
function julia_standardize(ls, cls, trim_lmax; lbuffer=2, extra_dims="y")
    cstart = Int(ls[1])
    nlen = trim_lmax + lbuffer
    cend = nlen - cstart
    
    if extra_dims == "xyy"
        out = zeros(size(cls, 1), nlen, nlen)
        out[:, (cstart+1):nlen, (cstart+1):nlen] .= cls[:, 1:cend, 1:cend]
        return out
    elseif extra_dims == "yy"
        out = zeros(nlen, nlen)
        out[(cstart+1):nlen, (cstart+1):nlen] .= cls[1:cend, 1:cend]
        return out
    elseif extra_dims == "xy"
        out = zeros(size(cls, 1), nlen)
        out[:, (cstart+1):nlen] .= cls[:, 1:cend]
        return out
    elseif extra_dims == "y"
        out = zeros(nlen)
        out[(cstart+1):nlen] .= cls[1:cend]
        return out
    else
        error("Unknown extra_dims $extra_dims")
    end
end

"""
    ACTPlanckSPTLensing

Pre-computed data structure for the ACT DR6 + Planck + SPT-3G joint lensing likelihood.

# Fields
- `data_vector::Vector{Float64}`: combined observed lensing band powers (length 35)
- `cov_chol::Cholesky{Float64, Matrix{Float64}}`: Cholesky factorisation of the covariance matrix
- `binmat_act::Matrix{Float64}`: binning matrix mapping cl_kk[0:2999] to ACT baseline bins (shape 10 x 3000)
- `binmat_planck::Matrix{Float64}`: binning matrix mapping cl_kk[0:2999] to Planck bins (shape 9 x 3000)
- `binmat_spt::Matrix{Float64}`: binning matrix mapping cl_kk[0:3101] to SPT bins (shape 16 x 3102)
- `has_corrections::Bool`: whether runtime CMB corrections are loaded
"""
struct ACTPlanckSPTLensing
    data_vector::Vector{Float64}
    cov_chol::Cholesky{Float64, Matrix{Float64}}
    binmat_act::Matrix{Float64}
    binmat_planck::Matrix{Float64}
    binmat_spt::Matrix{Float64}
    
    # Corrections flags
    has_corrections::Bool

    # Fiducial spectra (zeros if has_corrections is false)
    fiducial_cl_tt::Vector{Float64}
    fiducial_cl_ee::Vector{Float64}
    fiducial_cl_te::Vector{Float64}
    fiducial_cl_bb::Vector{Float64}
    fiducial_cl_kk::Vector{Float64}
    
    # Fiducial normalizations (zeros if has_corrections is false)
    fAL::Vector{Float64}
    fAL_planck::Vector{Float64}

    # Mask for l < 2 (zeros if has_corrections is false)
    mask_l2::Vector{Float64}

    # Correction matrices (zeros(0,0) if has_corrections is false)
    dN1_kk::Matrix{Float64}
    dN1_tt::Matrix{Float64}
    dN1_ee::Matrix{Float64}
    dN1_te::Matrix{Float64}
    dN1_bb::Matrix{Float64}
    dAL_dC_tt::Matrix{Float64}
    dAL_dC_ee::Matrix{Float64}
    dAL_dC_bb::Matrix{Float64}
    dAL_dC_te::Matrix{Float64}

    dN1_kk_planck::Matrix{Float64}
    dN1_tt_planck::Matrix{Float64}
    dN1_ee_planck::Matrix{Float64}
    dN1_te_planck::Matrix{Float64}
    dN1_bb_planck::Matrix{Float64}
    dAL_dC_planck_tt::Matrix{Float64}
    dAL_dC_planck_ee::Matrix{Float64}
    dAL_dC_planck_bb::Matrix{Float64}
    dAL_dC_planck_te::Matrix{Float64}
end

"""
    ACTPlanckSPTLensing(data_dir::AbstractString; load_corrections::Bool=false)

Load the joint lensing likelihood from the specified data directory.
If `load_corrections` is `true`, active CMB correction matrices are also loaded.
"""
function ACTPlanckSPTLensing(data_dir::AbstractString; load_corrections::Bool=false)
    data_vector = npzread(joinpath(data_dir, "data_binned_clkk.npy"))
    cinv = npzread(joinpath(data_dir, "cinv.npy"))
    binmat_act = npzread(joinpath(data_dir, "binmat_act.npy"))
    binmat_planck = npzread(joinpath(data_dir, "binmat_planck.npy"))
    binmat_spt = npzread(joinpath(data_dir, "binmat_spt.npy"))

    cov_hartlap = inv(cinv)
    cov_chol = cholesky(Symmetric(cov_hartlap, :L))

    if !load_corrections
        return ACTPlanckSPTLensing(
            Float64.(data_vector),
            cov_chol,
            Float64.(binmat_act),
            Float64.(binmat_planck),
            Float64.(binmat_spt),
            false,
            zeros(0), zeros(0), zeros(0), zeros(0), zeros(0),
            zeros(0), zeros(0), zeros(0),
            zeros(0,0), zeros(0,0), zeros(0,0), zeros(0,0), zeros(0,0),
            zeros(0,0), zeros(0,0), zeros(0,0), zeros(0,0),
            zeros(0,0), zeros(0,0), zeros(0,0), zeros(0,0), zeros(0,0),
            zeros(0,0), zeros(0,0), zeros(0,0), zeros(0,0)
        )
    end

    # Find like_corrs directory
    lensing_data_dir = dirname(dirname(data_dir))
    like_corrs_dir = if isdir(joinpath(lensing_data_dir, "like_corrs"))
        joinpath(lensing_data_dir, "like_corrs")
    else
        "/home/marcobonici/Desktop/work/CosmologicalLikelihoods/cmbliteplay/_sources/spt_act_likelihood/act_dr6_spt_lenslike/data/v1.2/like_corrs"
    end

    # 1. Load fiducial lensed CMB spectra
    fid_cls = delim_readdlm(joinpath(like_corrs_dir, "cosmo2017_10K_acc3_lensedCls.dat"))
    f_ls = fid_cls[:, 1]
    f_tt = fid_cls[:, 2] ./ (f_ls .* (f_ls .+ 1.0)) .* (2.0 * π)
    f_ee = fid_cls[:, 3] ./ (f_ls .* (f_ls .+ 1.0)) .* (2.0 * π)
    f_bb = fid_cls[:, 4] ./ (f_ls .* (f_ls .+ 1.0)) .* (2.0 * π)
    f_te = fid_cls[:, 5] ./ (f_ls .* (f_ls .+ 1.0)) .* (2.0 * π)
    
    trim_lmax = 2998
    fiducial_cl_tt = julia_standardize(f_ls, f_tt, trim_lmax)
    fiducial_cl_ee = julia_standardize(f_ls, f_ee, trim_lmax)
    fiducial_cl_bb = julia_standardize(f_ls, f_bb, trim_lmax)
    fiducial_cl_te = julia_standardize(f_ls, f_te, trim_lmax)

    # 2. Load fiducial lensing potential
    fid_pot = delim_readdlm(joinpath(like_corrs_dir, "cosmo2017_10K_acc3_lenspotentialCls.dat"))
    fd_ls = fid_pot[:, 1]
    f_kk = fid_pot[:, 6] .* (2.0 * π) ./ 4.0
    fiducial_cl_kk = julia_standardize(fd_ls, f_kk, trim_lmax)

    # 3. Load fiducial norm (fAL)
    fAL_data = delim_readdlm(joinpath(like_corrs_dir, "n0mv_fiducial_lmin600_lmax3000_Lmin0_Lmax4000.txt"))
    fAL_ls = fAL_data[1, :]
    fAL = julia_standardize(fAL_ls, fAL_data[2, :], trim_lmax)

    fAL_planck_data = delim_readdlm(joinpath(like_corrs_dir, "PLANCK_n0mv_fiducial_lmin600_lmax3000_Lmin0_Lmax3000.txt"))
    fAL_planck = julia_standardize(fAL_planck_data[1, :], fAL_planck_data[2, :], trim_lmax)

    mask_l2 = Float64[l < 2 ? 0.0 : 1.0 for l in 0:2999]

    # 4. Load normalization derivative matrices
    cmat = npzread(joinpath(like_corrs_dir, "norm_correction_matrix_Lmin0_Lmax4000.npy"))
    ls_cmat = 0:(size(cmat, 2) - 1)
    dAL_dC_all = julia_standardize(ls_cmat, cmat, trim_lmax; extra_dims="xyy")
    dAL_dC_tt = dAL_dC_all[1, :, :]
    dAL_dC_ee = dAL_dC_all[2, :, :]
    dAL_dC_bb = dAL_dC_all[3, :, :]
    dAL_dC_te = dAL_dC_all[4, :, :]

    cmat_p = npzread(joinpath(like_corrs_dir, "P18_norm_correction_matrix_Lmin0_Lmax3000.npy"))
    ls_cmat_p = 0:(size(cmat_p, 2) - 1)
    dAL_dC_planck_all = julia_standardize(ls_cmat_p, cmat_p, trim_lmax; extra_dims="xyy")
    dAL_dC_planck_tt = dAL_dC_planck_all[1, :, :]
    dAL_dC_planck_ee = dAL_dC_planck_all[2, :, :]
    dAL_dC_planck_bb = dAL_dC_planck_all[3, :, :]
    dAL_dC_planck_te = dAL_dC_planck_all[4, :, :]

    # 5. Load N1 derivative matrices
    dN1_kk = load_matrix_cached(joinpath(like_corrs_dir, "N1der_KK_lmin600_lmax3000_full.txt"))
    dN1_kk = julia_standardize(fAL_ls, dN1_kk, trim_lmax; extra_dims="yy")

    dN1_tt = load_matrix_cached(joinpath(like_corrs_dir, "N1der_TT_lmin600_lmax3000_full.txt"))
    dN1_tt = julia_standardize(fAL_ls, dN1_tt, trim_lmax; extra_dims="yy")

    dN1_ee = load_matrix_cached(joinpath(like_corrs_dir, "N1der_EE_lmin600_lmax3000_full.txt"))
    dN1_ee = julia_standardize(fAL_ls, dN1_ee, trim_lmax; extra_dims="yy")

    dN1_te = load_matrix_cached(joinpath(like_corrs_dir, "N1der_TE_lmin600_lmax3000_full.txt"))
    dN1_te = julia_standardize(fAL_ls, dN1_te, trim_lmax; extra_dims="yy")

    dN1_bb = load_matrix_cached(joinpath(like_corrs_dir, "N1der_BB_lmin600_lmax3000_full.txt"))
    dN1_bb = julia_standardize(fAL_ls, dN1_bb, trim_lmax; extra_dims="yy")

    # Planck N1
    dN1_kk_planck = load_matrix_cached(joinpath(like_corrs_dir, "N1_planck_der_KK_lmin100_lmax2048.txt"))
    dN1_kk_planck = julia_standardize(fAL_planck_data[:, 1], dN1_kk_planck, trim_lmax; extra_dims="yy")

    dN1_tt_planck = load_matrix_cached(joinpath(like_corrs_dir, "N1_planck_der_TT_lmin100_lmax2048.txt"))
    dN1_tt_planck = julia_standardize(fAL_planck_data[:, 1], dN1_tt_planck, trim_lmax; extra_dims="yy")

    dN1_ee_planck = load_matrix_cached(joinpath(like_corrs_dir, "N1_planck_der_EE_lmin100_lmax2048.txt"))
    dN1_ee_planck = julia_standardize(fAL_planck_data[:, 1], dN1_ee_planck, trim_lmax; extra_dims="yy")

    dN1_te_planck = load_matrix_cached(joinpath(like_corrs_dir, "N1_planck_der_TE_lmin100_lmax2048.txt"))
    dN1_te_planck = julia_standardize(fAL_planck_data[:, 1], dN1_te_planck, trim_lmax; extra_dims="yy")

    dN1_bb_planck = load_matrix_cached(joinpath(like_corrs_dir, "N1_planck_der_BB_lmin100_lmax2048.txt"))
    dN1_bb_planck = julia_standardize(fAL_planck_data[:, 1], dN1_bb_planck, trim_lmax; extra_dims="yy")

    return ACTPlanckSPTLensing(
        Float64.(data_vector),
        cov_chol,
        Float64.(binmat_act),
        Float64.(binmat_planck),
        Float64.(binmat_spt),
        true,
        Float64.(fiducial_cl_tt),
        Float64.(fiducial_cl_ee),
        Float64.(fiducial_cl_te),
        Float64.(fiducial_cl_bb),
        Float64.(fiducial_cl_kk),
        Float64.(fAL),
        Float64.(fAL_planck),
        Float64.(mask_l2),
        Float64.(dN1_kk),
        Float64.(dN1_tt),
        Float64.(dN1_ee),
        Float64.(dN1_te),
        Float64.(dN1_bb),
        Float64.(dAL_dC_tt),
        Float64.(dAL_dC_ee),
        Float64.(dAL_dC_bb),
        Float64.(dAL_dC_te),
        Float64.(dN1_kk_planck),
        Float64.(dN1_tt_planck),
        Float64.(dN1_ee_planck),
        Float64.(dN1_te_planck),
        Float64.(dN1_bb_planck),
        Float64.(dAL_dC_planck_tt),
        Float64.(dAL_dC_planck_ee),
        Float64.(dAL_dC_planck_bb),
        Float64.(dAL_dC_planck_te)
    )
end

"""
    pp_to_kk(cl_pp::AbstractVector)

Convert the CMB lensing potential spectrum C_ℓ^ϕϕ (cl_pp) to CMB lensing convergence spectrum C_ℓ^kk (cl_kk).
"""
function pp_to_kk(cl_pp::AbstractVector)
    ell = 0:(length(cl_pp)-1)
    return cl_pp .* (ell .* (ell .+ 1.0)).^2 ./ 4.0
end

"""
    get_corrected_clkk(like::ACTPlanckSPTLensing, cl_kk, cl_tt, cl_ee, cl_te, cl_bb; is_planck=false, act_calib=false)

Compute the lensed convergence spectrum after normalization and N1 corrections.
"""
function get_corrected_clkk(
    like::ACTPlanckSPTLensing, cl_kk, cl_tt, cl_ee, cl_te, cl_bb;
    is_planck=false, act_calib=false, do_norm_corr=true, do_N1kk_corr=true, do_N1cmb_corr=true
)
    # 1. N1 kk correction
    cl_kk_fid = like.fiducial_cl_kk
    N1_kk_corr = if do_N1kk_corr
        dN1_kk = is_planck ? like.dN1_kk_planck : like.dN1_kk
        dN1_kk * (cl_kk .- cl_kk_fid)
    else
        zeros(eltype(cl_kk), length(cl_kk))
    end

    # 2. Calibration factor (cal_fact)
    cal_fact = if act_calib && !is_planck
        sum(cl_tt[1002:2000] ./ like.fiducial_cl_tt[1002:2000]) / length(1002:2000)
    else
        one(eltype(cl_tt))
    end

    # 3. CMB spectra diffs and loop over specs
    N1_cmb_corr = zeros(eltype(cl_kk), length(cl_kk))
    norm_corr = zeros(eltype(cl_kk), length(cl_kk))

    fid_norm = is_planck ? like.fAL_planck : like.fAL
    fid_norm_safe = fid_norm .+ (fid_norm .== 0.0)
    mask = like.mask_l2

    # Unrolled computations to support Reactant JIT compilation and autodiff
    
    # TT
    cldiff_tt = (cl_tt ./ cal_fact) .- like.fiducial_cl_tt
    if do_N1cmb_corr
        dN1_tt = is_planck ? like.dN1_tt_planck : like.dN1_tt
        N1_cmb_corr = N1_cmb_corr .+ dN1_tt * cldiff_tt
    end
    if do_norm_corr
        dAL_dC_tt = is_planck ? like.dAL_dC_planck_tt : like.dAL_dC_tt
        c_tt = -2.0 .* (dAL_dC_tt * cldiff_tt)
        c_tt_masked = c_tt ./ fid_norm_safe
        norm_corr = norm_corr .+ (c_tt_masked .* mask)
    end

    # EE
    cldiff_ee = (cl_ee ./ cal_fact) .- like.fiducial_cl_ee
    if do_N1cmb_corr
        dN1_ee = is_planck ? like.dN1_ee_planck : like.dN1_ee
        N1_cmb_corr = N1_cmb_corr .+ dN1_ee * cldiff_ee
    end
    if do_norm_corr
        dAL_dC_ee = is_planck ? like.dAL_dC_planck_ee : like.dAL_dC_ee
        c_ee = -2.0 .* (dAL_dC_ee * cldiff_ee)
        c_ee_masked = c_ee ./ fid_norm_safe
        norm_corr = norm_corr .+ (c_ee_masked .* mask)
    end

    # BB
    cldiff_bb = (cl_bb ./ cal_fact) .- like.fiducial_cl_bb
    if do_N1cmb_corr
        dN1_bb = is_planck ? like.dN1_bb_planck : like.dN1_bb
        N1_cmb_corr = N1_cmb_corr .+ dN1_bb * cldiff_bb
    end
    if do_norm_corr
        dAL_dC_bb = is_planck ? like.dAL_dC_planck_bb : like.dAL_dC_bb
        c_bb = -2.0 .* (dAL_dC_bb * cldiff_bb)
        c_bb_masked = c_bb ./ fid_norm_safe
        norm_corr = norm_corr .+ (c_bb_masked .* mask)
    end

    # TE
    cldiff_te = (cl_te ./ cal_fact) .- like.fiducial_cl_te
    if do_N1cmb_corr
        dN1_te = is_planck ? like.dN1_te_planck : like.dN1_te
        N1_cmb_corr = N1_cmb_corr .+ dN1_te * cldiff_te
    end
    if do_norm_corr
        dAL_dC_te = is_planck ? like.dAL_dC_planck_te : like.dAL_dC_te
        c_te = -2.0 .* (dAL_dC_te * cldiff_te)
        c_te_masked = c_te ./ fid_norm_safe
        norm_corr = norm_corr .+ (c_te_masked .* mask)
    end

    cl_kk_corr = cl_kk .+ norm_corr .* cl_kk_fid .+ N1_kk_corr .+ N1_cmb_corr
    return cl_kk_corr
end

"""
    build_model_vector(like::ACTPlanckSPTLensing, cl_pp, cl_tt=nothing, cl_ee=nothing, cl_te=nothing, cl_bb=nothing; kwargs...)

Build the model binned lensing convergence band powers vector.
If CMB spectra are provided and `load_corrections = true`, active CMB corrections are applied.
"""
function build_model_vector(
    like::ACTPlanckSPTLensing, cl_pp, cl_tt=nothing, cl_ee=nothing, cl_te=nothing, cl_bb=nothing;
    act_calib=false, do_norm_corr=true, do_N1kk_corr=true, do_N1cmb_corr=true
)
    cl_kk = pp_to_kk(cl_pp)

    cl_kk_3000 = cl_kk[1:3000]
    cl_kk_3102 = cl_kk[1:3102]

    if like.has_corrections && !isnothing(cl_tt) && !isnothing(cl_ee) && !isnothing(cl_te) && !isnothing(cl_bb)
        # Apply corrections
        cl_kk_act = get_corrected_clkk(
            like, cl_kk_3000, cl_tt[1:3000], cl_ee[1:3000], cl_te[1:3000], cl_bb[1:3000];
            is_planck=false, act_calib=act_calib, do_norm_corr=do_norm_corr, do_N1kk_corr=do_N1kk_corr, do_N1cmb_corr=do_N1cmb_corr
        )
        
        cl_kk_planck = get_corrected_clkk(
            like, cl_kk_3000, cl_tt[1:3000], cl_ee[1:3000], cl_te[1:3000], cl_bb[1:3000];
            is_planck=true, act_calib=act_calib, do_norm_corr=do_norm_corr, do_N1kk_corr=do_N1kk_corr, do_N1cmb_corr=do_N1cmb_corr
        )
        
        cl_kk_spt = cl_kk_3102

        bclkk_act = like.binmat_act * cl_kk_act
        bclkk_planck = like.binmat_planck * cl_kk_planck
        bclkk_spt = like.binmat_spt * cl_kk_spt
        
        return vcat(bclkk_act, bclkk_planck, bclkk_spt)
    else
        bclkk_act = like.binmat_act * cl_kk_3000
        bclkk_planck = like.binmat_planck * cl_kk_3000
        bclkk_spt = like.binmat_spt * cl_kk_3102
        return vcat(bclkk_act, bclkk_planck, bclkk_spt)
    end
end

"""
    loglike(like::ACTPlanckSPTLensing, cl_pp, cl_tt=nothing, cl_ee=nothing, cl_te=nothing, cl_bb=nothing; kwargs...) -> Float64

Compute the Gaussian log-likelihood: -0.5 × χ².
"""
function loglike(
    like::ACTPlanckSPTLensing, cl_pp, cl_tt=nothing, cl_ee=nothing, cl_te=nothing, cl_bb=nothing;
    kwargs...
)
    model = build_model_vector(like, cl_pp, cl_tt, cl_ee, cl_te, cl_bb; kwargs...)
    return gaussian_loglike(like.cov_chol, like.data_vector, model)
end

"""
    chi2(like::ACTPlanckSPTLensing, cl_pp, cl_tt=nothing, cl_ee=nothing, cl_te=nothing, cl_bb=nothing; kwargs...) -> Float64

Compute the χ² value for the joint lensing likelihood.
"""
function chi2(
    like::ACTPlanckSPTLensing, cl_pp, cl_tt=nothing, cl_ee=nothing, cl_te=nothing, cl_bb=nothing;
    kwargs...
)
    model = build_model_vector(like, cl_pp, cl_tt, cl_ee, cl_te, cl_bb; kwargs...)
    return gaussian_chi2(like.cov_chol, like.data_vector .- model)
end

function Base.show(io::IO, like::ACTPlanckSPTLensing)
    n = length(like.data_vector)
    println(io, "ACTPlanckSPTLensing")
    println(io, "  Bins: ACT=10, Planck=9, SPT=16 = $n total")
    println(io, "  CMB Corrections: $(like.has_corrections ? "Active" : "Disabled")")
    print(io,   "  Data vector length: $n")
end

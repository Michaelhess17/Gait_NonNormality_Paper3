#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "src", "GaitNonNormalityPaper3.jl"))

using .GaitNonNormalityPaper3
using CairoMakie
using CSV
using DataFrames
using DMDAnalysis
using LaTeXStrings
using LinearAlgebra
using Printf
using Random
using SLIP
using Statistics

const BLUE = RGBf(0.18, 0.43, 0.72)
const RED = RGBf(0.72, 0.16, 0.18)
const GREEN = RGBf(0.18, 0.54, 0.34)
const GOLD = RGBf(0.86, 0.56, 0.12)
const GRAY = RGBf(0.25, 0.25, 0.25)
const HEATMAP_CMAP = :viridis
const STANCE_LEG = RIGHT
const DT = 0.002
const FS = 1 / DT
const N_DELAYS = 36
# Use five total trajectories for the representative DMD fits: the nominal
# trajectory plus four nearby initial conditions.
const N_PERTURB = 4
const PHASE_STEPS = 40
const RHO_ANALYSIS_MAX = 1.2
# Evaluate Figure 3G just outside the unit circle.  Exact unit-circle
# evaluation is dominated by many stabilized near-unit Hankel-DMD modes, while
# inside-circle evaluation makes those clamped modes reappear as large noisy
# trial-to-trial spikes.  A small outside radius gives a finite response
# envelope without the over-smoothed look of the earlier r=1.02 setting.
const RESOLVENT_PROFILE_RADIUS = 1.005
const SLIP_METRIC_VERSION = 4  # v4: single-channel (y) embedding, one-step TG horizon, step-section return map
const SLIP_DMD_FIXED_ND = 72
# Truncation rank for every SLIP operator fit. Was EnergyRank(1e-6), which in
# this codebase never actually truncates -- it returns the full Hankel row
# count, so a 72-delay fit carried 432 states. A sweep over
# (n_delays, rank) scored by leave-one-out identity R^2 of the response-set
# decoder (scripts/S58_slip_config_sweep.jl, 60 stable grid points) found every
# configuration between 0.33 and 0.47, with rank 10 best at this embedding
# (0.468 vs 0.372 for the old setting). Fixed rank also matches scripts 05, 06
# and 08, which all use FixedRank.
const SLIP_DMD_RANK_N = parse(Int, get(ENV, "SLIP3_RANK", "10"))
const SLIP_DMD_RANK = FixedRank(SLIP_DMD_RANK_N)

"""Run a named stage and print elapsed wall time immediately."""
function timed_stage(f::Function, label::AbstractString)
    println("[figure 3] starting: ", label)
    flush(stdout)
    t0 = time()
    value = f()
    @printf("[figure 3] finished: %s (%.1f s)\n", label, time() - t0)
    flush(stdout)
    return value
end

function parse_grid(spec)
    vals = split(spec, ":")
    length(vals) == 3 || error("Grid spec must be start:step:stop, got $spec")
    a, s, b = parse.(Float64, vals)
    return collect(a:s:b)
end

const AOA_FINE = parse_grid(get(ENV, "SLIP_AOA_FINE", "55:1:85"))
const KTILDE_FINE = parse_grid(get(ENV, "SLIP_KTILDE_FINE", "5:1:35"))
const AOA_METRIC = parse_grid(get(ENV, "SLIP_AOA_METRIC", "55:1:85"))
const KTILDE_METRIC = parse_grid(get(ENV, "SLIP_KTILDE_METRIC", "5:1:35"))
const FORCE_RECOMPUTE = lowercase(get(ENV, "FORCE_SLIP3", "false")) in ("1", "true", "yes")

Ebar_of_ktilde(ktilde) = -0.054 * log10(ktilde)^2 + 0.189 * log10(ktilde) + 0.898

function grid_tag(aoa_grid, ktilde_grid)
    astep = length(aoa_grid) > 1 ? aoa_grid[2] - aoa_grid[1] : 0.0
    kstep = length(ktilde_grid) > 1 ? ktilde_grid[2] - ktilde_grid[1] : 0.0
    return @sprintf("aoa%.1f-%.1f-step%.2f_k%.1f-%.1f-step%.2f_r%d",
        first(aoa_grid), last(aoa_grid), astep, first(ktilde_grid), last(ktilde_grid), kstep,
        SLIP_DMD_RANK_N)
end

function base_params(ktilde, aoa)
    params = biped_slip_default_params()
    params.kth = [0.0, 0.0]
    params.th0 = [0.0, 0.0]
    params.k = ktilde .* params.m .* params.g ./ params.L0
    params.th_contact = deg2rad.([90 - aoa, 90 - aoa])
    return params
end

function find_grid_fixed_points(aoa_grid, ktilde_grid)
    rho = fill(NaN, length(ktilde_grid), length(aoa_grid))
    zstar = Matrix{Union{Nothing,Vector{Float64}}}(nothing, length(ktilde_grid), length(aoa_grid))
    fallback = [0.98, 0.0]
    for (iy, ktilde) in enumerate(ktilde_grid)
        Ebar = Ebar_of_ktilde(ktilde)
        for (ix, aoa) in enumerate(aoa_grid)
            params = base_params(ktilde, aoa)
            guesses = Vector{Float64}[]
            for (jx, jy) in ((ix - 1, iy), (ix, iy - 1))
                jx >= 1 && jy >= 1 && zstar[jy, jx] !== nothing && rho[jy, jx] < 1 &&
                    push!(guesses, zstar[jy, jx])
            end
            for (jx, jy) in ((ix - 1, iy), (ix, iy - 1))
                jx >= 1 && jy >= 1 && zstar[jy, jx] !== nothing && push!(guesses, zstar[jy, jx])
            end
            push!(guesses, fallback)
            for guess in guesses
                try
                    gs = gait_stability_step(guess[1], guess[2], Ebar, STANCE_LEG, params;
                        opts=SimOptions(t_max=5.0, max_steps=3, dt=0.0005), maxiter=24)
                    rho[iy, ix] = gs.spectral_radius
                    zstar[iy, ix] = gs.z_star
                    break
                catch
                end
            end
            print(isfinite(rho[iy, ix]) ? (rho[iy, ix] < 1 ? "S" : "u") : ".")
        end
        println()
    end
    return rho, zstar
end

function fixed_points_dataframe(aoa_grid, ktilde_grid, rho, zstar)
    rows = NamedTuple[]
    for (iy, ktilde) in enumerate(ktilde_grid), (ix, aoa) in enumerate(aoa_grid)
        z = zstar[iy, ix]
        z === nothing && continue
        push!(rows, (
            aoa_from_horizontal=aoa,
            ktilde=ktilde,
            spectral_radius=rho[iy, ix],
            stable=rho[iy, ix] < 1,
            y_star=z[1],
            theta_star=z[2],
        ))
    end
    return DataFrame(rows)
end

function load_or_compute_fixed_points()
    ensure_dirs!()
    tag = grid_tag(AOA_FINE, KTILDE_FINE)
    path = joinpath(RESULTS_DIR, "figure3_fixed_points_$tag.csv")
    if isfile(path) && !FORCE_RECOMPUTE
        println("Loading cached fixed points: $path")
        df = CSV.read(path, DataFrame)
    else
        println("Finding SLIP fixed points over fine grid...")
        rho, zstar = find_grid_fixed_points(AOA_FINE, KTILDE_FINE)
        df = fixed_points_dataframe(AOA_FINE, KTILDE_FINE, rho, zstar)
        CSV.write(path, df)
        println("Wrote fixed points: $path")
    end
    return df
end

function matrix_from_fixed_points(df, aoa_grid, ktilde_grid, col)
    M = fill(NaN, length(ktilde_grid), length(aoa_grid))
    for row in eachrow(df)
        ix = findfirst(x -> isapprox(x, row.aoa_from_horizontal; atol=1e-8), aoa_grid)
        iy = findfirst(x -> isapprox(x, row.ktilde; atol=1e-8), ktilde_grid)
        ix === nothing || iy === nothing || (M[iy, ix] = row[col])
    end
    return M
end

function fixed_point_lookup(df, aoa, ktilde)
    sub = filter(row -> isapprox(row.aoa_from_horizontal, aoa; atol=1e-8) &&
                       isapprox(row.ktilde, ktilde; atol=1e-8), df)
    nrow(sub) == 0 && return nothing
    row = sub[1, :]
    return (rho=row.spectral_radius, z=[row.y_star, row.theta_star], stable=row.stable)
end

function reduced_trajectory(params, z, Ebar, y_offset, theta_offset; max_steps=12, t_max=10.0)
    E = Ebar * params.L0[STANCE_LEG] * params.m * params.g
    ic = biped_slip_energy_ic(params, STANCE_LEG, z[1] + y_offset, E, z[2] + theta_offset)
    result = simulate_biped_slip(params, ic, SimOptions(t_max=t_max, max_steps=max_steps, dt=DT))
    size(result.X, 1) < 100 && return nothing
    # SINGLE-CHANNEL delay embedding: CoM height only. In a delay embedding
    # ydot is close to a finite difference of delayed y, so the second channel
    # is largely redundant; a sweep over channel sets found y alone at low rank
    # matches the two-channel fit. The return map keeps the full 2-D section
    # state (see `vlo_phase_cycles`), which it needs to estimate a 2-D map.
    y = result.X[:, 2]
    return (trial=reshape(y .- mean(y), :, 1), result=result)
end

function interp_at_times(X::AbstractMatrix, t::AbstractVector, query_times::AbstractVector)
    out = Matrix{Float64}(undef, length(query_times), size(X, 2))
    for (j, tq) in enumerate(query_times)
        i = clamp(searchsortedlast(t, tq), 1, length(t) - 1)
        frac = (tq - t[i]) / max(t[i + 1] - t[i], eps())
        out[j, :] = X[i, :] .+ frac .* (X[i + 1, :] .- X[i, :])
    end
    return out
end

# `leg=nothing` sections on EVERY VLO crossing, i.e. one STEP per cycle; passing
# a specific leg sections same-leg crossings, i.e. one STRIDE (two steps).
function vlo_phase_cycles(result::BipedSLIPResults; steps=PHASE_STEPS, leg=STANCE_LEG)
    X = hcat(result.X[:, 2] .- mean(result.X[:, 2]), result.X[:, 4] .- mean(result.X[:, 4]))
    events = extract_vlo_events(result)
    vlo_times = sort([e.t for e in events if leg === nothing || e.leg == leg])
    length(vlo_times) < 2 && return Matrix{Float64}[]
    frac = collect(range(0, 1; length=steps + 1))[1:end-1]
    cycles = Matrix{Float64}[]
    for i in 1:(length(vlo_times) - 1)
        t0, t1 = vlo_times[i], vlo_times[i + 1]
        (t0 < result.T[1] || t1 > result.T[end]) && continue
        push!(cycles, interp_at_times(X, result.T, t0 .+ frac .* (t1 - t0)))
    end
    return cycles
end

function phase_trials(results::Vector{BipedSLIPResults})
    trials = Matrix{Float64}[]
    for result in results
        append!(trials, vlo_phase_cycles(result))
    end
    return [tr for tr in trials if size(tr, 1) >= PHASE_STEPS]
end

function perturbed_trials(params, z, Ebar; rng=MersenneTwister(11), max_steps=40, t_max=20.0,
        verbose=false)
    trials = Matrix{Float64}[]
    results = BipedSLIPResults[]
    offsets = [(0.0, 0.0)]
    for _ in 1:N_PERTURB
        push!(offsets, (0.003 * randn(rng), deg2rad(0.20) * randn(rng)))
    end
    for (i, (dy, dtheta)) in enumerate(offsets)
        verbose && println("[figure 3]   simulating training trajectory $i/$(length(offsets))")
        verbose && flush(stdout)
        t0 = time()
        try
            out = reduced_trajectory(params, z, Ebar, dy, dtheta; max_steps=max_steps, t_max=t_max)
            if out === nothing
                verbose && @printf("[figure 3]   trajectory %d returned no usable data (%.1f s)\n", i, time() - t0)
                continue
            end
            push!(trials, out.trial)
            push!(results, out.result)
            verbose && @printf("[figure 3]   trajectory %d complete: %d samples, %.1f s\n",
                i, size(out.trial, 1), time() - t0)
        catch err
            verbose && println("[figure 3]   trajectory $i failed after $(@sprintf("%.1f", time() - t0)) s: $err")
        end
        verbose && flush(stdout)
    end
    return trials, results
end

function short_noisy_trials(trials; n_keep=240, noise=0.002)
    rng = MersenneTwister(901)
    out = Matrix{Float64}[]
    for tr in trials
        n = min(n_keep, size(tr, 1))
        n < N_DELAYS + 2 && continue
        push!(out, tr[1:n, :] .+ noise .* randn(rng, n, size(tr, 2)))
    end
    return out
end

function fit_slip_operator(trials; stabilize=true)
    length(trials) < 3 && return nothing
    try
        hankel = HankelGenerator(n_delays=N_DELAYS, delay_interval=1)
        model = DMD(rank=SLIP_DMD_RANK, stabilize=stabilize)
        return fit(model, trials, hankel)
    catch
        return nothing
    end
end

spectral_radius(A) = maximum(abs.(eigvals(A)))

"""NOT in the response-set decoder -- see the note below. Numerical radius:
max over theta of the largest eigenvalue of the Hermitian
part of e^{-i theta} A. Unlike the spectral radius it is NOT pinned to the unit
circle by a limit cycle's neutral directions, and unlike sup_k ||A^k|| it needs
no horizon. r(A) > 1 certifies transient growth, and ||A^k|| <= 2 r(A)^k.
Computed and written to the metrics table for reference, but deliberately NOT
added to the response-set decoder. It recovers the true Floquet multiplier well
in a phase-normalised, rank-20 pipeline (r = 0.52, leave-one-out R2 = 0.236
alone), but in THIS pipeline it carries essentially no information on its own
(corr with the true decay rate = +0.02, leave-one-out R2 = -0.013) and the
figure's own identity R2 for the response set fell from 0.291 to 0.185 when it
was included. Whatever it gains in combination here is suppressor behaviour,
not signal."""

function numerical_radius(A::AbstractMatrix; n_angles::Int=120)
    best = 0.0
    for th in range(0, 2pi; length=n_angles + 1)[1:end-1]
        B = cis(-th) .* A
        H = (B + B') ./ 2
        best = max(best, maximum(real.(eigvals(Hermitian(Matrix(H))))))
    end
    return best
end
numerical_radius(result; n_angles::Int=120) = numerical_radius(result.A; n_angles=n_angles)

"""Largest per-STEP eigenvalue modulus that is not sitting on the unit circle:
the candidate "classical stability number" once the neutral, cycle-carrying
(Fourier) modes are excluded. A fitted operator on a limit cycle has
|lambda| ~ 1 per SAMPLE for the on-cycle modes, so the moduli are first raised
to fs/f_step to put them in per-step units, then anything within `gate` of 1
is discarded. Mirrors `transverse_per_step` in
scripts/S30_transverse_eigenvalue_decoder.jl."""
function transverse_per_step(ev::AbstractVector{<:Complex}, f_step, fs; gate::Real=1e-3)
    (isfinite(f_step) && f_step > 0) || return NaN
    rho = abs.(ev) .^ (fs / f_step)
    keep = isfinite.(rho) .& (rho .< 1 - gate) .& (rho .> 0)
    any(keep) || return NaN
    return maximum(rho[keep])
end

function transient_growth_diag(A::AbstractMatrix; kmax::Int)
    n = size(A, 1)
    Ak = Matrix{Float64}(I, n, n)
    best = opnorm(Ak, 2)
    best_k = 0
    for k in 1:kmax
        Ak = A * Ak
        v = opnorm(Ak, 2)
        if v > best
            best, best_k = v, k
        end
    end
    return best, best_k, best_k >= kmax
end
transient_growth_diag(result; kmax::Int) = transient_growth_diag(result.A; kmax=kmax)

function interp_linear(x::AbstractVector, y::AbstractVector, xq::Real)
    ok = isfinite.(x) .& isfinite.(y)
    count(ok) < 2 && return NaN
    xk = Float64.(x[ok])
    yk = Float64.(y[ok])
    order = sortperm(xk)
    xk = xk[order]
    yk = yk[order]
    (xq < first(xk) || xq > last(xk)) && return NaN
    j = searchsortedlast(xk, Float64(xq))
    j >= length(xk) && return yk[end]
    x1, x2 = xk[j], xk[j + 1]
    y1, y2 = yk[j], yk[j + 1]
    x2 == x1 && return y1
    t = (Float64(xq) - x1) / (x2 - x1)
    return (1 - t) * y1 + t * y2
end

function positive_log_mean_sem(vals)
    v = filter(x -> isfinite(x) && x > 0, Float64.(vals))
    isempty(v) && return (NaN, NaN, NaN)
    lv = log.(v)
    m = mean(lv)
    e = length(lv) > 1 ? std(lv) / sqrt(length(lv)) : 0.0
    return exp(m), exp(m - e), exp(m + e)
end

function profile_stats(mat)
    meanv = fill(NaN, size(mat, 2))
    lo = similar(meanv)
    hi = similar(meanv)
    for j in axes(mat, 2)
        meanv[j], lo[j], hi[j] = positive_log_mean_sem(@view mat[:, j])
    end
    return meanv, lo, hi
end

function regularized_resolvent_profile(result; n_angles=2000, fs=FS,
        radius=RESOLVENT_PROFILE_RADIUS, fmin=0.0)
    A = result.A
    n = size(A, 1)
    freqs = collect(range(fmin, fs / 2; length=n_angles))
    gains = similar(freqs)
    eye = Matrix{ComplexF64}(I, n, n)
    Ac = ComplexF64.(A)
    for (i, f) in enumerate(freqs)
        z = radius * exp(2im * pi * f / fs)
        gains[i] = 1 / max(minimum(svdvals(z .* eye .- Ac)), 1e-12)
    end
    return freqs, gains
end

function fit_pooled_slip_operator(trials::Vector{Matrix{Float64}}, fs; phase=NoPhaseNormalization(),
        stabilize=true, nd_scale=1.0, fixed_nd=nothing, rank_rule=SLIP_DMD_RANK)
    isempty(trials) && return nothing
    try
        pre = Preprocessor(normalization=NoNormalization(), state_mode=:position, phase=phase, fs=fs)
        Xp = DMDAnalysis.transform(pre, trials)
        fs_eff = phase isa PhaseNormalize ? phase.steps : fs
        f0s = [estimate_stride_frequency(t; fs=fs_eff, fmin=0.5, fmax=6.0) for t in Xp.trials]
        finite_f0s = filter(isfinite, f0s)
        isempty(finite_f0s) && return nothing
        f0 = median(finite_f0s)
        n_time_min = minimum(size(t, 1) for t in Xp.trials)
        cap = max(8, floor(Int, 0.4 * n_time_min))
        nd = fixed_nd === nothing ?
            (isfinite(f0) && f0 > 0 ? round(Int, nd_scale * fs_eff / f0) : 8) :
            Int(fixed_nd)
        nd = clamp(nd, 8, cap)
        hankel = HankelGenerator(n_delays=nd, delay_interval=1)
        model = DMD(rank=rank_rule, stabilize=stabilize)
        H = DMDAnalysis.transform(hankel, Xp)
        return fit(model, H)
    catch
        return nothing
    end
end

effective_rank(s) = (sum(s)^2) / max(sum(abs2, s), eps())

function template_prediction_error(trial, f0; n_steps)
    isfinite(f0) && f0 > 0 || return NaN
    period = clamp(round(Int, FS / f0), 8, max(8, div(size(trial, 1), 2)))
    n = min(n_steps, size(trial, 1))
    y = trial[1:n, :]
    yhat = similar(y)
    for i in 1:n
        yhat[i, :] .= trial[mod1(i, period), :]
    end
    rmse = sqrt(mean(abs2, yhat .- y))
    return rmse / max(sqrt(mean(abs2, y)), eps())
end

# Exact STEP frequency from the nominal trajectory's VLO crossings (both legs).
# This matches the single-step map that `gait_stability_step` linearises, and is
# what sets the one-cycle transient-growth horizon.
function step_frequency_from_results(results::Vector{BipedSLIPResults})
    isempty(results) && return NaN
    ev = sort([e.t for e in extract_vlo_events(results[1]) if e.t > 1e-7])
    length(ev) < 3 && return NaN
    dt_step = median(diff(ev))
    return dt_step > 0 ? 1 / dt_step : NaN
end

# Transient-growth horizon in operator steps spanning exactly ONE STEP CYCLE.
# The operator advances one sample per step, so this is fs/f_step.
function one_cycle_steps(f_step, fs; fallback::Int=144)
    isfinite(f_step) && f_step > 0 || return fallback
    return clamp(round(Int, fs / f_step), 8, 4000)
end

function metric_tuple(result, trials, rho; raw_radius=NaN, degraded=false,
        fs_metric=FS, f0_metric=NaN, f_step=NaN, raw_eigs=nothing)
    result === nothing && return nothing
    trial = trials[1]
    f0 = isfinite(f0_metric) && f0_metric > 0 ? f0_metric :
        estimate_stride_frequency(trial; fs=FS, fmin=0.2, fmax=3.0)
    peak = resolvent_peak(result; n_angles=240, fs=fs_metric, fmin=0.2)
    reg_f, reg_g = regularized_resolvent_profile(result; n_angles=240, fs=fs_metric, fmin=0.2)
    ok_reg = isfinite.(reg_f) .& isfinite.(reg_g) .& (reg_g .> 0)
    reg_peak_gain = any(ok_reg) ? maximum(reg_g[ok_reg]) : NaN
    reg_peak_frequency = any(ok_reg) ? reg_f[ok_reg][argmax(reg_g[ok_reg])] : NaN
    harmonics = isfinite(f0) ? harmonic_gains(result, f0; n_harmonics=5, fs=fs_metric) : fill(NaN, 5)
    horizon = isfinite(f0) && f0 > 0 ? min(round(Int, 2fs_metric / f0), size(trial, 1) - N_DELAYS - 2) : 300
    horizon = max(40, horizon)
    pred = rollout_prediction_error(result, trial; n_steps=horizon)
    template = template_prediction_error(trial, f0; n_steps=horizon)
    return (
        dmd_radius=raw_radius,
        eig_radius_error=abs(raw_radius - rho),
        henrici=henrici_departure(result),
        numerical_radius=numerical_radius(result),
        transverse_rho=raw_eigs === nothing ? NaN :
            transverse_per_step(raw_eigs, f_step, fs_metric),
        log_tgmax=log10(transient_growth_diag(result;
            kmax=one_cycle_steps(f_step, fs_metric; fallback=2 * result.hankel.n_delays))[1]),
        tg_horizon_steps=one_cycle_steps(f_step, fs_metric; fallback=2 * result.hankel.n_delays),
        step_frequency=f_step,
        resolvent_peak_gain=peak.gain,
        resolvent_peak_frequency=peak.frequency,
        regularized_resolvent_peak_gain=reg_peak_gain,
        regularized_resolvent_peak_frequency=reg_peak_frequency,
        stride_frequency=f0,
        h1_gain=harmonics[1],
        h2_over_h1=harmonics[1] > 0 ? harmonics[2] / harmonics[1] : NaN,
        harmonic_entropy=harmonic_entropy(harmonics),
        qfactor_h1=NaN,
        participation=effective_rank(result.singular_values),
        prediction_nrmse=pred.nrmse,
        template_nrmse=template,
        degraded=degraded,
    )
end

function collect_section_pairs(cycles_list::Vector{Vector{Matrix{Float64}}}, k::Int)
    pin = Vector{Vector{Float64}}()
    pout = Vector{Vector{Float64}}()
    for cycles in cycles_list
        length(cycles) < 2 && continue
        for c in 1:(length(cycles) - 1)
            push!(pin, vec(cycles[c][k, :]))
            push!(pout, vec(cycles[c + 1][k, :]))
        end
    end
    return pin, pout
end

function returnmap_rho_from_results(results::Vector{BipedSLIPResults})
    # STEP sections, to match the single-step ground truth from
    # `gait_stability_step`. Sectioning on one leg only would span a stride and
    # estimate rho^2 against a rho truth.
    cycles_list = [vlo_phase_cycles(r; leg=nothing) for r in results]
    section_idxs = unique(round.(Int, [1, PHASE_STEPS / 4 + 1, PHASE_STEPS / 2 + 1, 3PHASE_STEPS / 4 + 1]))
    estimates = Float64[]
    for k in section_idxs
        pin, pout = collect_section_pairs(cycles_list, k)
        length(pin) < 6 && continue
        pin_mat = reduce(hcat, pin)'
        pout_mat = reduce(hcat, pout)'
        o = vec(mean(vcat(pin_mat, pout_mat); dims=1))
        Xin = pin_mat .- o'
        Xout = pout_mat .- o'
        Beta = Matrix(Xin) \ Matrix(Xout)
        push!(estimates, maximum(abs.(eigvals(Beta))))
    end
    return isempty(estimates) ? NaN : median(estimates)
end

function section_spectral_radius(pin::Vector{Vector{Float64}}, pout::Vector{Vector{Float64}})
    length(pin) < 6 && return NaN
    Xi = reduce(hcat, pin)
    Yo = reduce(hcat, pout)
    offset = vec(mean(hcat(Xi, Yo); dims=2))
    Xi = Xi .- offset
    Yo = Yo .- offset
    return maximum(abs.(eigvals(Yo * pinv(Xi))))
end

function returnmap_section_rhos(results::Vector{BipedSLIPResults})
    step_in = Vector{Float64}[]
    step_out = Vector{Float64}[]
    stride_in = Vector{Float64}[]
    stride_out = Vector{Float64}[]
    for result in results
        events = extract_vlo_events(result)
        Xsection = hcat(result.X[:, 2], result.X[:, 4])
        all_times = sort([e.t for e in events])
        if length(all_times) >= 2
            S = interp_at_times(Xsection, result.T, all_times)
            for i in 1:(size(S, 1) - 1)
                push!(step_in, vec(S[i, :]))
                push!(step_out, vec(S[i + 1, :]))
            end
        end
        stance_times = sort([e.t for e in events if e.leg == STANCE_LEG])
        if length(stance_times) >= 2
            S = interp_at_times(Xsection, result.T, stance_times)
            for i in 1:(size(S, 1) - 1)
                push!(stride_in, vec(S[i, :]))
                push!(stride_out, vec(S[i + 1, :]))
            end
        end
    end
    rho_step = section_spectral_radius(step_in, step_out)
    rho_stride = section_spectral_radius(stride_in, stride_out)
    return (step=rho_step, stride=rho_stride,
        stride_sqrt=isfinite(rho_stride) && rho_stride >= 0 ? sqrt(rho_stride) : NaN)
end

function slip_metric_rows(fixed_df)
    tag = grid_tag(AOA_METRIC, KTILDE_METRIC)
    path = joinpath(RESULTS_DIR, "figure3_slip_metrics_$tag.csv")
    if isfile(path) && !FORCE_RECOMPUTE
        println("Loading cached SLIP-DMD metrics: $path")
        df = CSV.read(path, DataFrame)
        if :returnmap_rho in propertynames(df) && :dmd_radius in propertynames(df) &&
                :metric_version in propertynames(df) && all(df.metric_version .== SLIP_METRIC_VERSION) &&
                any(isfinite, df.dmd_radius) && count(isfinite, df.returnmap_rho) >= 10 &&
                maximum(skipmissing(df.spectral_radius)) <= 1.0 + 1e-8
            return df
        end
        println("Cached metrics are stale (not stable-only or missing fields); recomputing SLIP comparisons.")
    end

    rows = NamedTuple[]
    rng = MersenneTwister(37)
    println("Fitting Paper3-local DMD metrics from simulated perturbation trials...")
    total_metric = count(row -> row.stable && isfinite(row.spectral_radius), eachrow(fixed_df))
    n_metric = 0
    for ktilde in KTILDE_METRIC, aoa in AOA_METRIC
        fp = fixed_point_lookup(fixed_df, aoa, ktilde)
        fp === nothing && continue
        fp.stable || continue
        n_metric += 1
        @printf("[figure 3] metric grid %d/%d: aoa=%.1f k̃=%.1f ρ=%.3f\n",
            n_metric, total_metric, aoa, ktilde, fp.rho)
        flush(stdout)
        params = base_params(ktilde, aoa)
        Ebar = Ebar_of_ktilde(ktilde)
        raw_trials, results = perturbed_trials(params, fp.z, Ebar; rng=rng)
        trials = raw_trials
        fs_trials = FS
        result = fit_pooled_slip_operator(trials, fs_trials; phase=NoPhaseNormalization(),
            stabilize=true, fixed_nd=SLIP_DMD_FIXED_ND)
        raw_result = fit_pooled_slip_operator(trials, fs_trials; phase=NoPhaseNormalization(),
            stabilize=false, fixed_nd=SLIP_DMD_FIXED_ND)
        raw_radius = raw_result === nothing ? NaN : spectral_radius(raw_result.A)
        raw_eigs = raw_result === nothing ? nothing : eigvals(raw_result.A)
        f_step = step_frequency_from_results(results)
        full = metric_tuple(result, trials, fp.rho; raw_radius=raw_radius, degraded=false,
            fs_metric=FS, f_step=f_step, raw_eigs=raw_eigs)
        full === nothing && continue
        rm_rho = returnmap_rho_from_results(results)
        rm_sections = returnmap_section_rhos(results)
        noisy = short_noisy_trials(trials; n_keep=240, noise=0.002)
        noisy_result = fit_pooled_slip_operator(noisy, fs_trials; phase=NoPhaseNormalization(),
            stabilize=true, fixed_nd=SLIP_DMD_FIXED_ND)
        noisy_raw = fit_pooled_slip_operator(noisy, fs_trials; phase=NoPhaseNormalization(),
            stabilize=false, fixed_nd=SLIP_DMD_FIXED_ND)
        noisy_radius = noisy_raw === nothing ? NaN : spectral_radius(noisy_raw.A)
        noisy_eigs = noisy_raw === nothing ? nothing : eigvals(noisy_raw.A)
        short = metric_tuple(noisy_result, noisy, fp.rho; raw_radius=noisy_radius, degraded=true,
            fs_metric=FS, f_step=f_step, raw_eigs=noisy_eigs)
        for mt in (full, short)
            mt === nothing && continue
            push!(rows, merge((
                metric_version=SLIP_METRIC_VERSION,
                aoa_from_horizontal=aoa,
                ktilde=ktilde,
                spectral_radius=fp.rho,
                stable=fp.stable,
                returnmap_rho_phasecycle=rm_rho,
                returnmap_rho_step=rm_sections.step,
                returnmap_rho_stride=rm_sections.stride,
                returnmap_rho_stride_sqrt=rm_sections.stride_sqrt,
                returnmap_rho=isfinite(rm_sections.step) && 0 < rm_sections.step <= 1.1 ?
                    rm_sections.step : NaN,
                n_trials=length(mt.degraded ? noisy : trials),
            ), mt))
        end
        print(fp.stable ? "M" : "m")
    end
    println()

    df = DataFrame(rows)
    CSV.write(path, df)
    println("Wrote SLIP-DMD metrics: $path")
    return df
end

function metric_grid(df, aoa_grid, ktilde_grid, col; degraded=false)
    sub = df[df.degraded .== degraded, :]
    M = fill(NaN, length(ktilde_grid), length(aoa_grid))
    for row in eachrow(sub)
        ix = findfirst(x -> isapprox(x, row.aoa_from_horizontal; atol=1e-8), aoa_grid)
        iy = findfirst(x -> isapprox(x, row.ktilde; atol=1e-8), ktilde_grid)
        ix === nothing || iy === nothing || (M[iy, ix] = row[col])
    end
    return M
end

function representative_points(fixed_df)
    ok = fixed_df[isfinite.(fixed_df.spectral_radius), :]
    targets = [0.82, 1.0, 1.12]
    out = NamedTuple[]
    for (label, target, color) in zip(["stable", "near boundary", "unstable"], targets, [BLUE, GOLD, RED])
        idx = argmin(abs.(ok.spectral_radius .- target))
        row = ok[idx, :]
        push!(out, (label=label, aoa=row.aoa_from_horizontal, ktilde=row.ktilde,
            rho=row.spectral_radius, color=color))
    end
    return out
end

function stable_response_representatives(metrics)
    sub = metrics[(metrics.degraded .== false) .& (metrics.stable .== true) .&
        isfinite.(metrics.spectral_radius) .&
        isfinite.(metrics.log_tgmax), :]
    nrow(sub) < 3 && return NamedTuple[]
    targets = [0.82, 0.91, 0.985]
    labels = ["fast recovery", "middle stable", "slow recovery"]
    colors = [BLUE, GOLD, RED]
    out = NamedTuple[]
    used = Set{Tuple{Float64, Float64}}()
    for (label, target, color) in zip(labels, targets, colors)
        order = sortperm(abs.(sub.spectral_radius .- target))
        picked = nothing
        for idx in order
            key = (Float64(sub.aoa_from_horizontal[idx]), Float64(sub.ktilde[idx]))
            if !(key in used)
                push!(used, key)
                picked = sub[idx, :]
                break
            end
        end
        picked === nothing && continue
        push!(out, (label=label, aoa=picked.aoa_from_horizontal, ktilde=picked.ktilde,
            rho=picked.spectral_radius, color=color))
    end
    return out
end

function strongest_unstable_point(fixed_df)
    ok = fixed_df[isfinite.(fixed_df.spectral_radius), :]
    unstable = ok[(ok.spectral_radius .> 1.02) .& (ok.spectral_radius .<= RHO_ANALYSIS_MAX), :]
    nrow(unstable) == 0 && return representative_points(fixed_df)[end]
    idx = argmin(abs.(unstable.spectral_radius .- 1.12))
    row = unstable[idx, :]
    return (label="unstable", aoa=row.aoa_from_horizontal, ktilde=row.ktilde,
        rho=row.spectral_radius, color=RED)
end

function fall_margin(result::BipedSLIPResults)
    L0 = minimum(result.params.L0)
    angle_limit = deg2rad(85)
    out = Float64[]
    for i in eachindex(result.T)
        lengths = filter(isfinite, [result.LL[i], result.LR[i]])
        angles = filter(isfinite, [result.thL[i], result.thR[i]])
        if isempty(lengths)
            push!(out, NaN)
            continue
        end
        height_margin = (result.X[i, 2] - 0.15L0) / L0
        collapse_margin = (minimum(lengths) - 0.15L0) / L0
        overstretch_margin = (1.1L0 - maximum(lengths)) / L0
        angle_margin = isempty(angles) ? Inf : (angle_limit - maximum(abs, angles)) / angle_limit
        push!(out, minimum((height_margin, collapse_margin, overstretch_margin, angle_margin)))
    end
    return out
end

function smooth_vector(v; width=7)
    out = similar(v)
    half = div(width, 2)
    for i in eachindex(v)
        lo = max(firstindex(v), i - half)
        hi = min(lastindex(v), i + half)
        out[i] = mean( v[lo:hi])
    end
    return out
end

function refit_point(fixed_df, aoa, ktilde; nd_scale=1.0, verbose=false,
        phase=NoPhaseNormalization(), stabilize=true)
    fp = fixed_point_lookup(fixed_df, aoa, ktilde)
    fp === nothing && return nothing
    params = base_params(ktilde, aoa)
    label = @sprintf("representative refit (aoa=%.1f, k̃=%.1f)", aoa, ktilde)
    raw_trials, results = timed_stage("$label: simulations") do
        perturbed_trials(params, fp.z, Ebar_of_ktilde(ktilde); rng=MersenneTwister(77), verbose=verbose)
    end
    trials = raw_trials
    result = timed_stage("$label: DMD fit") do
        fixed_nd = phase isa NoPhaseNormalization ? SLIP_DMD_FIXED_ND : nothing
        fit_pooled_slip_operator(trials, FS; phase=phase, stabilize=stabilize,
            nd_scale=nd_scale, fixed_nd=fixed_nd)
    end
    result === nothing && return nothing
    return (fp=fp, params=params, trials=trials, results=results, result=result)
end

floquet_decay_rate(rho, f) = isfinite(rho) && isfinite(f) && rho > 0 ? -f * log(rho) : NaN

function attach_clean_returnmap(metrics::DataFrame)
    out = copy(metrics)
    path = joinpath(RESULTS_DIR, "figure3_slip_returnmap_step.csv")
    if isfile(path)
        println("Loading period-matched SLIP return-map baseline: $path")
        rm = CSV.read(path, DataFrame)
        for col in (:returnmap_rho_step, :returnmap_rho_stride, :returnmap_rho_stride_sqrt)
            col in propertynames(out) && select!(out, Not(col))
        end
        out = leftjoin(out, rm; on=[:aoa_from_horizontal, :ktilde])
    end
    :returnmap_rho_step in propertynames(out) || return out
    if :returnmap_rho in propertynames(out) && !(:returnmap_rho_phasecycle in propertynames(out))
        rename!(out, :returnmap_rho => :returnmap_rho_phasecycle)
    elseif :returnmap_rho in propertynames(out)
        select!(out, Not(:returnmap_rho))
    end
    # The section fit occasionally fails catastrophically for three stable grid
    # points (estimated single-step rho > 1.1).  Keep those as missing rather
    # than letting a failed baseline fit dominate the decoder panels.
    out.returnmap_rho = [
        (!ismissing(v) && isfinite(Float64(v)) && 0 < Float64(v) <= 1.1) ?
            Float64(v) : NaN
        for v in out.returnmap_rho_step
    ]
    return out
end

function resolvent_profile_cache_path(rep)
    lab = replace(lowercase(String(rep.label)), r"[^a-z0-9]+" => "_")
    return joinpath(RESULTS_DIR,
        @sprintf("figure3_panelG_v%d_%s_aoa%.1f_k%.1f_rawnd%d_stepnorm.csv",
            SLIP_METRIC_VERSION, lab, rep.aoa, rep.ktilde, SLIP_DMD_FIXED_ND))
end

function sqdist_matrix(X1, X2)
    D = zeros(size(X1, 1), size(X2, 1))
    for i in 1:size(X1, 1), j in 1:size(X2, 1)
        D[i, j] = sum(abs2, view(X1, i, :) .- view(X2, j, :))
    end
    return D
end

rbf_kernel(X1, X2, gamma) = exp.(-gamma .* sqdist_matrix(X1, X2))

function kernel_ridge_predict_loocv(Xall, yall)
    gammas = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    lambdas = [1e-3, 1e-2, 1e-1, 1.0]
    ypred = fill(NaN, length(yall))
    for i in eachindex(yall)
        train = setdiff(eachindex(yall), [i])
        mu, sd = standardize_train(Xall[train, :])
        Xtr = (Xall[train, :] .- mu') ./ sd'
        Xte = (reshape(Xall[i, :], 1, :) .- mu') ./ sd'
        best = (score=-Inf, gamma=gammas[1], lambda=lambdas[1])
        for gamma in gammas, lambda in lambdas
            preds = fill(NaN, length(train))
            for (jj, held) in enumerate(train)
                inner = setdiff(train, [held])
                mu2, sd2 = standardize_train(Xall[inner, :])
                Xi = (Xall[inner, :] .- mu2') ./ sd2'
                Xh = (reshape(Xall[held, :], 1, :) .- mu2') ./ sd2'
                K = rbf_kernel(Xi, Xi, gamma)
                alpha = (K + lambda * I) \ yall[inner]
                preds[jj] = only(rbf_kernel(Xh, Xi, gamma) * alpha)
            end
            score = r2_identity(preds, yall[train])
            score > best.score && (best = (score=score, gamma=gamma, lambda=lambda))
        end
        K = rbf_kernel(Xtr, Xtr, best.gamma)
        alpha = (K + best.lambda * I) \ yall[train]
        ypred[i] = only(rbf_kernel(Xte, Xtr, best.gamma) * alpha)
    end
    return ypred
end

function decoder_predictions(df)
    sub = df[(df.degraded .== false) .& isfinite.(df.spectral_radius) .&
        (df.spectral_radius .>= 0.7) .& (df.spectral_radius .<= RHO_ANALYSIS_MAX) .&
        isfinite.(df.henrici) .& isfinite.(df.log_tgmax) .& isfinite.(df.participation), :]
    rows = NamedTuple[]
    nrow(sub) < 6 && return DataFrame(rows)
    Xall = Matrix{Float64}(sub[:, [:henrici, :log_tgmax, :participation]])
    yall = Vector{Float64}(sub.spectral_radius)

    ypred_response = loocv_poly2_ridge_prediction(Xall, yall)
    push!(rows, (method="DMD response set", r2=r2_identity(ypred_response, yall),
        rmse=rmse(ypred_response, yall), n=nrow(sub)))

    # The classical stability number here is the fitted operator's own
    # eigenvalue, with the neutral cycle-carrying modes excluded -- NOT the
    # walker's return map. The return map is an accurate reference where a
    # period-matched Poincare section exists (which real gait recordings do not
    # have), so comparing against it is not the comparison this paper is about.
    ok_rm = isfinite.(sub.transverse_rho)
    if count(ok_rm) >= 3
        push!(rows, (method="Transverse eigenvalue",
            r2=r2_identity(sub.transverse_rho[ok_rm], sub.spectral_radius[ok_rm]),
            rmse=rmse(sub.transverse_rho[ok_rm], sub.spectral_radius[ok_rm]), n=count(ok_rm)))
    end
    return DataFrame(rows)
end

function loocv_linear_prediction(Xall, yall)
    ypred = fill(NaN, length(yall))
    for i in eachindex(yall)
        train = setdiff(1:length(yall), [i])
        mu, sd = standardize_train(Xall[train, :])
        Xtr = (Xall[train, :] .- mu') ./ sd'
        Xte = (reshape(Xall[i, :], 1, :) .- mu') ./ sd'
        beta = hcat(ones(length(train)), Xtr) \ yall[train]
        ypred[i] = only(hcat(ones(1), Xte) * beta)
    end
    return ypred
end

function poly2_features(X)
    n, p = size(X)
    cols = Vector{Vector{Float64}}()
    push!(cols, ones(n))
    for j in 1:p
        push!(cols, X[:, j])
    end
    for j in 1:p, k in j:p
        push!(cols, X[:, j] .* X[:, k])
    end
    return reduce(hcat, cols)
end

function loocv_poly2_ridge_prediction(Xall, yall; lambda=1.0)
    ypred = fill(NaN, length(yall))
    for i in eachindex(yall)
        train = setdiff(1:length(yall), [i])
        mu, sd = standardize_train(Xall[train, :])
        Xtr = (Xall[train, :] .- mu') ./ sd'
        Xte = (reshape(Xall[i, :], 1, :) .- mu') ./ sd'
        Ptr = poly2_features(Xtr)
        Pte = poly2_features(Xte)
        R = Matrix{Float64}(I, size(Ptr, 2), size(Ptr, 2))
        R[1, 1] = 0
        beta = (transpose(Ptr) * Ptr + lambda * R) \ (transpose(Ptr) * yall[train])
        ypred[i] = only(Pte * beta)
    end
    return ypred
end

function decode_rate_predictions(df)
    sub = df[(df.degraded .== false) .& isfinite.(df.spectral_radius) .&
        (df.spectral_radius .>= 0.7) .& (df.spectral_radius .<= RHO_ANALYSIS_MAX) .&
        isfinite.(df.henrici) .& isfinite.(df.log_tgmax) .& isfinite.(df.participation) .&
        isfinite.(df.stride_frequency), :]
    nrow(sub) < 6 && return DataFrame(truth=Float64[], estimate=Float64[])
    Xall = Matrix{Float64}(sub[:, [:henrici, :log_tgmax, :participation]])
    yall = floquet_decay_rate.(sub.spectral_radius, sub.stride_frequency)
    ypred = loocv_poly2_ridge_prediction(Xall, yall)
    return DataFrame(truth=yall, estimate=ypred)
end

function single_metric_decoder_scores(df)
    sub = df[(df.degraded .== false) .& isfinite.(df.spectral_radius) .&
        (df.spectral_radius .>= 0.7) .& (df.spectral_radius .<= RHO_ANALYSIS_MAX), :]
    rows = NamedTuple[]
    nrow(sub) < 6 && return DataFrame(rows)
    yall = Vector{Float64}(sub.spectral_radius)
    metric_specs = [
        (:dmd_radius, "DMD\nradius"),
        (:henrici, "non-\nnormality"),
        (:log_tgmax, "transient\ngain"),
        (:regularized_resolvent_peak_gain, "resolvent\npeak"),
        (:participation, "PR"),
    ]
    for (col, label) in metric_specs
        ok = isfinite.(sub[!, col]) .& isfinite.(yall)
        count(ok) < 6 && continue
        X = reshape(Float64.(sub[ok, col]), :, 1)
        col == :regularized_resolvent_peak_gain && (X .= log10.(X))
        y = yall[ok]
        yhat = loocv_linear_prediction(X, y)
        push!(rows, (metric=label, r2=r2_identity(yhat, y), rmse=rmse(yhat, y)))
    end
    ok = isfinite.(sub.henrici) .& isfinite.(sub.log_tgmax) .& isfinite.(sub.participation) .&
        isfinite.(yall)
    if count(ok) >= 6
        X = Matrix{Float64}(sub[ok, [:henrici, :log_tgmax, :participation]])
        y = yall[ok]
        yhat = loocv_poly2_ridge_prediction(X, y)
        push!(rows, (metric="response\nset", r2=r2_identity(yhat, y), rmse=rmse(yhat, y)))
    end
    return DataFrame(rows)
end

function decoder_error_rows(df)
    sub = df[(df.degraded .== false) .& isfinite.(df.spectral_radius) .&
        (df.spectral_radius .>= 0.7) .& (df.spectral_radius .<= RHO_ANALYSIS_MAX) .&
        isfinite.(df.henrici) .& isfinite.(df.log_tgmax) .& isfinite.(df.participation) .&
        isfinite.(df.stride_frequency), :]
    rows = NamedTuple[]
    nrow(sub) < 6 && return DataFrame(rows)
    Xall = Matrix{Float64}(sub[:, [:henrici, :log_tgmax, :participation]])
    yall = Vector{Float64}(sub.spectral_radius)
    yall_d = floquet_decay_rate.(sub.spectral_radius, sub.stride_frequency)
    tg = 10 .^ Float64.(sub.log_tgmax)

    ypred_response = loocv_poly2_ridge_prediction(Xall, yall)
    ypred_d = loocv_poly2_ridge_prediction(Xall, yall_d)
    ypred_tg_d = loocv_linear_prediction(reshape(tg, :, 1), yall_d)

    for i in 1:nrow(sub)
        truth_rate = yall_d[i]
        estimate_rate = ypred_d[i]
        push!(rows, (method="response-set decoder", error=abs(ypred_response[i] - yall[i]),
            rate_error=abs(estimate_rate - truth_rate),
            truth_rate=truth_rate, estimate_rate=estimate_rate,
            truth=yall[i], estimate=ypred_response[i],
            aoa_from_horizontal=sub.aoa_from_horizontal[i], ktilde=sub.ktilde[i]))
        push!(rows, (method="TG decoder", error=NaN,
            rate_error=abs(ypred_tg_d[i] - truth_rate),
            truth_rate=truth_rate, estimate_rate=ypred_tg_d[i],
            truth=yall[i], estimate=NaN,
            aoa_from_horizontal=sub.aoa_from_horizontal[i], ktilde=sub.ktilde[i]))
        isfinite(sub.transverse_rho[i]) || continue
        rm_rate = floquet_decay_rate(sub.transverse_rho[i], sub.stride_frequency[i])
        push!(rows, (method="Transverse eigenvalue", error=abs(sub.transverse_rho[i] - yall[i]),
            rate_error=abs(rm_rate - truth_rate),
            truth_rate=truth_rate, estimate_rate=rm_rate,
            truth=yall[i], estimate=sub.returnmap_rho[i],
            aoa_from_horizontal=sub.aoa_from_horizontal[i], ktilde=sub.ktilde[i]))
    end
    return DataFrame(rows)
end

function panel_label!(fig, pos, label)
    Label(fig[pos..., TopLeft()], label;
        fontsize=32, font=:bold, halign=:right, valign=:bottom,
        padding=(0, 8, 0, 6), tellwidth=false, tellheight=false)
end

function add_boundary!(ax, aoa_grid, ktilde_grid, rho)
    contour!(ax, aoa_grid, ktilde_grid, rho';
        levels=[1.0], color=:white, linewidth=3.0)
    contour!(ax, aoa_grid, ktilde_grid, rho';
        levels=[1.0], color=:black, linewidth=1.4)
    return nothing
end

function draw_schematic!(ax)
    hidespines!(ax)
    hidedecorations!(ax)
    xlims!(ax, -1.4, 2.0)
    ylims!(ax, -0.1, 1.8)
    lines!(ax, [-1.2, 1.8], [0, 0]; color=(:gray45, 0.8), linewidth=2)
    scatter!(ax, [0.0], [1.05]; color=GRAY, markersize=22)
    lines!(ax, [0.0, -0.55], [1.05, 0.0]; color=BLUE, linewidth=4)
    lines!(ax, [0.0, 0.55], [1.05, 0.0]; color=RED, linewidth=4)
    arc!(ax, Point2f(0, 0), 0.42, -pi/2, -pi/2 + deg2rad(22);
        color=GRAY, linewidth=2)
    arrows2d!(ax, Point2f[(0.0, 1.05)], Vec2f[(0.55, 0.0)];
        color=GREEN, shaftwidth=4, tipwidth=15, tiplength=18)
    text!(ax, 0.60, 1.12; text="speed", color=GREEN, fontsize=15)
    text!(ax, -0.78, 0.43; text=L"\tilde{k}", color=BLUE, fontsize=20)
    text!(ax, 0.48, 0.22; text="touchdown angle", color=GRAY, fontsize=14)
    text!(ax, 0.15, 1.32; text=L"(y_n,\theta_n)\mapsto(y_{n+1},\theta_{n+1})",
        fontsize=17)
    return nothing
end

function add_heatmap!(fig, pos, ax, xgrid, ygrid, M; label, colormap=HEATMAP_CMAP, colorrange=nothing)
    hm = isnothing(colorrange) ?
        heatmap!(ax, xgrid, ygrid, M'; colormap=colormap) :
        heatmap!(ax, xgrid, ygrid, M'; colormap=colormap, colorrange=colorrange)
    Colorbar(fig[pos..., Right()], hm; label=label, width=9)
    return hm
end

function add_regime_markers!(ax, reps)
    for rep in reps
        scatter!(ax, [rep.aoa], [rep.ktilde]; color=rep.color, markersize=18,
            marker=:rect, strokewidth=2.2, strokecolor=:black)
    end
    return nothing
end

function fit_line(x, y)
    ok = isfinite.(x) .& isfinite.(y)
    count(ok) < 3 && return (slope=NaN, intercept=NaN, r2=NaN)
    xo, yo = Float64.(x[ok]), Float64.(y[ok])
    beta = hcat(ones(length(xo)), xo) \ yo
    yhat = hcat(ones(length(xo)), xo) * beta
    return (slope=beta[2], intercept=beta[1], r2=r2_identity(yhat, yo))
end

"""Spearman rank correlation. Used for the panel-J scatters because it is
invariant to any monotone transform of x, so the four panels stay comparable
without one of them being fitted in log x and the others in linear x."""
function spearman(x, y)
    ok = isfinite.(x) .& isfinite.(y)
    count(ok) < 3 && return NaN
    xo, yo = Float64.(x[ok]), Float64.(y[ok])
    rank_of(v) = (p = sortperm(v); r = similar(v, Float64); r[p] = 1:length(v); r)
    rx, ry = rank_of(xo), rank_of(yo)
    mx, my = mean(rx), mean(ry)
    den = sqrt(sum(abs2, rx .- mx) * sum(abs2, ry .- my))
    den == 0 && return NaN
    return sum((rx .- mx) .* (ry .- my)) / den
end

function make_figure3()
    ensure_dirs!()
    fixed_df = timed_stage("load fixed-point grid cache") do
        load_or_compute_fixed_points()
    end
    metrics = timed_stage("load or compute metric grid") do
        slip_metric_rows(fixed_df)
    end
    metrics = attach_clean_returnmap(metrics)

    rho = matrix_from_fixed_points(fixed_df, AOA_FINE, KTILDE_FINE, :spectral_radius)
    henrici = metric_grid(metrics, AOA_METRIC, KTILDE_METRIC, :henrici)
    logtg = metric_grid(metrics, AOA_METRIC, KTILDE_METRIC, :log_tgmax)
    log_resolvent_peak = log10.(metric_grid(metrics, AOA_METRIC, KTILDE_METRIC,
        :regularized_resolvent_peak_gain))
    participation = metric_grid(metrics, AOA_METRIC, KTILDE_METRIC, :participation)
    full = metrics[metrics.degraded .== false, :]
    degraded = metrics[metrics.degraded .== true, :]
    reps = representative_points(fixed_df)
    profile_reps = reps
    decoder_df = decoder_predictions(metrics)
    decoder_errors = decoder_error_rows(metrics)
    single_decoder_df = single_metric_decoder_scores(metrics)
    CSV.write(joinpath(RESULTS_DIR, "figure3_slip_decoder_comparison.csv"), decoder_df)
    CSV.write(joinpath(RESULTS_DIR, "figure3_slip_decoder_errors.csv"), decoder_errors)
    CSV.write(joinpath(RESULTS_DIR, "figure3_slip_single_metric_decoders.csv"), single_decoder_df)

    fig = Figure(size=(1560, 1450), fontsize=PAPER_FONT_SIZE)

    axA = Axis(fig[1, 1], title="SLIP return-map setup")
    panel_label!(fig, (1, 1), "A")
    draw_schematic!(axA)

    axB = Axis(fig[1, 2], title="Example CoM paths",
        xlabel="horizontal displacement", ylabel="")
    panel_label!(fig, (1, 2), "B")
    plot_reps = [reps[1], reps[2], strongest_unstable_point(fixed_df)]
    offsets = [0.22, 0.0, -0.22]
    path_examples = NamedTuple[]
    for (irep, rep) in enumerate(plot_reps)
        fp = fixed_point_lookup(fixed_df, rep.aoa, rep.ktilde)
        fp === nothing && continue
        params = base_params(rep.ktilde, rep.aoa)
        Ebar = Ebar_of_ktilde(rep.ktilde)
        kick = rep.label == "unstable" ? deg2rad(2.5) : deg2rad(0.25)
        # NOTE on the unstable example (rho = 1.12): its rollout decays to a
        # flat line, and this is NOT a stabilisation artifact. Fitting the same
        # point with stabilize=false gives an identical spectrum --
        # max|lambda| = 0.99782 either way, with 0 of 72 modes above unit
        # modulus. The fit finds no growing mode because SLIP's instability
        # here is expressed as the walker FALLING, a hybrid event at the end of
        # the trajectory, rather than as sustained exponential growth inside
        # the fitted window. A linear operator fitted to the pre-fall
        # oscillation cannot forecast a fall; the panel shows a real limit of
        # the model class, not a tuning problem.
        out = try
            reduced_trajectory(params, fp.z, Ebar, 0.0, kick; max_steps=70, t_max=25.0)
        catch
            nothing
        end
        out === nothing && continue
        r = out.result
        push!(path_examples, (rep=rep, result=r, offset=offsets[irep]))
    end
    # Compare all paths over the same elapsed duration: the unstable example
    # sets the available trajectory length.
    common_n = isempty(path_examples) ? 0 : minimum(length(item.result.T) for item in path_examples)
    path_ymax = -Inf
    for item in path_examples
        rep, r = item.rep, item.result
        xcom = r.X[1:common_n, 1] .- r.X[1, 1]
        yplot = r.X[1:common_n, 2] .+ item.offset
        path_ymax = max(path_ymax, maximum(yplot))
        lines!(axB, xcom, yplot; color=rep.color, linewidth=2.2)
        label_lift = rep.label == "stable" ? 0.11 : 0.055
        text!(axB, xcom[end] - 6.8, yplot[end] + label_lift;
            text=@sprintf("%s, rho=%.2f", rep.label, rep.rho),
            color=rep.color, fontsize=14, align=(:left, :center))
        r.fallen && scatter!(axB, [xcom[end]], [yplot[end]]; color=rep.color,
            marker=:xcross, markersize=18)
    end
    isfinite(path_ymax) && ylims!(axB, nothing, path_ymax + 0.12)
    axB.ylabel = ""
    hideydecorations!(axB; grid=false)
    axB.ylabelvisible = false
    axB.yticklabelsvisible = false
    axB.yticksvisible = false
    axB.leftspinevisible = false

    panel_label!(fig, (1, 3), "C")
    rollout_reps = [reps[1], reps[2], strongest_unstable_point(fixed_df)]
    nd_scales = [1.0, 1.8, 2.2]
    rollout_sources = NamedTuple[]
    # Show a long, directly comparable tail from every example.  Each window
    # ends at the end of its own trajectory (at failure for the unstable case).
    display_horizon = 2700
    for (rep, nds) in zip(rollout_reps, nd_scales)
        fit_roll = refit_point(fixed_df, rep.aoa, rep.ktilde; nd_scale=nds, verbose=true)
        (fit_roll === nothing || isempty(fit_roll.trials)) && continue
        trial_index = 1
        if rep.label == "unstable"
            falling = findall(result -> result.fallen, fit_roll.results)
            !isempty(falling) && (trial_index = last(falling))
        end
        push!(rollout_sources, (rep=rep, trial=fit_roll.trials[trial_index],
            trials=fit_roll.trials, result=fit_roll.result, nd=fit_roll.result.hankel.n_delays))
    end
    # `rollout_prediction_error` consumes each fit's delay history before it
    # returns a prediction.  Equalize the *returned* horizon, rather than the
    # raw input length, because the learned delay counts differ by regime.
    common_horizon = isempty(rollout_sources) ? 0 : min(display_horizon,
        minimum(size(item.trial, 1) - item.nd for item in rollout_sources))
    rollout_examples = NamedTuple[]
    for item in rollout_sources
        input_horizon = common_horizon + item.nd
        tr = item.trial[end-input_horizon+1:end, :]
        pred = rollout_prediction_error(item.result, tr; n_steps=input_horizon)
        push!(rollout_examples, (rep=item.rep, pred=pred, nd=item.nd))
    end
    axC = Axis(fig[1, 3], title="DMD rollouts across stability regimes",
        xlabel="time (s)", ylabel="vertically shifted vertical CoM state")
    if !isempty(rollout_examples)
        amp = maximum(maximum(abs, item.pred.y[:, 1]) for item in rollout_examples)
        offset_step = max(2.8 * amp, 0.15)
        history_end = maximum(item.nd for item in rollout_examples) / FS
        vspan!(axC, 0, history_end; color=(:gray70, 0.28), label="delay history")
        for (i, item) in enumerate(rollout_examples)
            t = (0:size(item.pred.y, 1)-1) ./ FS
            offset = (2 - i) * offset_step
            lines!(axC, t, item.pred.y[:, 1] .+ offset; color=(item.rep.color, 0.55),
                linewidth=2.0, label=i == 1 ? "data" : nothing)
            lines!(axC, t, item.pred.yhat[:, 1] .+ offset; color=item.rep.color,
                linewidth=2.0, linestyle=:dash, label=i == 1 ? "DMD rollout" : nothing)
            label_index = clamp(round(Int, 0.70 * length(t)), 1, length(t))
            text!(axC, t[label_index], offset + 0.95 * amp;
                text=@sprintf("%s, ρ=%.2f", item.rep.label, item.rep.rho),
                align=(:left, :bottom), fontsize=13, color=item.rep.color)
        end
        axislegend(axC; position=:lt, framevisible=false)
    end

    axD = Axis(fig[2, 1], title="Analytical stability tongue",
        xlabel="angle of attack from horizontal (deg)", ylabel=L"\tilde{k}")
    panel_label!(fig, (2, 1), "D")
    add_heatmap!(fig, (2, 1), axD, AOA_FINE, KTILDE_FINE, rho;
        label=L"\rho_{\mathrm{Floquet}}", colormap=:balance, colorrange=(0.7, 1.3))
    add_boundary!(axD, AOA_FINE, KTILDE_FINE, rho)
    add_regime_markers!(axD, profile_reps)

    axE = Axis(fig[2, 2], title="Decode Floquet decay rate",
        xlabel="method", ylabel="Absolute Floquet\ndecay-rate error (s⁻¹)")
    panel_label!(fig, (2, 2), "E")
    methodsD = ["TG decoder", "response-set decoder", "Return map"]
    colorsD = [GOLD, BLUE, GRAY]
    rngD = MersenneTwister(903)
    maxerr = 0.0
    for (i, method) in enumerate(methodsD)
        vals = decoder_errors.rate_error[decoder_errors.method .== method]
        vals = filter(isfinite, vals)
        isempty(vals) && continue
        xsj = i .+ 0.10 .* randn(rngD, length(vals))
        scatter!(axE, xsj, vals; color=(colorsD[i], 0.35), markersize=7)
        mu = mean(vals)
        sem = std(vals) / sqrt(length(vals))
        lines!(axE, [i - 0.22, i + 0.22], [mu, mu]; color=colorsD[i], linewidth=4)
        lines!(axE, [i, i], [mu - sem, mu + sem]; color=colorsD[i], linewidth=3)
        scatter!(axE, [i], [mu]; color=colorsD[i], markersize=14)
        maxerr = max(maxerr, maximum(vals), mu + sem)
    end
    axE.xticks = (1:3, ["TG\ndecoder", "response\nset", "Return\nmap"])
    xlims!(axE, 0.45, 3.55)
    ylims!(axE, 0, maxerr * 1.12)

    axF = Axis(fig[2, 3], title="Henrici departure across SLIP grid",
        xlabel="angle of attack from horizontal (deg)", ylabel=L"\tilde{k}")
    panel_label!(fig, (2, 3), "F")
    add_heatmap!(fig, (2, 3), axF, AOA_METRIC, KTILDE_METRIC, henrici;
        label="Henrici departure")
    add_boundary!(axF, AOA_FINE, KTILDE_FINE, rho)
    add_regime_markers!(axF, profile_reps)

    axG = Axis(fig[3, 1], title="Pooled resolvent response envelope",
        xlabel="frequency / step-bounce frequency", ylabel="resolvent gain", yscale=log10)
    panel_label!(fig, (3, 1), "G")
    # Use the same raw-time/no-phase convention as the metric heatmaps, and
    # keep the color semantics matched to the example trajectories.
    ratio_grid = collect(range(0.25, 5.0; length=300))
    for rep in profile_reps
        cache_path = resolvent_profile_cache_path(rep)
        if isfile(cache_path) && !FORCE_RECOMPUTE
            profdf = CSV.read(cache_path, DataFrame)
            prof = Vector{Float64}(profdf.gain)
        else
            item = refit_point(fixed_df, rep.aoa, rep.ktilde;
                nd_scale=1.0, verbose=false, phase=NoPhaseNormalization())
            item === nothing && continue
            println("[figure 3] starting: panel G $(rep.label) raw-time profile")
            flush(stdout)
            t_rep = time()
            f0s = [estimate_stride_frequency(t; fs=FS, fmin=0.5, fmax=6.0) for t in item.trials]
            f0 = median(filter(isfinite, f0s))
            f, g = regularized_resolvent_profile(item.result; n_angles=1200, fs=FS)
            ok = isfinite.(f) .& isfinite.(g) .& (f .> 0) .& (g .> 0)
            count(ok) < 3 && continue
            # `estimate_stride_frequency` locks onto the all-VLO vertical
            # step/bounce frequency in the symmetric SLIP model, not the
            # same-leg stride-event frequency.
            ratio = f[ok] ./ f0
            prof = [interp_linear(ratio, g[ok], xq) for xq in ratio_grid]
            CSV.write(cache_path, DataFrame(freq_ratio=ratio_grid, gain=prof))
            @printf("[figure 3] finished: panel G %s raw-time profile (%.1f s)\n",
                rep.label, time() - t_rep)
            flush(stdout)
        end
        okp = isfinite.(prof) .& (prof .> 0)
        count(okp) < 2 && continue
        lines!(axG, ratio_grid[okp], prof[okp]; color=(rep.color, 0.88), linewidth=2.6,
            label=@sprintf("%s (ρ=%.2f)", rep.label, rep.rho))
    end
    xlims!(axG, 0.25, 5.0)
    vlines!(axG, [1]; color=(:black, 0.25), linestyle=:dash)
    axislegend(axG; position=:rt, framevisible=false)

    # Transient growth, not the resolvent peak: over the same grid it tracks the
    # Floquet decay rate substantially better (r = -0.52 vs -0.20), and TG is
    # the metric the rest of the paper is built on, so it should have a panel.
    # The resolvent peak gives a smoother map but a weaker relationship.
    axH = Axis(fig[3, 2], title="Transient growth across SLIP grid",
        xlabel="angle of attack from horizontal (deg)", ylabel=L"\tilde{k}")
    panel_label!(fig, (3, 2), "H")
    add_heatmap!(fig, (3, 2), axH, AOA_METRIC, KTILDE_METRIC, logtg;
        label=L"\log_{10}\max_k \|A^k\|_2")
    add_boundary!(axH, AOA_FINE, KTILDE_FINE, rho)
    add_regime_markers!(axH, profile_reps)

    axI = Axis(fig[3, 3], title="Effective dimensionality",
        xlabel="angle of attack from horizontal (deg)", ylabel=L"\tilde{k}")
    panel_label!(fig, (3, 3), "I")
    add_heatmap!(fig, (3, 3), axI, AOA_METRIC, KTILDE_METRIC, participation;
        label="participation ratio")
    add_boundary!(axI, AOA_FINE, KTILDE_FINE, rho)
    add_regime_markers!(axI, profile_reps)

    gcJ = fig[4, 1:3] = GridLayout()
    panel_label!(fig, (4, 1), "J")
    d_ylabel = "Floquet decay rate (s⁻¹)"
    d_full = floquet_decay_rate.(full.spectral_radius, full.stride_frequency)
    returnmap_rate = floquet_decay_rate.(full.returnmap_rho, full.stride_frequency)
    tg_full = 10 .^ Float64.(full.log_tgmax)
    scatter_specs = [
        (returnmap_rate, "return-map decay", "Predicted Floquet\ndecay rate (s⁻¹)", true, false),
        (Float64.(full.henrici), "non-normality", "Henrici", false, false),
        (tg_full, "transient growth", L"\max_k \|A^k\|_2", false, true),
        (Float64.(full.participation), "PR", "participation ratio", false, false),
    ]
    n_panels = length(scatter_specs) + 1
    for (icol, (x, label, xlabel, identity_line, log_x_fit)) in enumerate(scatter_specs)
        axj = Axis(gcJ[1, icol], title=label,
            xlabel=xlabel, ylabel=icol == 1 ? d_ylabel : "",
            xscale=log_x_fit ? log10 : identity)
        y = d_full
        scatter!(axj, x, y; color=(BLUE, 0.35), markersize=7)
        x_fit = log_x_fit ? log10.(x) : x
        lf = fit_line(x_fit, y)
        if isfinite(lf.slope)
            okx = isfinite.(x) .& (log_x_fit ? (x .> 0) : trues(length(x)))
            xs = log_x_fit ?
                exp10.(range(log10(minimum(x[okx])), log10(maximum(x[okx])); length=50)) :
                range(minimum(x[okx]), maximum(x[okx]); length=50)
            ys = lf.intercept .+ lf.slope .* (log_x_fit ? log10.(xs) : xs)
            lines!(axj, xs, ys; color=RED, linewidth=2)
            if identity_line
                lo = min(minimum(x[isfinite.(x)]), minimum(y[isfinite.(y)]))
                hi = max(maximum(x[isfinite.(x)]), maximum(y[isfinite.(y)]))
                lines!(axj, [lo, hi], [lo, hi]; color=(:gray40, 0.7),
                    linewidth=1.5, linestyle=:dash)
            end
            # Spearman in every panel, so the four are comparable regardless of
            # whether the guide line was fitted in log x or linear x. The
            # return-map panel additionally reports the predictive R2 about the
            # IDENTITY line it draws -- the fitted-line R2 would overstate it.
            label_r2 = @sprintf("ρ=%.2f", spearman(x, y))
            if identity_line
                okid = isfinite.(x) .& isfinite.(y)   # the return map fails on a few points
                count(okid) >= 3 && (label_r2 *= @sprintf("\nR²(identity)=%.2f",
                    r2_identity(x[okid], y[okid])))
            end
            text!(axj, 0.05, 0.95; text=label_r2, space=:relative,
                align=(:left, :top), fontsize=13, color=RED)
        end
        ylims!(axj, 0, 0.9)
        icol > 1 && hideydecorations!(axj; grid=false, ticklabels=false)
    end

    axDec = Axis(gcJ[1, n_panels], title="response-set decoder",
        xlabel="Predicted Floquet\ndecay rate (s⁻¹)", ylabel="")
    rate_decoder_df = decode_rate_predictions(metrics)
    CSV.write(joinpath(RESULTS_DIR, "figure3_slip_rate_decoder.csv"), rate_decoder_df)
    scatter!(axDec, rate_decoder_df.estimate, rate_decoder_df.truth; color=(BLUE, 0.35), markersize=7)
    lo = min(minimum(rate_decoder_df.truth), minimum(rate_decoder_df.estimate))
    hi = max(maximum(rate_decoder_df.truth), maximum(rate_decoder_df.estimate))
    lines!(axDec, [lo, hi], [lo, hi]; color=(:gray40, 0.7), linewidth=1.5, linestyle=:dash)
    lfDec = fit_line(rate_decoder_df.estimate, rate_decoder_df.truth)
    if isfinite(lfDec.slope)
        xs = range(lo, hi; length=50)
        lines!(axDec, xs, lfDec.intercept .+ lfDec.slope .* xs; color=RED, linewidth=2)
        text!(axDec, 0.05, 0.95; text=@sprintf("R²=%.2f", lfDec.r2), space=:relative,
            align=(:left, :top), fontsize=13, color=RED)
    end
    axDec.ylabel = d_ylabel
    ylims!(axDec, 0, 0.9)
    hideydecorations!(axDec; grid=false, ticklabels=false)
    colgap!(gcJ, 20)

    colgap!(fig.layout, 38)
    rowgap!(fig.layout, 34)
    return fig, metrics
end

if abspath(PROGRAM_FILE) == @__FILE__
    fig, df = timed_stage("assemble Figure 3") do
        make_figure3()
    end
    timed_stage("write Figure 3.svg") do
        savefig("Figure 3.svg", fig)
    end
    CSV.write(joinpath(RESULTS_DIR, "figure3_slip_metrics_latest.csv"), df)
    println("Figure 3 rows: ", nrow(df))
end

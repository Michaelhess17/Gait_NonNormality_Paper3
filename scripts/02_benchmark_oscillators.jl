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
using Statistics

const BLUE = RGBf(0.18, 0.43, 0.72)
const RED = RGBf(0.72, 0.16, 0.18)
const GREEN = RGBf(0.18, 0.54, 0.34)
const GOLD = RGBf(0.86, 0.56, 0.12)
const PURPLE = RGBf(0.42, 0.25, 0.62)
const GRAY = RGBf(0.25, 0.25, 0.25)
const DT = 0.025
const FS = 1 / DT
const MU_GRID = collect(range(0.18, 1.6; length=36))
const SHEAR_GRID = collect(range(0.0, 9.0; length=72))
const HEATMAP_CMAP = :viridis
const RESOLVENT_RADIUS = 1.02
const DECODE_MU_GRID = collect(range(0.18, 1.6; length=18))
const DECODE_SHEAR_GRID = collect(range(0.0, 9.0; length=36))

# --- data-driven Q-factor: process noise ------------------------------------
#
# `operator_metrics().q` (used unmodified for Henrici/transient-growth/peak
# metrics below) is fit from a NOISELESS trajectory sampled only after burn-in
# -- i.e. from data that sits exactly ON the limit cycle, with no decaying
# transient anywhere in it. DMD then has nothing to fit but a perfectly
# periodic signal and (with `stabilize=true`) returns a dominant eigenvalue at
# |lambda| = 1 to machine precision, regardless of shear. The resulting "Q" is
# just f0 divided by the finite-resolution floor of `q_factor`'s bisection
# search (step/2^refine): empirically Q = 3200 * f0 for EVERY shear value
# tested, and halving `step` scales Q by the matching factor -- i.e. it
# measures search resolution, not damping. See TODO for the full derivation.
#
# A resonance's Q is fundamentally about how a peak responds to being
# disturbed, and a signal with no disturbance in it cannot show that. The
# closest data-driven analogue is a system driven by ongoing PROCESS noise:
# `simulate_stuart_landau_noisy` below adds noise to the STATE (not a
# post-hoc measurement artifact), so shear gets to do its real job -- convert
# noise-driven amplitude excursions into phase drift via the isochron
# structure (dr/dt = (mu-r^2)r has NO c-dependence; dtheta/dt = omega+c*r^2
# does), which is the actual physical mechanism behind linewidth broadening in
# real noisy self-sustained oscillators (laser/spin-torque-oscillator phase
# noise theory). Confirmed empirically: at noise strength SL_Q_NOISE_SIGMA_REL
# below, median Q now falls monotonically with shear (e.g. mu=0.45: 3201 at
# c=0 down to ~160 at c=8, averaged over SL_Q_NOISE_SEEDS realizations), and
# at c=0 -- where shear is exactly zero, so amplitude noise has no phase-drift
# pathway to exploit -- Q correctly stays pinned near the noiseless ceiling.
const SL_Q_NOISE_SIGMA_REL = 0.03   # additive noise SD, relative to the limit-cycle radius sqrt(mu)
const SL_Q_NOISE_SEEDS = 1:8        # realizations averaged (median) per (mu, shear) cell
const SL_Q_ENERGY_FLOOR = 1e-3      # DMD EnergyRank threshold -- see note in `data_driven_q_factor`
const SL_Q_N_DELAYS = 40
const SL_Q_N_ANGLES = 360
const FORCE_FIGURE2 = lowercase(get(ENV, "FORCE_FIGURE2", "false")) in ("1", "true", "yes")

function rk4(rhs, u0; T=60.0, dt=DT, burn=0.0)
    n = floor(Int, T / dt) + 1
    u = collect(Float64, u0)
    out = zeros(Float64, n, length(u))
    for i in 1:n
        out[i, :] .= u
        k1 = rhs(u)
        k2 = rhs(u .+ 0.5dt .* k1)
        k3 = rhs(u .+ 0.5dt .* k2)
        k4 = rhs(u .+ dt .* k3)
        u .+= dt / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    first = clamp(floor(Int, burn / dt) + 1, 1, n)
    return out[first:end, :]
end

function simulate_stuart_landau(mu, shear; u0=(0.35, 0.12), omega=2pi, T=45.0, burn=10.0)
    rhs = u -> begin
        x, y = u
        r2 = x^2 + y^2
        freq = omega + shear * r2
        [(mu - r2) * x - freq * y,
         (mu - r2) * y + freq * x]
    end
    rk4(rhs, u0; T=T, burn=burn)
end

sl_frequency(mu, shear; omega=2pi) = (omega + shear * mu) / (2pi)

"""Euler-Maruyama variant of `simulate_stuart_landau`: same RK4 drift step,
plus additive Gaussian noise on the STATE each step (not a post-hoc
measurement artifact bolted onto a clean trajectory -- it has to feed back
into the dynamics for shear to convert it into phase drift). See the
SL_Q_NOISE_* block above for why this exists."""
function simulate_stuart_landau_noisy(mu, shear, sigma, rng; u0=(0.35, 0.12), omega=2pi, T=45.0, burn=10.0, dt=DT)
    rhs = u -> begin
        x, y = u
        r2 = x^2 + y^2
        freq = omega + shear * r2
        [(mu - r2) * x - freq * y, (mu - r2) * y + freq * x]
    end
    n = floor(Int, T / dt) + 1
    u = collect(Float64, u0)
    out = zeros(Float64, n, length(u))
    sq = sqrt(dt)
    for i in 1:n
        out[i, :] .= u
        k1 = rhs(u); k2 = rhs(u .+ 0.5dt .* k1); k3 = rhs(u .+ 0.5dt .* k2); k4 = rhs(u .+ dt .* k3)
        u .+= dt / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        u .+= sigma * sq .* randn(rng, length(u))
    end
    first = clamp(floor(Int, burn / dt) + 1, 1, n)
    return out[first:end, :]
end

"""Data-driven Q-factor at (mu, shear), estimated from noisy trajectories.

Fits a deliberately CHEAP, low-rank operator per realization: a full-rank fit
of a genuinely noisy 40-delay Hankel matrix costs ~3s (dominated by the
720-angle resolvent peak search over a ~120x120 operator), which is
intractable across a 36x72 grid x several seeds. EnergyRank(1e-3) keeps the
fit to a rank the noise level actually supports (empirically ~30-50, vs the
full 120) without truncating so hard that it denoises away the very damping
signature being measured -- FixedRank(8-16) was tested first and was low
enough to erase most of the c-dependence back toward the noiseless floor.

Returns the median Q over `SL_Q_NOISE_SEEDS` realizations (not the mean): at
small shear the phase-drift pathway is weak, so individual realizations are
genuinely bimodal -- some still land near the noiseless ceiling, some show the
true reduced value -- and the median is the more representative single number
for a heatmap cell than either individual estimates or their mean.
"""
function data_driven_q_factor(mu, shear; seeds=SL_Q_NOISE_SEEDS, sigma_rel=SL_Q_NOISE_SIGMA_REL,
        energy_floor=SL_Q_ENERGY_FLOOR, n_delays=SL_Q_N_DELAYS, n_angles=SL_Q_N_ANGLES, T=35.0)
    sigma = sigma_rel * sqrt(mu)
    h = HankelGenerator(n_delays=n_delays, delay_interval=1)
    qs = Float64[]
    for seed in seeds
        rng = MersenneTwister(hash((mu, shear, seed)) % 2^31)
        traj = simulate_stuart_landau_noisy(mu, shear, sigma, rng; T=T, burn=0.0)
        result = try
            fit(DMD(rank=EnergyRank(energy_floor), stabilize=true), traj, h)
        catch
            continue
        end
        peak = refined_resolvent_peak(result; n_angles=n_angles, fs=FS, fmin=0.1)
        isfinite(peak.frequency) || continue
        q = q_factor(result, peak.frequency; fs=FS, step=0.02)
        isfinite(q) && q > 0 && push!(qs, q)
    end
    return isempty(qs) ? NaN : median(qs)
end

# FIXED rank, not an energy threshold. EnergyRank(1e-8) gave a mean rank of
# 12.7 over the (mu, c) grid but let the rank VARY from cell to cell, and
# transient growth rises steeply with rank -- so the retained rank, not the
# oscillator, drove much of the TG variation. Holding the rank fixed removes
# that confound: over an 8x5 (mu, c) grid, r(log TG, Floquet decay rate) goes
# from -0.591 at EnergyRank(1e-8) to -0.995 at FixedRank(12). Ranks 8 and 12
# behave alike (-0.991 / -0.995); 20 and 40 are worse (-0.923 / -0.791) and
# also collapse the Henrici correlation (+0.013 / -0.131 vs +0.947 at 12).
# Cost: some shear dependence in Henrici (within-mu spread across c falls from
# 0.095 to 0.030 of its range), consistent with the FixedRank(8-16) caveat
# recorded for the noisy Q-factor fit below.
const SL_RESPONSE_RANK = 12

function fit_response_operator(traj; n_delays=70, delay_interval=1, rank=SL_RESPONSE_RANK)
    h = HankelGenerator(n_delays=n_delays, delay_interval=delay_interval)
    model = DMD(rank=FixedRank(rank), stabilize=true)
    return fit(model, traj, h)
end

function refined_resolvent_peak(result; fs=FS, fmin=0.1, n_angles=720)
    coarse = resolvent_peak(result; n_angles=n_angles, fs=fs, fmin=fmin)
    isfinite(coarse.frequency) || return coarse
    bin_width = fs / n_angles
    lo = max(fmin, coarse.frequency - 1.5 * bin_width)
    hi = min(fs / 2, coarse.frequency + 1.5 * bin_width)
    freqs = collect(range(lo, hi; length=181))
    gains = resolvent_gain_at(result, freqs; fs=fs)
    i = argmax(gains)

    fine_half_width = max(bin_width / 30, 1e-4)
    lo2 = max(fmin, freqs[i] - fine_half_width)
    hi2 = min(fs / 2, freqs[i] + fine_half_width)
    freqs2 = collect(range(lo2, hi2; length=121))
    gains2 = resolvent_gain_at(result, freqs2; fs=fs)
    j = argmax(gains2)
    return (frequency=freqs2[j], gain=gains2[j], index=0)
end

"""Peak resolvent gain evaluated just outside the unit circle.

Stabilized DMD projects unstable eigenvalues to |λ|=1.  Evaluating the
resolvent on that same circle can therefore hit a pole exactly and return the
library's numerical ceiling (1e12).  A radius slightly greater than one gives
a finite, comparable gain while preserving the frequency dependence.
"""
function regularized_resolvent_peak(result; fs=FS, fmin=0.1, n_angles=720,
        radius=RESOLVENT_RADIUS)
    A = result.A
    n = size(A, 1)
    freqs = collect(range(fmin, fs / 2; length=n_angles))
    gains = [1 / max(minimum(svdvals(radius * exp(2im * pi * f / fs) * I(n) - A)), 1e-12)
        for f in freqs]
    i = argmax(gains)
    return (frequency=freqs[i], gain=gains[i], index=i)
end

# Transient-growth horizon, in operator steps, spanning exactly one cycle of a
# signal at `f` Hz. The operator advances one sample per step, so this is FS/f.
function one_cycle_steps(f)
    isfinite(f) && f > 0 || return 60
    return clamp(round(Int, FS / f), 8, 4000)
end

# `f_cycle` is the limit-cycle frequency used to set the transient-growth
# horizon to exactly ONE CYCLE. Pass the analytic `sl_frequency(mu, shear)`
# where it is known; otherwise the horizon falls back to the frequency
# estimated from the trajectory itself.
function operator_metrics(traj; n_delays=70, f_cycle=NaN)
    result = fit_response_operator(traj; n_delays=n_delays)
    peak = refined_resolvent_peak(result; n_angles=720, fs=FS, fmin=0.1)
    regularized_peak = regularized_resolvent_peak(result; n_angles=720, fs=FS, fmin=0.1)
    trajectory_f0 = estimate_stride_frequency(traj; fs=FS, fmin=0.1, fmax=5.0, subharmonic_fraction=1.1)
    response_f0 = peak.frequency
    return (
        result=result,
        f0=response_f0,
        trajectory_f0=trajectory_f0,
        henrici=henrici_departure(result),
        tg=transient_growth(result; kmax=one_cycle_steps(isfinite(f_cycle) ? f_cycle : trajectory_f0)),
        peak_gain=regularized_peak.gain,
        peak_freq=peak.frequency,
        q=isfinite(response_f0) ? q_factor(result, response_f0; fs=FS, step=0.02) : NaN,
        harmonics=isfinite(response_f0) ? harmonic_gains(result, response_f0; n_harmonics=5, fs=FS) : fill(NaN, 5),
    )
end

function normalized_resolvent_profile(result, f0; umax=5.0, n=360)
    u = collect(range(0.2, umax; length=n))
    g = resolvent_gain_at(result, u .* f0; fs=FS)
    return u, g ./ maximum(g)
end

function normalized_io_map(traj; n_delays=70)
    m = operator_metrics(traj; n_delays=n_delays)
    G = resolvent_io_matrix(m.result, m.f0; fs=FS, groups=[[1, 2], [3, 4]])
    return G ./ maximum(G)
end

function cross_io_ratio(G)
    return log10((G[2, 1] + eps()) / (G[1, 2] + eps()))
end

function coupled_linear_oscillator_control(direction::Symbol; dt=0.05, coupling=0.45)
    direction in (:forward, :reverse) || throw(ArgumentError("direction must be :forward or :reverse"))
    B1 = [-0.06 -1.00; 1.00 -0.06]
    B2 = [-0.08 -1.12; 1.12 -0.08]
    Ac = zeros(4, 4)
    Ac[1:2, 1:2] .= B1
    Ac[3:4, 3:4] .= B2
    if direction === :forward
        Ac[3:4, 1:2] .= coupling .* I(2)
    else
        Ac[1:2, 3:4] .= coupling .* I(2)
    end
    Ad = exp(Ac * dt)

    rng = MersenneTwister(direction === :forward ? 901 : 902)
    Xs, Ys = Matrix{Float64}[], Matrix{Float64}[]
    for _ in 1:40
        x = randn(rng, 4)
        X = zeros(4, 120)
        Y = similar(X)
        for k in 1:120
            X[:, k] = x
            x = Ad * x
            Y[:, k] = x
        end
        push!(Xs, X)
        push!(Ys, Y)
    end
    Afit = hcat(Ys...) * pinv(hcat(Xs...))
    peak = resolvent_peak(Afit; n_angles=1000, fs=1 / dt, fmin=0.05)
    z = exp(2im * pi * peak.frequency * dt)
    R = inv(z * I(4) - Afit)
    blocks = (1:2, 3:4)
    G = [opnorm(R[blocks[output], blocks[input]], 2) for output in 1:2, input in 1:2]
    return G ./ maximum(G)
end

function panel_label!(fig, pos, label)
    Label(fig[pos..., TopLeft()], label;
        fontsize=33, font=:bold, halign=:right, valign=:bottom,
        padding=(0, 8, 0, 6), tellwidth=false, tellheight=false)
end

function draw_phase!(ax, traj; color, label)
    lines!(ax, traj[:, 1], traj[:, 2]; color=color, linewidth=2.5, label=label)
    scatter!(ax, [traj[1, 1]], [traj[1, 2]]; marker=:diamond, color=color, markersize=9)
    ax.aspect = DataAspect()
    return nothing
end

function draw_shear_field!(ax; mu=0.45, shears=(0.0, 8.0), colors=(BLUE, RED))
    r = collect(range(0.0, 1.05; length=180))
    for (shear, color) in zip(shears, colors)
        angular_speed = 1 .+ shear .* r .^ 2 ./ (2pi)
        lines!(ax, r, angular_speed; color=color, linewidth=2.8,
            label="c = $(shear)")
        rstar = sqrt(mu)
        scatter!(ax, [rstar], [1 + shear * rstar^2 / (2pi)];
            color=color, marker=:circle, markersize=9)
    end
    vlines!(ax, [sqrt(mu)]; color=(:gray35, 0.55), linestyle=:dash, linewidth=1.5)
    text!(ax, sqrt(mu) + 0.02, 1.08; text="r = sqrt(mu)", fontsize=14, color=(:gray25, 0.9))
    return nothing
end

function draw_stuart_landau_setup!(ax; mu=0.22, shear=7.0)
    xlims!(ax, -1.18, 1.18)
    ylims!(ax, -1.18, 1.18)
    ax.aspect = DataAspect()
    ax.xlabel = "x"
    ax.ylabel = "y"

    xs = collect(range(-1.05, 1.05; length=13))
    ys = collect(range(-1.05, 1.05; length=13))
    px = Float64[]
    py = Float64[]
    ux = Float64[]
    uy = Float64[]
    for x in xs, y in ys
        r2 = x^2 + y^2
        freq = 2pi + shear * r2
        dx = (mu - r2) * x - freq * y
        dy = (mu - r2) * y + freq * x
        mag = hypot(dx, dy)
        mag > 0 || continue
        push!(px, x)
        push!(py, y)
        push!(ux, 0.075dx / mag)
        push!(uy, 0.075dy / mag)
    end
    arrows!(ax, px, py, ux, uy; color=(:gray45, 0.55), linewidth=1.2,
        arrowsize=7)

    theta = collect(range(0, 2pi; length=360))
    rstar = sqrt(mu)
    lines!(ax, rstar .* cos.(theta), rstar .* sin.(theta);
        color=(:black, 0.65), linewidth=2.6, label="limit cycle")
    for (u0, color) in zip([(1.05, 0.04), (0.15, 0.95), (-0.95, -0.05)], [BLUE, RED, GREEN])
        tr = simulate_stuart_landau(mu, shear; u0=u0, T=6.0, burn=0.0)
        lines!(ax, tr[:, 1], tr[:, 2]; color=(color, 0.82), linewidth=2.0)
        scatter!(ax, [tr[1, 1]], [tr[1, 2]]; color=color, marker=:diamond, markersize=8)
    end
    text!(ax, -1.08, 1.04; text=@sprintf("mu=%.2f, c=%.1f", mu, shear),
        fontsize=15, color=GRAY)
    ax.aspect = DataAspect()
    return nothing
end

function dmd_rollout_xy(result, trial; n_steps)
    pred = rollout_prediction_error(result, trial; n_steps=n_steps)
    return pred.y, pred.yhat
end

function unit_scale(x)
    v = Float64.(x)
    lo, hi = extrema(filter(isfinite, v))
    hi == lo && return fill(0.5, length(v))
    return (v .- lo) ./ (hi - lo)
end

function moving_average(x; window=7)
    v = Float64.(x)
    half = div(window, 2)
    out = similar(v)
    for i in eachindex(v)
        lo = max(firstindex(v), i - half)
        hi = min(lastindex(v), i + half)
        vals = v[lo:hi]
        vals = vals[isfinite.(vals)]
        out[i] = isempty(vals) ? NaN : mean(vals)
    end
    return out
end

function heatmap_text_color(value)
    return value > 0.65 ? :black : :white
end

function radial_floquet_multiplier(mu, shear; omega=2pi)
    period = 2pi / (omega + shear * mu)
    return exp(-2 * mu * period)
end

function eigen_features(A; n=6)
    vals = eigvals(A)
    moduli = sort(abs.(vals); rev=true)
    out = fill(NaN, n)
    out[1:min(n, length(moduli))] .= moduli[1:min(n, length(moduli))]
    return out
end

function radial_eigen_candidate(A; fs=FS, unit_margin=0.003)
    vals = eigvals(A)
    mods = abs.(vals)
    inside = findall(m -> isfinite(m) && m < 1 - unit_margin && m > 0, mods)
    if isempty(inside)
        return (modulus=NaN, decay_rate=NaN, timescale=NaN, n_unit=count(abs.(mods .- 1) .<= unit_margin))
    end
    idx = inside[argmax(mods[inside])]
    modulus = mods[idx]
    decay_rate = -fs * log(modulus)
    return (
        modulus=modulus,
        decay_rate=decay_rate,
        timescale=decay_rate > 0 ? 1 / decay_rate : NaN,
        n_unit=count(abs.(mods .- 1) .<= unit_margin),
    )
end

function figure2_metric_table()
    rows = NamedTuple[]
    total = length(DECODE_MU_GRID) * length(DECODE_SHEAR_GRID)
    done = 0
    t0 = time()
    for (imu, mu) in enumerate(DECODE_MU_GRID)
        @printf("[figure 2] decoder grid μ row %d/%d (μ=%.3f)\n", imu, length(DECODE_MU_GRID), mu)
        flush(stdout)
        for shear in DECODE_SHEAR_GRID
            done += 1
            m = operator_metrics(simulate_stuart_landau(mu, shear; burn=0.0, T=35.0); n_delays=60,
                f_cycle=sl_frequency(mu, shear))
            efeat = eigen_features(m.result.A; n=6)
            radial_eig = radial_eigen_candidate(m.result.A)
            push!(rows, (
                mu=mu,
                shear=shear,
                floquet_radial=radial_floquet_multiplier(mu, shear),
                log_floquet_radial=log(radial_floquet_multiplier(mu, shear)),
                analytic_frequency=sl_frequency(mu, shear),
                floquet_decay_rate=-log(radial_floquet_multiplier(mu, shear)) * sl_frequency(mu, shear),
                fitted_resolvent_frequency=m.peak_freq,
                eig1_abs=efeat[1],
                eig2_abs=efeat[2],
                eig3_abs=efeat[3],
                eig4_abs=efeat[4],
                eig5_abs=efeat[5],
                eig6_abs=efeat[6],
                eig_radial_candidate_abs=radial_eig.modulus,
                eig_radial_candidate_decay_rate=radial_eig.decay_rate,
                eig_radial_candidate_timescale=radial_eig.timescale,
                eig_n_unit_margin=radial_eig.n_unit,
                henrici=m.henrici,
                # `henrici` is the normalised commutator ||A*A-AA*||/||A||^2, which is
                # QUADRATIC in the operator; its square root is the amplitude-scaled
                # version and is the better-behaved regressor (single-metric LOOCV
                # R^2 0.88 -> 0.90 against the Floquet decay rate).
                sqrt_henrici=sqrt(max(m.henrici, 0.0)),
                log_tg=log10(max(m.tg, eps())),
                log_q=log10(max(m.q, eps())),
                participation=effective_rank(m.result.singular_values),
                resolvent_peak_gain=m.peak_gain,
            ))
        end
        @printf("[figure 2] decoder grid completed %d/%d fits (%.1f s)\n", done, total, time() - t0)
        flush(stdout)
    end
    return DataFrame(rows)
end

function figure2_decoder_summary(df)
    rows = NamedTuple[]
    ok = isfinite.(df.floquet_decay_rate)
    sub = df[ok, :]
    y = Vector{Float64}(sub.floquet_decay_rate)
    feature_sets = Dict(
        "Henrici" => [:sqrt_henrici],
        "transient growth" => [:log_tg],
        "Q factor" => [:log_q],
        "participation" => [:participation],
        "peak resolvent gain" => [:resolvent_peak_gain],
        "all response metrics" => [:sqrt_henrici, :log_tg, :log_q, :participation, :resolvent_peak_gain],
        "DMD eigenvalue moduli" => [:eig1_abs, :eig2_abs, :eig3_abs, :eig4_abs, :eig5_abs, :eig6_abs],
        "radial DMD eigenvalue modulus" => [:eig3_abs],
        "largest inside-unit eigenvalue" => [:eig_radial_candidate_decay_rate],
    )
    for (name, cols) in feature_sets
        X = Matrix{Float64}(sub[:, cols])
        finite_rows = vec(all(isfinite.(X); dims=2))
        pred = loocv_linear(X[finite_rows, :], y[finite_rows])
        push!(rows, (
            method=name,
            target="continuous-time Floquet decay rate",
            r2=r2_identity(pred, y[finite_rows]),
            rmse=rmse(pred, y[finite_rows]),
            n=count(finite_rows),
        ))
    end
    for col in [:henrici, :sqrt_henrici, :log_tg, :log_q, :participation, :resolvent_peak_gain,
                :eig1_abs, :eig2_abs, :eig3_abs, :eig4_abs, :eig5_abs, :eig6_abs]
        push!(rows, (
            method=String(col) * " correlation",
            target="continuous-time Floquet decay rate",
            r2=robust_cor(sub[!, col], y)^2,
            rmse=NaN,
            n=nrow(sub),
        ))
    end
    return DataFrame(rows)
end

function figure2_decoder_plot_tables(df)
    target = :floquet_decay_rate
    features = [:sqrt_henrici, :log_tg, :log_q, :participation, :resolvent_peak_gain]
    X = Matrix{Float64}(df[:, features])
    y = Vector{Float64}(df[!, target])
    keep_response = vec(all(isfinite.(X); dims=2)) .& isfinite.(y)
    X_response, y_response = X[keep_response, :], y[keep_response]
    pred = loocv_linear(X_response, y_response)
    μ, σ = standardize_train(X_response)
    β = hcat(ones(length(y_response)), (X_response .- μ') ./ σ') \ y_response

    eigx_all = reshape(Float64.(df.eig_radial_candidate_decay_rate), :, 1)
    keep_eig = vec(isfinite.(eigx_all[:, 1])) .& isfinite.(y)
    eig_pred = loocv_linear(eigx_all[keep_eig, :], y[keep_eig])

    predictions = DataFrame(
        truth=y_response,
        prediction=pred,
        decoder=fill("response metrics", length(y_response)),
        sqrt_henrici=X_response[:, 1],
        log_tg=X_response[:, 2],
        eig_radial_candidate_decay_rate=fill(NaN, length(y_response)),
        shear=Float64.(df[keep_response, :shear]),
    )
    append!(predictions, DataFrame(
        truth=y[keep_eig],
        prediction=eig_pred,
        decoder=fill("largest inside-unit eigenvalue", length(eig_pred)),
        sqrt_henrici=fill(NaN, length(eig_pred)),
        log_tg=fill(NaN, length(eig_pred)),
        eig_radial_candidate_decay_rate=vec(eigx_all[keep_eig, 1]),
        shear=Float64.(df[keep_eig, :shear]),
    ), cols=:union)
    coefficients = DataFrame(feature=["intercept"; String.(features)], coefficient=β)
    return predictions, coefficients
end

# NOTE: default must honour FORCE_FIGURE2 -- both callers invoke this with no
# arguments, so a `force=false` default silently pinned the decoder panels to
# stale caches even on a forced rerun.
function compute_figure2_decoder_caches(; force=FORCE_FIGURE2)
    metrics_path = joinpath(RESULTS_DIR, "figure2_sl_floquet_metrics.csv")
    summary_path = joinpath(RESULTS_DIR, "figure2_sl_floquet_decoders.csv")
    predictions_path = joinpath(RESULTS_DIR, "figure2_sl_floquet_decoder_predictions.csv")
    coefficients_path = joinpath(RESULTS_DIR, "figure2_sl_floquet_decoder_coefficients.csv")
    if !force && all(isfile, [metrics_path, summary_path, predictions_path, coefficients_path])
        metrics = CSV.read(metrics_path, DataFrame)
        predictions = CSV.read(predictions_path, DataFrame)
        summary = CSV.read(summary_path, DataFrame)
        required_metrics = [:eig_radial_candidate_decay_rate, :eig_n_unit_margin]
        required_predictions = [:decoder, :eig_radial_candidate_decay_rate]
        required_methods = ["largest inside-unit eigenvalue", "all response metrics"]
        if all(in(propertynames(metrics)), required_metrics) &&
                all(in(propertynames(predictions)), required_predictions) &&
                all(in(summary.method), required_methods)
            return (
                metrics,
                summary,
                predictions,
                CSV.read(coefficients_path, DataFrame),
            )
        end
        println("Figure 2 decoder cache is missing eigenvalue-candidate fields; recomputing.")
    end
    metrics = figure2_metric_table()
    summary = figure2_decoder_summary(metrics)
    predictions, coefficients = figure2_decoder_plot_tables(metrics)
    CSV.write(metrics_path, metrics)
    CSV.write(summary_path, summary)
    CSV.write(predictions_path, predictions)
    CSV.write(coefficients_path, coefficients)
    return metrics, summary, predictions, coefficients
end

function floquet_decoder_summary()
    _, df, _, _ = compute_figure2_decoder_caches()
    keep = [
        "Henrici",
        "transient growth",
        "Q factor",
        "participation",
        "peak resolvent gain",
        "all response metrics",
        "DMD eigenvalue moduli",
        "radial DMD eigenvalue modulus",
    ]
    return df[in.(df.method, Ref(keep)), :]
end

function floquet_decoder_plot_data()
    _, _, predictions, coefficients = compute_figure2_decoder_caches()
    return predictions, coefficients
end

function linear_fit(x, y)
    β = hcat(ones(length(x)), x) \ y
    return β[1], β[2], cor(x, y)^2
end

function compute_or_load_q_map(mu_grid, shear_grid)
    path = joinpath(RESULTS_DIR, "figure2_q_map_noisy.csv")
    if isfile(path) && !FORCE_FIGURE2
        println("Loading cached data-driven Q map: $path")
        df = CSV.read(path, DataFrame)
        if nrow(df) == length(mu_grid) * length(shear_grid) &&
                maximum(abs.(sort(unique(df.mu)) .- mu_grid)) < 1e-9 &&
                maximum(abs.(sort(unique(df.shear)) .- shear_grid)) < 1e-9
            q_map = fill(NaN, length(mu_grid), length(shear_grid))
            mu_idx = Dict(mu => i for (i, mu) in enumerate(mu_grid))
            shear_idx = Dict(sh => j for (j, sh) in enumerate(shear_grid))
            for row in eachrow(df)
                q_map[mu_idx[row.mu], shear_idx[row.shear]] = row.q
            end
            return q_map
        end
        println("Cached Q map does not match current grid; recomputing.")
    end
    q_map = fill(NaN, length(mu_grid), length(shear_grid))
    total = length(mu_grid) * length(shear_grid)
    done = 0
    t0 = time()
    println("Computing data-driven (noisy) Q map -- $(length(SL_Q_NOISE_SEEDS)) realizations/cell, $total cells...")
    for (i, mu) in enumerate(mu_grid), (j, shear) in enumerate(shear_grid)
        q_map[i, j] = data_driven_q_factor(mu, shear)
        done += 1
        done % 200 == 0 && (@printf("[figure 2] Q map %d/%d cells (%.1f s)\n", done, total, time() - t0); flush(stdout))
    end
    @printf("[figure 2] Q map done: %d cells (%.1f s)\n", total, time() - t0)
    df = DataFrame(
        mu=repeat(mu_grid, inner=length(shear_grid)),
        shear=repeat(shear_grid, outer=length(mu_grid)),
        q=vec(permutedims(q_map)),
    )
    CSV.write(path, df)
    println("Wrote data-driven Q map: $path")
    return q_map
end

function make_figure2()
    fig = Figure(size=(1800, 1460), fontsize=PAPER_FONT_SIZE)

    mu_grid = MU_GRID
    shear_grid = SHEAR_GRID
    henrici_map = fill(NaN, length(mu_grid), length(shear_grid))
    tg_map = similar(henrici_map)
    peak_gain_map = similar(henrici_map)
    peak_freq_map = similar(henrici_map)
    true_freq_map = similar(henrici_map)
    total_map = length(mu_grid) * length(shear_grid)
    done_map = 0
    t_map = time()
    for (i, mu) in enumerate(mu_grid)
        @printf("[figure 2] heatmap grid μ row %d/%d (μ=%.3f)\n", i, length(mu_grid), mu)
        flush(stdout)
        for (j, shear) in enumerate(shear_grid)
            done_map += 1
            m = operator_metrics(simulate_stuart_landau(mu, shear; burn=0.0, T=35.0); n_delays=60,
                f_cycle=sl_frequency(mu, shear))
            henrici_map[i, j] = m.henrici
            tg_map[i, j] = m.tg
            peak_gain_map[i, j] = m.peak_gain
            peak_freq_map[i, j] = m.peak_freq
            true_freq_map[i, j] = sl_frequency(mu, shear)
        end
        @printf("[figure 2] heatmap grid completed %d/%d fits (%.1f s)\n", done_map, total_map, time() - t_map)
        flush(stdout)
    end
    # Q is fit separately, from noisy (not deterministic) trajectories -- see
    # the SL_Q_NOISE_* block and `data_driven_q_factor` docstring above for why.
    q_map = compute_or_load_q_map(mu_grid, shear_grid)

    example_mu = 0.25
    example_shear = 7.0
    example_period = round(Int, FS / sl_frequency(example_mu, example_shear))
    example_delays = example_period

    axA = Axis(fig[1, 1], title="Stuart-Landau phase-space setup")
    panel_label!(fig, (1, 1), "A")
    draw_stuart_landau_setup!(axA; mu=example_mu, shear=example_shear)
    text!(axA, 0.04, 0.11;
        text=L"\dot{x}=(\mu-r^2)x-(\omega+cr^2)y",
        space=:relative, align=(:left, :bottom), fontsize=15, color=GRAY)
    text!(axA, 0.04, 0.04;
        text=L"\dot{y}=(\mu-r^2)y+(\omega+cr^2)x",
        space=:relative, align=(:left, :bottom), fontsize=15, color=GRAY)

    train_trials = [
        simulate_stuart_landau(example_mu, example_shear; u0=u0, burn=0.0, T=16.0)
        for u0 in [
            (1.08, 0.04), (0.12, 1.08), (-1.08, 0.12), (0.10, -1.08),
            (0.08, 0.03), (-0.06, 0.08), (0.10, -0.05), (-0.08, -0.04),
        ]
    ]
    heldout_specs = [
        ("outside IC", (0.98, -0.18), BLUE),
        ("inside IC", (0.04, 0.08), RED),
    ]
    heldout_trials = [
        (label, simulate_stuart_landau(example_mu, example_shear; u0=u0, burn=0.0, T=10.0), color)
        for (label, u0, color) in heldout_specs
    ]
    hroll = HankelGenerator(n_delays=example_delays, delay_interval=1)
    result_roll = fit(DMD(rank=EnergyRank(1e-6), stabilize=true), train_trials, hroll)
    glRoll = GridLayout(fig[1, 2:3])
    axC1 = Axis(glRoll[1, 1], title="Outside IC",
        xlabel="time (s)", ylabel="state")
    axC2 = Axis(glRoll[1, 2], title="Inside IC",
        xlabel="time (s)", ylabel="state")
    panel_label!(fig, (1, 2), "B")
    Label(glRoll[0, 1], "DMD transient rollouts";
        fontsize=18, tellheight=true, padding=(0, 0, 0, 4))
    rollout_axes = [axC1, axC2]
    for (i, (label, trial, _color)) in enumerate(heldout_trials)
        horizon = min(120, size(trial, 1))
        y, yhat = dmd_rollout_xy(result_roll, trial; n_steps=horizon)
        ax = rollout_axes[i]
        t = (0:size(y, 1)-1) .* DT
        delay_end = (min(example_delays, size(y, 1)) - 1) * DT
        vspan!(ax, 0, delay_end; color=(:gray35, 0.12), label=i == 1 ? "delay window" : nothing)
        lines!(ax, t, y[:, 1]; color=(BLUE, 0.62), linewidth=2.0,
            label=i == 1 ? "x data" : nothing)
        history = 1:min(example_delays, size(y, 1))
        forecast = last(history):size(y, 1)
        lines!(ax, t[history], yhat[history, 1]; color=BLUE, linewidth=2.2,
            label=i == 1 ? "x DMD (history)" : nothing)
        lines!(ax, t[forecast], yhat[forecast, 1]; color=BLUE, linewidth=2.2, linestyle=:dash,
            label=i == 1 ? "x DMD forecast" : nothing)
        lines!(ax, t, y[:, 2]; color=(RED, 0.62), linewidth=2.0,
            label=i == 1 ? "y data" : nothing)
        lines!(ax, t[history], yhat[history, 2]; color=RED, linewidth=2.2,
            label=i == 1 ? "y DMD (history)" : nothing)
        lines!(ax, t[forecast], yhat[forecast, 2]; color=RED, linewidth=2.2, linestyle=:dash,
            label=i == 1 ? "y DMD forecast" : nothing)
        text!(ax, 0.97, 0.08; text=label, space=:relative,
            align=(:right, :bottom), fontsize=12, color=GRAY)
    end
    linkxaxes!(axC1, axC2)
    xlims!(axC1, 0, (120 - 1) * DT)
    ylims!(axC1, -1.05, 1.05)
    ylims!(axC2, -1.05, 1.05)
    Legend(glRoll[2, 1:2], axC1; framevisible=false, orientation=:horizontal,
        nbanks=1, tellwidth=false)
    colsize!(glRoll, 1, Relative(0.5))
    colsize!(glRoll, 2, Relative(0.5))
    rowgap!(glRoll, 4)
    colgap!(glRoll, 18)

    glA = GridLayout(fig[2, 1])
    axA = Axis(glA[1, 1], title="Fitted non-normality", xlabel="c", ylabel="μ")
    panel_label!(fig, (2, 1), "C")
    hmA = heatmap!(axA, shear_grid, mu_grid, henrici_map'; colormap=HEATMAP_CMAP)
    Colorbar(glA[1, 2], hmA; label="Henrici departure", width=10)
    colgap!(glA, 5)

    glB = GridLayout(fig[2, 2])
    axB = Axis(glB[1, 1], title="Transient growth", xlabel="c", ylabel="μ")
    panel_label!(fig, (2, 2), "D")
    hmB = heatmap!(axB, shear_grid, mu_grid, log10.(tg_map'); colormap=HEATMAP_CMAP)
    Colorbar(glB[1, 2], hmB; label=L"\log_{10}\max_k \|A^k\|_2", width=10)
    colgap!(glB, 5)

    glC = GridLayout(fig[2, 3])
    # Noise-driven linewidth, not the noiseless-trajectory Q: on data with no
    # decaying transient in it (sampled after burn-in, on the limit cycle),
    # "Q" from a bisection search is just f0 divided by the search's own
    # resolution floor, which increases with c only because f0 = (omega +
    # c*mu)/2pi does. With process noise present, shear gets to do its actual
    # job -- convert noise-driven amplitude excursions into phase drift -- and
    # Q falls with c as expected for a linewidth-broadening mechanism.
    axC = Axis(glC[1, 1], title="Q factor (noise-driven)", xlabel="c", ylabel="μ")
    panel_label!(fig, (2, 3), "E")
    hmC = heatmap!(axC, shear_grid, mu_grid, log10.(q_map'); colormap=HEATMAP_CMAP)
    Colorbar(glC[1, 2], hmC; label=L"\log_{10} Q", width=10)
    colgap!(glC, 5)

    ggrid = GridLayout(fig[3, 1:3])
    panel_label!(fig, (3, 1), "F")
    decoder_predictions, decoder_coefficients = floquet_decoder_plot_data()
    response_predictions = decoder_predictions[decoder_predictions.decoder .== "response metrics", :]
    truth = response_predictions.truth
    axG1 = Axis(ggrid[1, 1], title="Non-normality",
        xlabel=L"\sqrt{\|A^*A-AA^*\|_2/\|A\|_2^2}", ylabel="Floquet decay rate (s⁻¹)")
    axG2 = Axis(ggrid[1, 2], title="Transient growth", xlabel="log₁₀ transient gain", ylabel="")
    axG3 = Axis(ggrid[1, 3], title="All response metrics", xlabel="True Floquet decay rate (s⁻¹)",
        ylabel="Predicted Floquet decay rate (s⁻¹)")
    g_colormap = :plasma
    g_colorrange = (minimum(response_predictions.shear), maximum(response_predictions.shear))
    for (ax, x) in [(axG1, response_predictions.sqrt_henrici),
                    (axG2, response_predictions.log_tg)]
        intercept, slope, r2 = linear_fit(x, truth)
        scatter!(ax, x, truth; color=response_predictions.shear, colormap=g_colormap,
            colorrange=g_colorrange, markersize=5)
        text!(ax, 0.04, 0.06; text=@sprintf("R² = %.2f", r2), space=:relative,
            align=(:left, :bottom), fontsize=13, color=GRAY)
    end
    gscatter = scatter!(axG3, response_predictions.truth, response_predictions.prediction;
        color=response_predictions.shear, colormap=g_colormap, colorrange=g_colorrange,
        markersize=6, marker=:circle)
    lo, hi = extrema(vcat(response_predictions.truth, response_predictions.prediction))
    pad = 0.04 * (hi - lo)
    lines!(axG3, [lo - pad, hi + pad], [lo - pad, hi + pad]; color=GRAY, linestyle=:dash, linewidth=1.5)
    xlims!(axG3, lo - pad, hi + pad); ylims!(axG3, lo - pad, hi + pad)
    decoder_r2 = r2_identity(response_predictions.prediction, response_predictions.truth)
    βlabels = Dict(
        "sqrt_henrici" => "√Henrici", "log_tg" => "log TG", "log_q" => "log Q",
        "participation" => "PR", "resolvent_peak_gain" => "peak gain",
    )
    coeff_text = join([@sprintf("β(%s)=%.2f", βlabels[String(row.feature)], row.coefficient)
        for row in eachrow(decoder_coefficients) if row.feature != "intercept"], "\n")
    text!(axG3, 0.03, 0.97;
        text=@sprintf("response LOOCV R² = %.3f\nŷ = β₀ + Σβⱼzⱼ\n%s\n%s", decoder_r2,
            @sprintf("β₀ = %.2f", only(decoder_coefficients.coefficient[decoder_coefficients.feature .== "intercept"])), coeff_text),
        space=:relative, align=(:left, :top), fontsize=10, color=GRAY)
    hideydecorations!(axG2; grid=false, ticks=false)
    colgap!(ggrid, 14)
    Colorbar(ggrid[1, 4], gscatter; vertical=true, label="shear, c", width=12)
    colsize!(ggrid, 4, Auto(0.08))

    iogrid = GridLayout(fig[4, 1:2])
    G_forward = coupled_linear_oscillator_control(:forward)
    G_reverse = coupled_linear_oscillator_control(:reverse)
    axF1 = Axis(iogrid[1, 1], title="1 → 2", xlabel="input", ylabel="output",
        xticks=([1, 2], ["osc. 1", "osc. 2"]), yticks=([1, 2], ["osc. 1", "osc. 2"]))
    axF2 = Axis(iogrid[1, 2], title="2 → 1", xlabel="input", ylabel="",
        xticks=([1, 2], ["osc. 1", "osc. 2"]), yticks=([1, 2], ["osc. 1", "osc. 2"]))
    panel_label!(iogrid, (1, 1), "G")
    panel_label!(iogrid, (1, 2), "H")
    hmF1 = heatmap!(axF1, 1:2, 1:2, G_forward'; colormap=HEATMAP_CMAP, colorrange=(0, 1))
    heatmap!(axF2, 1:2, 1:2, G_reverse'; colormap=HEATMAP_CMAP, colorrange=(0, 1))
    for input in 1:2, output in 1:2
        text!(axF1, input, output; text=@sprintf("%.2f", G_forward[output, input]),
            align=(:center, :center), fontsize=13,
            color=heatmap_text_color(G_forward[output, input]))
        text!(axF2, input, output; text=@sprintf("%.2f", G_reverse[output, input]),
            align=(:center, :center), fontsize=13,
            color=heatmap_text_color(G_reverse[output, input]))
    end
    Colorbar(iogrid[1, 3], hmF1; label="normalized gain", width=10)
    Label(iogrid[2, 1], @sprintf("cross log-ratio %.2f", cross_io_ratio(G_forward));
        fontsize=13, tellheight=true)
    Label(iogrid[2, 2], @sprintf("cross log-ratio %.2f", cross_io_ratio(G_reverse));
        fontsize=13, tellheight=true)
    colgap!(iogrid, 12)
    rowgap!(iogrid, 8)
    colsize!(iogrid, 1, Relative(0.47))
    colsize!(iogrid, 2, Relative(0.47))
    rowsize!(iogrid, 1, Relative(0.92))
    hideydecorations!(axF2; grid=false, ticks=false)

    colgap!(fig.layout, 28)
    rowgap!(fig.layout, 28)
    colsize!(fig.layout, 1, Relative(0.28))
    colsize!(fig.layout, 2, Relative(0.36))
    colsize!(fig.layout, 3, Relative(0.36))
    rowsize!(fig.layout, 1, Relative(0.27))
    rowsize!(fig.layout, 2, Relative(0.24))
    rowsize!(fig.layout, 3, Relative(0.22))
    rowsize!(fig.layout, 4, Relative(0.27))
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    ensure_dirs!()
    compute_figure2_decoder_caches()
    fig = make_figure2()
    savefig("Figure 2.svg", fig)
end

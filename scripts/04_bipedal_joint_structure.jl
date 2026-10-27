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
const HEATMAP_CMAP = :viridis
const FS = 40.0
const N_DELAYS = 15
const WINDOW_LEN = 240        # ~6 s, ~5 strides at FS=40 Hz (prediction-horizon panels)
const HOP = 80                # ~2 s
const WINDOW_LEN_SPEED = 160  # ~4 s, ~3.5 strides (per-window speed-scatter panels)
const HOP_SPEED = 80          # ~2 s
const TRAIN_FRAC = 0.7

const DATA_DIR = joinpath(GaitNonNormalityPaper3.ROOT, "external", "loco_mujoco_mocap", "processed")

const H1_MOTIONS = ["walk", "run", "random_walk", "stepinplace1"]
const GAIT_MOTIONS = ["walk", "run"]
const MOTION_LABELS = Dict(
    "walk" => "walk", "run" => "run",
    "random_walk" => "random walk", "stepinplace1" => "step in place",
)
# H1 and G1 adjacent: they share an ego-centered DOF set that maps one-to-one
# (see COMPARABLE_DOF_LABELS), so panel E reads as a direct side-by-side.
# Ordering is display-only -- every consumer builds a Dict keyed by robot,
# and bootstrap_metrics seeds its own RNG per call.
const MECH_ROBOTS = ["UnitreeH1", "UnitreeG1", "Atlas"]
const ROBOT_COLORS = Dict("UnitreeH1" => BLUE, "Atlas" => RED, "UnitreeG1" => GOLD)
const MOTION_COLORS = Dict(
    "walk" => BLUE, "run" => GREEN, "random_walk" => GOLD, "stepinplace1" => PURPLE,
)
const FORCE_RECOMPUTE = lowercase(get(ENV, "FORCE_FIGURE4", "false")) in ("1", "true", "yes")

function timed_stage(f::Function, label::AbstractString)
    println("[figure 4] starting: ", label)
    flush(stdout)
    t0 = time()
    value = f()
    @printf("[figure 4] finished: %s (%.1f s)\n", label, time() - t0)
    flush(stdout)
    return value
end

const COMPARABLE_DOF_LABELS = Dict(
    "UnitreeH1" => [
        ("hip_rotation_l", "hip rot L"),
        ("hip_adduction_l", "hip add L"),
        ("hip_flexion_l", "hip flex L"),
        ("knee_l", "knee L"),
        ("ankle_l", "ankle pitch L"),
        ("hip_rotation_r", "hip rot R"),
        ("hip_adduction_r", "hip add R"),
        ("hip_flexion_r", "hip flex R"),
        ("knee_r", "knee R"),
        ("ankle_r", "ankle pitch R"),
    ],
    "Atlas" => [
        ("hip_rotation_l", "hip rot L"),
        ("hip_adduction_l", "hip add L"),
        ("hip_flexion_l", "hip flex L"),
        ("knee_l", "knee L"),
        ("ankle_l", "ankle pitch L"),
        ("hip_rotation_r", "hip rot R"),
        ("hip_adduction_r", "hip add R"),
        ("hip_flexion_r", "hip flex R"),
        ("knee_r", "knee R"),
        ("ankle_r", "ankle pitch R"),
    ],
    "UnitreeG1" => [
        ("hip_yaw_l", "hip rot L"),
        ("hip_roll_l", "hip add L"),
        ("hip_pitch_l", "hip flex L"),
        ("knee_l", "knee L"),
        ("ankle_pitch_l", "ankle pitch L"),
        ("hip_yaw_r", "hip rot R"),
        ("hip_roll_r", "hip add R"),
        ("hip_pitch_r", "hip flex R"),
        ("knee_r", "knee R"),
        ("ankle_pitch_r", "ankle pitch R"),
    ],
)

# --- data loading -----------------------------------------------------------

function load_channels(robot, motion)
    path = joinpath(DATA_DIR, "$(robot)_$(motion).csv")
    df = CSV.read(path, DataFrame)
    labels = String.(filter(!=("speed_mps"), names(df)))
    X = Matrix{Float64}(df[:, labels])
    speed = Float64.(df.speed_mps)
    return X, labels, speed
end

function window_trials(X, speed; len=WINDOW_LEN, hop=HOP, first_row=1, last_row=size(X, 1))
    trials = Matrix{Float64}[]
    speeds = Float64[]
    starts = Int[]
    t = first_row
    while t + len - 1 <= last_row
        w = X[t:t+len-1, :]
        push!(trials, w .- mean(w; dims=1))
        push!(speeds, mean(@view speed[t:t+len-1]))
        push!(starts, t)
        t += hop
    end
    return trials, speeds, starts
end

function train_test_windows(X, speed; len=WINDOW_LEN, hop=HOP, train_frac=TRAIN_FRAC)
    n = size(X, 1)
    split = floor(Int, train_frac * n)
    train, train_speed, train_starts = window_trials(X, speed; len=len, hop=hop, first_row=1, last_row=split)
    test, test_speed, test_starts = window_trials(X, speed; len=len, hop=hop, first_row=split + 1, last_row=n)
    return train, test, train_speed, test_speed, train_starts, test_starts, split
end

# --- fitting and metrics -----------------------------------------------------

# Per-joint resolvent input-output map from the delay-embedded operator: for each
# (output joint, input joint) pair, the operator 2-norm of the resolvent block
# linking that joint's whole lag stack to the other's. Identical in form to the
# function of the same name in scripts/05 and 06, so the robot maps and the human
# maps are computed the same way and can be compared directly.
function hankel_block_resolvent_io_matrix(result, freq; fs=FS)
    A = result.A
    r = size(A, 1)
    z = exp(2im * pi * freq / fs)
    R = result.U * inv(z * I(r) - A) * result.U'
    out = zeros(Float64, result.n_features, result.n_features)
    for j_out in 1:result.n_features, j_in in 1:result.n_features
        idx_out = DMDAnalysis.hankel_feature_indices([j_out];
            n_delays=result.hankel.n_delays, n_features=result.n_features)
        idx_in = DMDAnalysis.hankel_feature_indices([j_in];
            n_delays=result.hankel.n_delays, n_features=result.n_features)
        out[j_out, j_in] = opnorm(R[idx_out, idx_in], 2)
    end
    return out
end

function fit_operator(trials; n_delays=N_DELAYS)
    isempty(trials) && return nothing
    try
        h = HankelGenerator(n_delays=n_delays, delay_interval=1)
        model = DMD(rank=EnergyRank(1e-6), stabilize=true)
        return fit(model, trials, h)
    catch
        return nothing
    end
end

function future_rollout_prediction_error(result, trial; n_steps)
    n_steps == 0 && return (nrmse=0.0,)
    n_init = result.hankel.n_delays
    n_steps <= size(trial, 1) - n_init || return (nrmse=NaN,)
    Hhat = rollout(result, trial; n_steps=n_init + n_steps)
    yhat_all = permutedims(Hhat[1:result.n_features, :])
    idx = n_init+1:n_init+n_steps
    yhat = yhat_all[idx, :]
    y = Matrix{Float64}(trial)[idx, 1:result.n_features]
    rmse = sqrt(mean(abs2, yhat .- y))
    return (nrmse=rmse / max(sqrt(mean(abs2, y)), eps(Float64)),)
end

spectral_radius(A) = maximum(abs.(eigvals(A)))
effective_rank(s) = (sum(s)^2) / max(sum(abs2, s), eps())

# Stride period, in samples, from the multi-joint autocorrelation peak, refined
# to sub-sample precision by parabolic interpolation.
#
# Two octave traps have to be dodged, and they pull in opposite directions:
#
#  - The FFT estimate this replaces is bin-limited. `estimate_stride_frequency`
#    on a WINDOW_LEN=240 window has a bin width of fs/240 = 0.167 Hz, a 19%
#    quantization at H1's ~0.88 Hz walking stride, and coarse enough that the
#    estimator's own subharmonic guard (tolerance max(0.08, 1.5*fs/n_t) =
#    0.25 Hz here, wider than the gap between candidates) accepts a spurious
#    f_peak/2. It reported 0.667 Hz for H1 walk against a true 0.880.
#  - Autocorrelation has the mirror failure: a T-periodic signal peaks almost
#    as strongly at 2T, 3T, ... as at T, so taking the global argmax picks a
#    MULTIPLE of the stride. Unguarded, this returned lag 90 (0.444 Hz) on more
#    than half of the 240-sample walk windows -- exactly half the true rate.
#
# So the peak is not taken as the global argmax: it is the SMALLEST local
# maximum reaching `peak_tol` of the global one, which is the fundamental
# whenever the harmonic ladder is present. Half-stride (step) lags do not
# compete, because left and right joints are antiphase there and the
# across-joint mean correlation goes negative.
function refine_stride_period(Xc::AbstractMatrix; fs=FS, fmin=0.2, fmax=3.0, peak_tol=0.85)
    n = size(Xc, 1)
    lo = max(4, floor(Int, fs / fmax))
    hi = min(div(n, 2) - 2, ceil(Int, fs / fmin))
    hi > lo + 2 || return NaN
    lags = lo:hi
    v = [mean(cor(view(Xc, 1:n-lag, j), view(Xc, lag+1:n, j)) for j in 1:size(Xc, 2))
         for lag in lags]
    all(isfinite, v) || return NaN
    vbest = maximum(v)
    vbest > 0 || return NaN
    # Walk up from the shortest lag and stop at the first local max that is
    # within peak_tol of the tallest one -- the fundamental, not its multiples.
    best_lag = lags[argmax(v)]
    for k in 2:length(v)-1
        if v[k] >= peak_tol * vbest && v[k] >= v[k-1] && v[k] >= v[k+1]
            best_lag = lags[k]
            break
        end
    end
    k = best_lag - lo + 1
    (k <= 1 || k >= length(v)) && return Float64(best_lag)
    ym1, y0, yp1 = v[k-1], v[k], v[k+1]
    denom = ym1 - 2y0 + yp1
    abs(denom) > eps() || return Float64(best_lag)
    return best_lag + 0.5 * (ym1 - yp1) / denom
end

"""Stride frequency (Hz) of a set of trials: the median of each trial's own
autocorrelation estimate. Prefer the continuous-recording form where a
continuous segment is available -- a 240-sample window holds only ~5 strides,
so the per-window estimate is noisier even with octave protection."""
function stride_frequency(trials::AbstractVector; fs=FS)
    fs_est = Float64[]
    for tr in trials
        Xc = Matrix{Float64}(tr) .- mean(tr; dims=1)
        p = refine_stride_period(Xc; fs=fs)
        isfinite(p) && p > 0 && push!(fs_est, fs / p)
    end
    return isempty(fs_est) ? NaN : median(fs_est)
end

function stride_frequency(X::AbstractMatrix; fs=FS)
    p = refine_stride_period(Matrix{Float64}(X) .- mean(X; dims=1); fs=fs)
    return (isfinite(p) && p > 0) ? fs / p : NaN
end

# Phase-averaged baseline template, folded from the *continuous, un-windowed*
# training segment (never the held-out test trial). Three earlier bugs here:
#
# 1. Wrapping the test trial onto itself, so for horizons <= one stride period
#    yhat[i] = trial[mod1(i, period)] = trial[i] exactly -- i.e. "predicting"
#    the test data using the test data, giving trivially ~zero error for the
#    first stride and then a sharp jump once that coincidence ran out.
# 2. After fixing (1) by folding training *windows* instead: those windows
#    were already independently demeaned and start at different absolute
#    times (overlapping every `hop` samples, not every `period` samples), so
#    window-local index i does not correspond to a fixed gait phase across
#    windows. Averaging them by local index folded together samples from many
#    different true phases -- a phase-scrambled, nearly-flat blend.
# 3. Folding the continuous signal at a fixed INTEGER period, taken from a
#    per-window FFT estimate of f0. Both halves of that were wrong. The window
#    FFT put H1 walk's period at 60 samples instead of 45.4 (see
#    `refine_stride_period`), and even with the right frequency, folding at
#    `round(period)` drifts by the rounding error every cycle -- 0.44 samples
#    per cycle here, ~9 samples over the 20 folded cycles, a fifth of a stride.
#    The two together left the template holding 5.6% of the signal's amplitude:
#    a nearly flat line. `template_affine` then scaled that flat line up ~18x
#    to match the initialization window, turning an underfit that would have
#    scored NRMSE ~= 1 into an amplified, wrong-phase NRMSE ~= 1.4.
#
# This version takes the period from `refine_stride_period` at sub-sample
# precision and folds by fractional phase with linear interpolation, so
# rounding cannot accumulate. Only the trailing `n_cycles` periods are folded:
# the recording spans 0.79-1.66 m/s and stride frequency moves with speed, so
# folding the whole thing blurs the template for a real physical reason rather
# than a numerical one (empirically 60 cycles scores NRMSE 0.44 vs 20 cycles'
# 0.27). The trailing cycles are also the ones adjacent to the test split.
function build_phase_template(train_continuous::AbstractMatrix, period_f; n_cycles=20)
    (isfinite(period_f) && period_f >= 8) || return nothing
    n_bins = clamp(round(Int, period_f), 8, 10_000)
    n_total = size(train_continuous, 1)
    span = min(n_total, ceil(Int, n_cycles * period_f))
    row0 = n_total - span + 1
    Xc = train_continuous .- mean(train_continuous; dims=1)
    acc = zeros(n_bins, size(Xc, 2))
    counts = zeros(n_bins)
    for i in row0:n_total
        # True fractional phase of sample i, split across the two nearest bins.
        ph = mod(i / period_f, 1.0) * n_bins
        b0 = floor(Int, ph)
        w = ph - b0
        i0 = mod(b0, n_bins) + 1
        i1 = mod(b0 + 1, n_bins) + 1
        @views acc[i0, :] .+= (1 - w) .* Xc[i, :]
        @views acc[i1, :] .+= w .* Xc[i, :]
        counts[i0] += 1 - w
        counts[i1] += w
    end
    any(<(1e-9), counts) && return nothing
    return acc ./ counts, n_bins
end

# A fixed nominal `period` (`round(fs/f0)`) drifts out of true phase over any
# large time gap, so anchoring the template to the test trial via absolute row
# index (an earlier version of this function) still fails once the test
# window is far in time from wherever the template was built. Instead, this
# version finds the template phase offset that best matches the test trial's
# own first `n_init` samples -- the same information DMD's rollout already
# conditions on as its initial condition (an N_DELAYS-sample history), so this
# is not looking at the future -- and only then rolls the template forward
# from that offset for the actual forecast comparison.
function template_affine(pred_init, init)
    n_features = size(init, 2)
    scale = ones(n_features)
    delta = zeros(n_features)
    for j in 1:n_features
        x = pred_init[:, j]
        y = init[:, j]
        mx, my = mean(x), mean(y)
        xc = x .- mx
        yc = y .- my
        denom = sum(abs2, xc)
        if denom > eps()
            scale[j] = sum(xc .* yc) / denom
        end
        delta[j] = my - scale[j] * mx
    end
    return scale, delta
end

function template_prediction_error(template, period, trial; n_steps, n_init=N_DELAYS, affine_correct=true)
    template === nothing && return NaN
    n_steps == 0 && return 0.0
    n_init = min(n_init, size(trial, 1), period)
    n_steps <= size(trial, 1) - n_init || return NaN
    init = @view trial[1:n_init, :]
    best_off, best_err = 0, Inf
    best_scale = ones(size(trial, 2))
    best_delta = zeros(size(trial, 2))
    for off in 0:period-1
        pred_init = template[[mod1(off + i, period) for i in 1:n_init], :]
        scale, delta = affine_correct ? template_affine(pred_init, init) :
            (ones(size(trial, 2)), zeros(size(trial, 2)))
        err = sum(abs2, pred_init .* scale' .+ delta' .- init)
        if err < best_err
            best_err = err
            best_off = off
            best_scale = scale
            best_delta = delta
        end
    end
    idx = n_init+1:n_init+n_steps
    y = trial[idx, :]
    yhat = similar(y)
    for (row, i) in enumerate(idx)
        yhat[row, :] .= template[mod1(best_off + i, period), :] .* best_scale .+ best_delta
    end
    rmse = sqrt(mean(abs2, yhat .- y))
    return rmse / max(sqrt(mean(abs2, y)), eps())
end

function condition_metrics(result, trials)
    result === nothing && return nothing
    f0 = stride_frequency(trials; fs=FS)
    peak = resolvent_peak(result; n_angles=720, fs=FS, fmin=0.2)
    return (
        dmd_radius=spectral_radius(result.A),
        henrici=henrici_departure(result),
        log_tgmax=log10(transient_growth(result; kmax=4N_DELAYS)),
        resolvent_peak_gain=peak.gain,
        resolvent_peak_frequency=peak.frequency,
        stride_frequency=f0,
        participation=effective_rank(result.singular_values),
    )
end

function bootstrap_metrics(trials; n_boot=24, k=6, rng=MersenneTwister(2024))
    rows = NamedTuple[]
    length(trials) < k && return DataFrame(rows)
    for _ in 1:n_boot
        sample = trials[rand(rng, 1:length(trials), k)]
        result = fit_operator(sample)
        m = condition_metrics(result, sample)
        m === nothing || push!(rows, m)
    end
    return DataFrame(rows)
end

# One DMD fit per short window (like a single subject-trial in the human-gait
# papers), tagged with that window's own mean forward speed, so metrics can be
# scattered against a continuous speed axis instead of a handful of named
# motion categories.
function raw_spectral_radius(window; n_delays=N_DELAYS)
    try
        h = HankelGenerator(n_delays=n_delays, delay_interval=1)
        model = DMD(rank=EnergyRank(1e-6), stabilize=false)
        return spectral_radius(fit(model, [window], h).A)
    catch
        return NaN
    end
end

function per_window_metrics(robot, motion; len=WINDOW_LEN_SPEED, hop=HOP_SPEED)
    X, _, speed = load_channels(robot, motion)
    windows, speeds, _ = window_trials(X, speed; len=len, hop=hop, first_row=1, last_row=size(X, 1))
    rows = NamedTuple[]
    for (w, sp) in zip(windows, speeds)
        result = fit_operator([w])
        m = condition_metrics(result, [w])
        m === nothing && continue
        # `dmd_radius` from the stabilized fit is trivially ~1.0 for nearly every
        # window (stabilize=true clamps any |eigenvalue|>1 back to the unit
        # circle, and gait dynamics are almost always in that regime) -- so we
        # also record the raw, unstabilized radius, which is the quantity a
        # naive eigenvalue-first analysis would actually report.
        eig_radius_raw = raw_spectral_radius(w)
        push!(rows, merge((robot=robot, motion=motion, speed=sp, eig_radius_raw=eig_radius_raw), m))
    end
    return DataFrame(rows)
end

# --- panel helpers ------------------------------------------------------------

function panel_label!(fig, pos, label)
    Label(fig[pos..., TopLeft()], label;
        fontsize=32, font=:bold, halign=:right, valign=:bottom,
        padding=(0, 8, 0, 6), tellwidth=false, tellheight=false)
end

function draw_biped_schematic!(ax)
    hidespines!(ax)
    hidedecorations!(ax)
    xlims!(ax, -1.6, 1.9)
    ylims!(ax, -0.15, 1.9)
    lines!(ax, [-1.6, 1.9], [0, 0]; color=(:gray45, 0.8), linewidth=2)

    hip = Point2f(0, 1.15)
    for (sign, color, tag) in ((-1, BLUE, "L"), (1, RED, "R"))
        knee = Point2f(sign * 0.18, 0.60)
        ankle = Point2f(sign * 0.10, 0.0)
        lines!(ax, [hip[1], knee[1], ankle[1]], [hip[2], knee[2], ankle[2]];
            color=color, linewidth=4)
        scatter!(ax, [hip[1]], [hip[2]]; color=color, markersize=12)
        scatter!(ax, [knee[1]], [knee[2]]; color=color, markersize=10)
        scatter!(ax, [ankle[1]], [ankle[2]]; color=color, markersize=10)
        # Left-side labels hang left of their point and right-side labels right,
        # so the two ankles (0.2 apart) do not print on top of each other.
        side = sign < 0 ? (:right, :center) : (:left, :center)
        text!(ax, knee[1] + 0.08sign, knee[2]; text="knee $tag", fontsize=13, color=color, align=side)
        text!(ax, ankle[1] + 0.08sign, ankle[2] + 0.12; text="ankle $tag", fontsize=13, color=color, align=side)
    end
    scatter!(ax, [0.0], [1.15]; color=GRAY, markersize=10)
    text!(ax, 0.10, 1.08; text="hip flex./add./rot. (L,R)", fontsize=13, color=GRAY, align=(:left, :top))
    scatter!(ax, [0.0], [1.55]; color=GRAY, markersize=20)

    text!(ax, -1.55, 1.75; text="speed / control:", fontsize=14, font=:bold)
    text!(ax, -1.55, 1.60; text="walk -> run -> random walk -> step-in-place", fontsize=13)
    text!(ax, -1.55, 1.40; text="mechanics:", fontsize=14, font=:bold)
    text!(ax, -1.55, 1.25; text="UnitreeH1 (10 dof/leg-set) vs Atlas (+ankle abd.) vs UnitreeG1", fontsize=12)
    return nothing
end

function add_heatmap!(fig, pos, ax, xs, ys, M; label, colormap=HEATMAP_CMAP, colorrange=automatic)
    hm = heatmap!(ax, xs, ys, M'; colormap=colormap, colorrange=colorrange)
    Colorbar(fig[pos..., Right()], hm; label=label, width=9)
    return hm
end

function robust_cor(x, y)
    ok = isfinite.(x) .& isfinite.(y)
    count(ok) < 3 && return NaN
    return cor(Float64.(x[ok]), Float64.(y[ok]))
end

function cohens_d(group_a, group_b)
    a = filter(isfinite, Float64.(group_a))
    b = filter(isfinite, Float64.(group_b))
    length(a) >= 2 && length(b) >= 2 || return NaN
    s2 = ((length(a) - 1) * var(a) + (length(b) - 1) * var(b)) /
         (length(a) + length(b) - 2)
    s2 > 0 || return NaN
    return (mean(b) - mean(a)) / sqrt(s2)
end

function walk_run_effect_size(df, col; transform=identity)
    walk = transform.(df[df.motion .== "walk", col])
    run = transform.(df[df.motion .== "run", col])
    return cohens_d(walk, run)
end

# Ordinary-least-squares fit y ~ a + b*x over finite pairs; returns the fitted
# line evaluated at the two endpoints of x (a single straight line through the
# pooled scatter -- not a smoother, and not implying continuity between the
# discrete speed clusters the points actually come from).
function ols_line(x, y)
    ok = isfinite.(x) .& isfinite.(y)
    count(ok) < 3 && return (Float64[], Float64[])
    xs, ys = Float64.(x[ok]), Float64.(y[ok])
    xbar, ybar = mean(xs), mean(ys)
    b = sum((xs .- xbar) .* (ys .- ybar)) / max(sum(abs2, xs .- xbar), eps())
    a = ybar - b * xbar
    x_ends = [minimum(xs), maximum(xs)]
    return x_ends, a .+ b .* x_ends
end

function shared_io_matrix(io_mats, mech_data, robot, shared_labels)
    haskey(io_mats, robot) || return nothing
    labels = mech_data[robot].labels
    idx = [findfirst(==(l), labels) for l in shared_labels]
    any(isnothing, idx) && return nothing
    return io_mats[robot][Int.(idx), Int.(idx)]
end

function normalize_io_mass(M)
    total = sum(filter(isfinite, vec(M)))
    total > eps() || return M .* NaN
    return M ./ total
end

function comparable_io_matrix(io_mats, mech_data, robot)
    haskey(io_mats, robot) || return nothing
    haskey(COMPARABLE_DOF_LABELS, robot) || return nothing
    labels = mech_data[robot].labels
    native = first.(COMPARABLE_DOF_LABELS[robot])
    idx = [findfirst(==(l), labels) for l in native]
    any(isnothing, idx) && return nothing
    M = io_mats[robot][Int.(idx), Int.(idx)]
    return normalize_io_mass(M)
end

comparable_tick_labels(robot) = last.(COMPARABLE_DOF_LABELS[robot])

# Native MuJoCo joint name -> the short ego-centered label used on the panel E
# axes, so panels F/G can name pathways the same way panel E does.
const PRETTY_DOF = Dict{String,String}(
    native => pretty
    for pairs in values(COMPARABLE_DOF_LABELS) for (native, pretty) in pairs
)
pretty_dof(name) = get(PRETTY_DOF, name, name)
pathway_label(input, output) = "$(pretty_dof(input)) \u2192 $(pretty_dof(output))"

# --- figure assembly ---------------------------------------------------------

function make_figure4()
    ensure_dirs!()

    # Load all H1 motion conditions and split train/test windows.
    h1_data = timed_stage("load H1 motion windows") do
        out = Dict{String,Any}()
        for motion in H1_MOTIONS
            X, labels, speed = load_channels("UnitreeH1", motion)
            train, test, _, _, _, test_starts, split = train_test_windows(X, speed)
            out[motion] = (X=X, labels=labels, speed=speed, train=train, test=test,
                test_starts=test_starts, split=split)
        end
        out
    end

    mech_data = timed_stage("load mechanical comparison windows") do
        out = Dict{String,Any}()
        for robot in MECH_ROBOTS
            X, labels, speed = load_channels(robot, "walk")
            train, test, _, _, _, _, _ = train_test_windows(X, speed)
            out[robot] = (X=X, labels=labels, speed=speed, train=train, test=test)
        end
        out
    end

    # Panel B/C: fit walk operator, evaluate reconstruction and horizon curves.
    walk_result = timed_stage("fit H1 walk operator") do
        fit_operator(h1_data["walk"].train)
    end
    walk_test = h1_data["walk"].test

    # Panel D-right + mechanical comparison: bootstrap metric distributions per robot.
    robot_boot = timed_stage("bootstrap mechanical-condition metrics") do
        Dict(r => bootstrap_metrics(vcat(mech_data[r].train, mech_data[r].test))
            for r in MECH_ROBOTS)
    end

    # Panel D-left / G / H: one DMD fit per short window across all H1 motions,
    # each tagged with its own mean forward speed -> continuous speed axis
    # instead of four discrete motion buckets (H1 walk ~1.5 m/s, run ~3.0 m/s;
    # random_walk sweeps ~0-2 m/s within a single recording; stepinplace ~0 m/s).
    # This is the expensive step (~630 DMD fits x2), so it's cached to disk;
    # set FORCE_FIGURE4=true to recompute (e.g. after changing window/hop or
    # the metrics themselves).
    window_cache_path = joinpath(RESULTS_DIR, "figure4_h1_window_metrics.csv")
    if isfile(window_cache_path) && !FORCE_RECOMPUTE
        println("Loading cached per-window H1 metrics: $window_cache_path")
        h1_window_df = timed_stage("load cached per-window H1 metrics") do
            CSV.read(window_cache_path, DataFrame)
        end
    else
        println("Fitting per-window H1 metrics across $(length(H1_MOTIONS)) motions (slow: one DMD fit per window)...")
        h1_window_df = timed_stage("fit per-window H1 metrics") do
            vcat([per_window_metrics("UnitreeH1", m) for m in H1_MOTIONS]...)
        end
        CSV.write(window_cache_path, h1_window_df)
        println("Wrote per-window H1 metrics: $window_cache_path")
    end
    h1_gait_df = h1_window_df[in.(h1_window_df.motion, Ref(GAIT_MOTIONS)), :]

    # Panel C: forecast-horizon curves per motion.
    horizons = collect(0:10:180)
    horizon_rows = NamedTuple[]
    timed_stage("compute forecast-horizon curves") do
        for motion in GAIT_MOTIONS
            d = h1_data[motion]
            result = fit_operator(d.train)
            result === nothing && continue
            # Period comes from the continuous training segment, not from the
            # 240-sample windows: at that length the FFT bin is 0.167 Hz and the
            # estimate collapses to a subharmonic (see `refine_stride_period`).
            train_continuous = @view d.X[1:d.split, :]
            period_f = refine_stride_period(train_continuous .- mean(train_continuous; dims=1); fs=FS)
            template_info = build_phase_template(train_continuous, period_f)
            @printf("[figure 4]   %s stride frequency: %.3f Hz (period %.2f samples)\n",
                motion, FS / period_f, period_f)
            for h in horizons
                nrmses = Float64[]
                template_nrmses = Float64[]
                for tr in d.test
                    h > size(tr, 1) - N_DELAYS && continue
                    pred = future_rollout_prediction_error(result, tr; n_steps=h)
                    push!(nrmses, pred.nrmse)
                    if template_info !== nothing
                        template, period = template_info
                        push!(template_nrmses, template_prediction_error(template, period, tr; n_steps=h))
                    end
                end
                isempty(nrmses) && continue
                push!(horizon_rows, (motion=motion, horizon=h, dmd_nrmse=mean(nrmses),
                    template_nrmse=mean(filter(isfinite, template_nrmses))))
            end
        end
    end
    horizon_df = DataFrame(horizon_rows)

    # Panel E/F: resolvent input-output structure, H1 vs mechanical variants.
    io_freqs = Dict{String,Float64}()
    io_mats = Dict{String,Matrix{Float64}}()
    timed_stage("fit mechanical resolvent I/O maps") do
        for (robot, d) in mech_data
            println("[figure 4]   I/O map: $robot")
            flush(stdout)
            result = fit_operator(d.train)
            result === nothing && continue
            # Autocorrelation, not the windowed FFT: at WINDOW_LEN=240 the FFT
            # bin is 0.167 Hz and `estimate_stride_frequency` collapsed every
            # robot to 0.667 Hz against a true ~0.880 (see refine_stride_period),
            # so these maps were being evaluated at 76% of stride frequency.
            f0 = stride_frequency(d.train; fs=FS)
            isfinite(f0) || continue
            io_freqs[robot] = f0
            # I/O structure from the SAME delay-embedded operator that drives
            # the prediction panels, using the per-joint Hankel-block form that
            # Figures 5 and 6 use for the human maps. This previously read the
            # explicit [angle, angular velocity] state instead, which made the
            # robot maps and the human maps incomparable -- the two halves of
            # the model-to-human bridge were computed in different coordinate
            # systems.
            Mio = hankel_block_resolvent_io_matrix(result, f0; fs=FS)
            Mio === nothing && continue
            io_mats[robot] = Mio
        end
    end

    fig = Figure(size=(1680, 1320), fontsize=PAPER_FONT_SIZE)

    # A: schematic
    axA = Axis(fig[1, 1:2], title="Robot mocap validation setup")
    panel_label!(fig, (1, 1), "A")
    draw_biped_schematic!(axA)

    # B: example kinematics + reconstruction
    axB = Axis(fig[1, 3:4], title="Kinematics vs DMD rollout (H1 walk, held-out)",
        xlabel="time (s)", ylabel="joint angle (rad)")
    panel_label!(fig, (1, 3), "B")
    if walk_result !== nothing && !isempty(walk_test)
        tr = walk_test[1]
        horizon_b = min(160, size(tr, 1) - N_DELAYS - 2)
        pred = rollout_prediction_error(walk_result, tr; n_steps=horizon_b)
        t = (0:horizon_b-1) ./ FS
        labels = h1_data["walk"].labels
        idx_hip = findfirst(==( "hip_flexion_r"), labels)
        idx_knee = findfirst(==( "knee_r"), labels)
        vspan!(axB, t[1], t[min(N_DELAYS, length(t))]; color=(:gray70, 0.28))
        lines!(axB, t, pred.y[:, idx_hip]; color=BLUE, linewidth=2, label="hip flex R (data)")
        lines!(axB, t, pred.yhat[:, idx_hip]; color=BLUE, linestyle=:dash, linewidth=2, label="hip flex R (DMD)")
        lines!(axB, t, pred.y[:, idx_knee]; color=RED, linewidth=2, label="knee R (data)")
        lines!(axB, t, pred.yhat[:, idx_knee]; color=RED, linestyle=:dash, linewidth=2, label="knee R (DMD)")
    end
    axislegend(axB; position=:rb, framevisible=false)

    # C: forecast horizon. Solid = DMD rollout on held-out test windows.
    # Dotted = a phase-averaged template built only from training windows
    # (period-length cycle, averaged across training data, then wrapped onto
    # the test trial) -- it has no access to the test trial's own future, so
    # unlike a naive "wrap the test trial onto itself" baseline it cannot
    # trivially reproduce short-horizon values. It is also not phase-locked to
    # any given test trial (windows are arbitrary slices of a continuous
    # recording, not heel-strike aligned), so its error reflects a fixed,
    # trial-dependent phase offset rather than growing smoothly with horizon.
    axC = Axis(fig[1, 5:6], title="Forecast accuracy vs horizon",
        xlabel="horizon (s)", ylabel="normalized RMSE")
    panel_label!(fig, (1, 5), "C")
    for motion in GAIT_MOTIONS
        sub = horizon_df[horizon_df.motion .== motion, :]
        isempty(sub) && continue
        lines!(axC, sub.horizon ./ FS, sub.dmd_nrmse; color=MOTION_COLORS[motion], linewidth=2.4,
            label=MOTION_LABELS[motion])
        lines!(axC, sub.horizon ./ FS, sub.template_nrmse; color=MOTION_COLORS[motion],
            linewidth=1.4, linestyle=:dot)
    end
    text!(axC, 0.98, 0.04; text="solid = DMD rollout\ndotted = training-derived phase template",
        space=:relative, align=(:right, :bottom), fontsize=10, color=(:gray20, 0.9))
    axislegend(axC; position=:lt, framevisible=false)

    # D: retained mechanical-condition comparison (the speed and redundant
    # eigenvalue/participation panels were removed for a tighter main figure).
    glD = GridLayout(fig[2, 1:3])
    panel_label!(fig, (2, 1), "D")
    axD2 = Axis(glD[1, 1:3], title="Transient gain across mechanical condition (walk)",
        xticklabelrotation=0.5)
    robot_xs = 1:length(MECH_ROBOTS)
    for (i, robot) in enumerate(MECH_ROBOTS)
        vals = filter(isfinite, robot_boot[robot].log_tgmax)
        isempty(vals) && continue
        q = quantile(vals, [0.25, 0.5, 0.75])
        lines!(axD2, [i, i], [q[1], q[3]]; color=ROBOT_COLORS[robot], linewidth=6)
        scatter!(axD2, [i], [q[2]]; color=ROBOT_COLORS[robot], markersize=13)
        scatter!(axD2, fill(i, length(vals)) .+ 0.12 .* randn(MersenneTwister(100 + i), length(vals)), vals;
            color=(ROBOT_COLORS[robot], 0.25), markersize=5)
    end
    axD2.xticks = (robot_xs, MECH_ROBOTS)
    axD2.ylabel = L"\log_{10}\max_k \|A^k\|_2"
    colgap!(glD, 12)

    # E: resolvent input-output heatmaps across joints
    glE = GridLayout(fig[2, 4:6])
    panel_label!(fig, (2, 4), "E")
    io_f0s = filter(isfinite, [get(io_freqs, r, NaN) for r in MECH_ROBOTS])
    io_f0_label = isempty(io_f0s) ? "" : @sprintf(" (%.2f Hz)", mean(io_f0s))
    Label(glE[1, 1:3], "Joint resolvent I/O gain at stride frequency" * io_f0_label;
        fontsize=17, font=:bold)
    io_plots = [
        ("UnitreeH1", "H1"),
        ("UnitreeG1", "G1 (mapped)"),
        ("Atlas", "Atlas (shared)"),
    ]
    comparable_mats = Dict{String,Matrix{Float64}}()
    for (robot, _) in io_plots
        M = comparable_io_matrix(io_mats, mech_data, robot)
        M === nothing || (comparable_mats[robot] = M)
    end
    io_mass_hi = isempty(comparable_mats) ? 1.0 :
        maximum(filter(isfinite, reduce(vcat, [vec(M) for M in values(comparable_mats)])))
    shared_labels = haskey(mech_data, "UnitreeH1") && haskey(mech_data, "Atlas") ?
        filter(l -> l in mech_data["Atlas"].labels, mech_data["UnitreeH1"].labels) :
        String[]
    # These I/O maps are rank one -- >= 99.9% of the energy sits in the first
    # singular value, against ~0.57-0.62 for a shuffled null of the same gains
    # (scripts/S57_io_to_vectors.jl). A 10x10 heatmap therefore shows 100 cells
    # that are the outer product of a RECEPTION weight and a DRIVE weight.
    # Unlike the human maps, the robots' drive vector is NOT flat -- hip
    # flexion drives at 0.18-0.24 against 0.04-0.05 for knee -- so both vectors
    # are reported here.
    vec_hi = 0.0
    robot_vecs = Dict{String,Any}()
    for (robot, _) in io_plots
        haskey(comparable_mats, robot) || continue
        F = svd(comparable_mats[robot]); sv = svdvals(comparable_mats[robot])
        u = abs.(F.U[:, 1]); v = abs.(F.V[:, 1])
        robot_vecs[robot] = (sink=u ./ sum(u), source=v ./ sum(v),
            e1=sv[1]^2 / sum(abs2, sv))
        vec_hi = max(vec_hi, maximum(u ./ sum(u)), maximum(v ./ sum(v)))
    end
    vec_hi *= 1.22
    for (icol, (robot, title)) in enumerate(io_plots)
        haskey(robot_vecs, robot) || continue
        plot_labels = comparable_tick_labels(robot)
        n = length(plot_labels)
        axE = Axis(glE[2, icol], title=title, xticklabelrotation=1.0,
            xticks=(1:n, plot_labels), xlabel="DOF",
            ylabel=(icol == 1 ? "weight" : ""))
        # Same convention as Figures 5-6: reception dark, drive light, one colour.
        barplot!(axE, (1:n) .- 0.20, robot_vecs[robot].sink[1:n];
            color=ROBOT_COLORS[robot], width=0.38, label="reception")
        barplot!(axE, (1:n) .+ 0.20, robot_vecs[robot].source[1:n];
            color=(ROBOT_COLORS[robot], 0.35), width=0.38, label="drive")
        ylims!(axE, 0, vec_hi)
        icol == 1 && axislegend(axE; position=:lt, framevisible=false, labelsize=12)
        icol > 1 && hideydecorations!(axE; grid=false)
        text!(axE, 0.98, 0.96; space=:relative, align=(:right, :top), fontsize=11,
            text=@sprintf("rank one: %.1f%%", 100 * robot_vecs[robot].e1))
    end
    rowgap!(glE, 4)

    # F/G: H1 vs Atlas on their shared joints. The shared 10x10 map is rank one
    # (see above), so the comparison is between the two robots' reception and
    # drive profiles -- 10 + 10 numbers -- not 100 independent pathways.
    axF = Axis(fig[3, 1:3], title="Shared joints: H1 vs Atlas profiles",
        xlabel="H1 weight", ylabel="Atlas weight")
    panel_label!(fig, (3, 1), "F")
    axG = Axis(fig[3, 4:6], title="Profile differences (Atlas − H1)",
        xlabel="weight difference", ylabel="",
        subtitle="red: Atlas higher · blue: H1 higher · dark: reception · light: drive")
    panel_label!(fig, (3, 4), "G")
    corr_rows = NamedTuple[]
    if !isempty(shared_labels)
        Mh1 = shared_io_matrix(io_mats, mech_data, "UnitreeH1", shared_labels)
        Matlas = shared_io_matrix(io_mats, mech_data, "Atlas", shared_labels)
        if Mh1 !== nothing && Matlas !== nothing
            function rank_one_profiles(M)
                F = svd(M)
                u = abs.(F.U[:, 1]); v = abs.(F.V[:, 1])
                return u ./ sum(u), v ./ sum(v)
            end
            rec_h, drv_h = rank_one_profiles(Mh1)
            rec_a, drv_a = rank_one_profiles(Matlas)
            r_rec = robust_cor(rec_h, rec_a)
            r_drv = robust_cor(drv_h, drv_a)
            names = [get(PRETTY_DOF, l, l) for l in shared_labels]
            n = length(names)

            # --- F: every joint's weight in each robot, against the identity line.
            allv = vcat(rec_h, rec_a, drv_h, drv_a)
            hi = maximum(allv) * 1.08
            lines!(axF, [0, hi], [0, hi]; color=(:black, 0.45), linestyle=:dash, linewidth=1.5)
            scatter!(axF, rec_h, rec_a; color=GRAY, markersize=13, label="reception")
            scatter!(axF, drv_h, drv_a; color=:white, strokecolor=GRAY, strokewidth=1.8,
                marker=:rect, markersize=12, label="drive")
            limits!(axF, 0, hi, 0, hi)
            text!(axF, 0.04, 0.97;
                text=@sprintf("reception r = %.2f\ndrive r = %.2f  (n = %d joints)", r_rec, r_drv, n),
                space=:relative, align=(:left, :top), fontsize=13, font=:bold)
            axislegend(axF; position=:rb, framevisible=false, labelsize=12)

            # --- G: signed differences per joint, reception dark and drive light.
            d_rec = rec_a .- rec_h
            d_drv = drv_a .- drv_h
            ys = 1:n
            vlines!(axG, [0.0]; color=(:black, 0.35), linewidth=1)
            barplot!(axG, ys .- 0.19, d_rec; direction=:x, width=0.36,
                color=[d >= 0 ? RED : BLUE for d in d_rec], label="reception")
            barplot!(axG, ys .+ 0.19, d_drv; direction=:x, width=0.36,
                color=[(d >= 0 ? RED : BLUE, 0.35) for d in d_drv], label="drive")
            axG.yticks = (ys, names)
            axG.yticklabelsize = 11
            axG.yreversed = true
            xmax = maximum(abs, vcat(d_rec, d_drv))
            xlims!(axG, -1.18xmax, 1.18xmax)
            for j in 1:n
                push!(corr_rows, (joint=shared_labels[j],
                    reception_h1=rec_h[j], reception_atlas=rec_a[j],
                    drive_h1=drv_h[j], drive_atlas=drv_a[j],
                    reception_r=r_rec, drive_r=r_drv))
            end
        end
    end

    colgap!(fig.layout, 34)
    rowgap!(fig.layout, 30)

    metrics_df = vcat(
        insertcols!(copy(h1_window_df), :condition_type => "speed_window"),
        [insertcols!(copy(df), :condition_type => "mechanical", :condition => r, :speed => NaN,
            :motion => "walk", :robot => r)
         for (r, df) in robot_boot]...;
        cols=:union,
    )
    corr_df = (@isdefined(corr_rows) && !isempty(corr_rows)) ? DataFrame(corr_rows) : DataFrame()
    return fig, metrics_df, horizon_df, corr_df
end

if abspath(PROGRAM_FILE) == @__FILE__
    fig, metrics_df, horizon_df, corr_df = make_figure4()
    timed_stage("write Figure 4.svg") do
        savefig("Figure 4.svg", fig)
    end
    CSV.write(joinpath(RESULTS_DIR, "figure4_bootstrap_metrics.csv"), metrics_df)
    CSV.write(joinpath(RESULTS_DIR, "figure4_horizon_curves.csv"), horizon_df)
    if nrow(corr_df) > 0
        CSV.write(joinpath(RESULTS_DIR, "figure4_h1_atlas_shared_io_pathways.csv"), corr_df)
    end
    println("Figure 4 bootstrap metric rows: ", nrow(metrics_df))
end

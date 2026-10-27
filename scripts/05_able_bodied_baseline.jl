#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "src", "GaitNonNormalityPaper3.jl"))

using .GaitNonNormalityPaper3
using CairoMakie
using CSV
using DataFrames
using DMDAnalysis
using LaTeXStrings
using LinearAlgebra
using NPZ
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

const FS = 100.0
const N_DELAYS = parse(Int, get(ENV, "FIGURE5_N_DELAYS", "32"))
const STEPS_PER_STRIDE = 100
const TRAIN_ROWS = 1000
const TEST_HORIZON = 220
const FORCE_RECOMPUTE = lowercase(get(ENV, "FORCE_FIGURE5", "false")) in ("1", "true", "yes")
const FORCE_RESOLVENT = FORCE_RECOMPUTE || lowercase(get(ENV, "FORCE_FIGURE5_RESOLVENT", "false")) in ("1", "true", "yes")
const NOMINAL_STRIDE_FREQUENCY = FS / STEPS_PER_STRIDE
const PROFILE_RESOLVENT_ANGLES = 1000
const PANEL_METRIC_RANK = parse(Int, get(ENV, "FIGURE5_PANEL_METRIC_RANK", "70"))

# Single-source figure: able-bodied trials from the Healthy Young Adult study
# only. The superset's other able-bodied trials (the PostStroke study's own
# controls) were preprocessed differently and are not comparable at this rank --
# see the SOURCE_* docstring in src/GaitNonNormalityPaper3.jl. They are used
# instead as the control group of Figure 6, alongside the stroke trials that
# share their pipeline.
const SOURCE = get(ENV, "FIGURE5_SOURCE", SOURCE_HYA)
const SOURCE_TAG = get(SOURCE_TAGS, SOURCE, "src")

# The source slug is part of the cache tag so a restricted run cannot silently
# read a cache built from the pooled superset.
const DMD_CONFIG_TAG = "d$(N_DELAYS)_fixedrank$(PANEL_METRIC_RANK)_$(SOURCE_TAG)$(PHASE_TAG)"

# Able-bodied only: channels are in the source export's native order, taken
# as right-then-left (see the limb convention in GaitNonNormalityPaper3.jl).
const JOINT_LABELS = JOINT_LABELS_AB

function trial_stride_frequency(trial; fs=FS)
    UNIFORM_STRIDE && return fs / PHASE_STEPS        # exactly one stride per PHASE_STEPS samples
    f0 = estimate_stride_frequency(center_trial(trial); fs=fs, fmin=0.25, fmax=5.0)
    return isfinite(f0) && f0 > 0 ? f0 : NOMINAL_STRIDE_FREQUENCY
end

function fit_trial_operator(trial; n_delays=N_DELAYS, rank_rule=FixedRank(PANEL_METRIC_RANK), stabilize=true)
    h = HankelGenerator(n_delays=n_delays, delay_interval=1)
    model = DMD(rank=rank_rule, stabilize=stabilize)
    return fit_gait_operator(model, center_trial(trial), h)
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

function resolvent_io_matrix(result, freq; fs=FS)
    A = result.A
    r = size(A, 1)
    n = result.n_features
    z = exp(2im * pi * freq / fs)
    R = inv(z * I(r) - A)
    C = result.U[1:n, :]
    B = result.U[1:n, :]'
    return abs.(C * R * B)
end

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

# Stride period at sub-sample resolution, from a multi-joint autocorrelation peak.
#
# Ported from scripts/04_bipedal_joint_structure.jl, where an integer-period
# phase template was found to be scoring NRMSE > 1 -- worse than predicting the
# mean -- and the cause was traced to two compounding errors. This function
# addresses the first: `estimate_stride_frequency` reads a peak off an FFT whose
# bin spacing is coarse enough to admit a subharmonic, and on these 15 s trials
# it returns a period off by ~3 samples typically and by a full octave for at
# least one subject (AB09: 59 samples against a true 117).
#
# Octave protection matters in BOTH directions: the FFT can accept f_peak/2,
# while a raw autocorrelation argmax prefers a MULTIPLE of the stride, since a
# T-periodic signal peaks nearly as high at 2T. Taking the SMALLEST local
# maximum that reaches `peak_tol` of the global peak selects the fundamental.
# Half-stride lags do not compete because left and right joints are antiphase
# there, so the across-joint mean correlation goes negative.
function refine_stride_period(Xc::AbstractMatrix; fs=FS, fmin=0.2, fmax=3.0, peak_tol=0.85)
    n_t = size(Xc, 1)
    n_t > 8 || return NaN
    Xz = Matrix{Float64}(Xc) .- mean(Xc; dims=1)
    ac = zeros(Float64, n_t)
    for j in axes(Xz, 2)
        v = @view Xz[:, j]
        denom = sum(abs2, v)
        denom > eps() || continue
        for lag in 0:n_t-1
            ac[lag+1] += sum(v[1:n_t-lag] .* v[lag+1:n_t]) / denom
        end
    end
    ac ./= size(Xz, 2)
    lo = max(2, floor(Int, fs / fmax))
    hi = min(n_t - 2, ceil(Int, fs / fmin))
    hi > lo + 1 || return NaN
    seg = @view ac[lo+1:hi+1]
    gpk = maximum(seg)
    gpk > 0 || return NaN
    for i in 2:length(seg)-1
        if seg[i] >= seg[i-1] && seg[i] >= seg[i+1] && seg[i] >= peak_tol * gpk
            k = lo + i - 1
            y0, y1, y2 = ac[k], ac[k+1], ac[k+2]
            den = y0 - 2y1 + y2
            return den == 0 ? Float64(k) : k + 0.5 * (y0 - y2) / den
        end
    end
    return Float64(lo + argmax(seg) - 1)
end

# Phase-average template with FRACTIONAL-phase folding.
#
# This addresses the second half of the Figure 4 bug. Folding at `round(period)`
# drifts by the rounding error every cycle, so over the folded history the
# template smears toward a flat line -- measured here at 0.745 of the data's own
# amplitude, which `template_affine` then scales back up, converting an underfit
# into an amplified wrong-phase prediction. Splitting each sample across its two
# neighbouring phase bins by its true fractional phase means rounding cannot
# accumulate; template amplitude recovers to 0.993 of the data.
function build_phase_template(train::AbstractMatrix, period_f; n_cycles=10)
    (isfinite(period_f) && period_f >= 8) || return nothing
    n_bins = clamp(round(Int, period_f), 8, 10_000)
    n_total, n_features = size(train)
    span = min(n_total, ceil(Int, n_cycles * period_f))
    row0 = n_total - span + 1
    acc = zeros(Float64, n_bins, n_features)
    counts = zeros(Float64, n_bins)
    for i in row0:n_total
        ph = mod(i / period_f, 1.0) * n_bins
        b0 = floor(Int, ph)
        w = ph - b0
        i0 = mod(b0, n_bins) + 1
        i1 = mod(b0 + 1, n_bins) + 1
        @views acc[i0, :] .+= (1 - w) .* train[i, :]
        @views acc[i1, :] .+= w .* train[i, :]
        counts[i0] += 1 - w
        counts[i1] += w
    end
    any(<(1e-9), counts) && return nothing
    return acc ./ counts
end

function template_prediction(template, n_steps; offset=0, start=0, scale=nothing, delta=nothing)
    period = size(template, 1)
    yhat = template[[mod1(offset + start + i, period) for i in 1:n_steps], :]
    scale === nothing || (yhat .*= scale')
    delta === nothing || (yhat .+= delta')
    return yhat
end

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

function best_template_offset(template, init; affine_correct=true)
    period = size(template, 1)
    n_init = min(size(init, 1), period)
    best, best_err = 0, Inf
    best_scale = ones(size(init, 2))
    best_delta = zeros(size(init, 2))
    for off in 0:period-1
        pred_init = template[[mod1(off + i, period) for i in 1:n_init], :]
        scale, delta = affine_correct ? template_affine(pred_init, init[1:n_init, :]) :
            (ones(size(init, 2)), zeros(size(init, 2)))
        err = sum(abs2, pred_init .* scale' .+ delta' .- init[1:n_init, :])
        if err < best_err
            best, best_err = off, err
            best_scale = scale
            best_delta = delta
        end
    end
    return best, best_scale, best_delta
end

function prediction_curves(result, test, template; horizons=collect(0:10:TEST_HORIZON))
    rows = NamedTuple[]
    template === nothing && return DataFrame(rows)
    for h in horizons
        if h == 0
            push!(rows, (horizon=h, dmd_nrmse=0.0, phase_nrmse=0.0))
            continue
        end
        h > size(test, 1) - N_DELAYS && continue
        pred = future_rollout_prediction_error(result, test; n_steps=h)
        off, scale, delta = best_template_offset(template, @view test[1:N_DELAYS, :])
        ybase = template_prediction(template, h; offset=off, start=N_DELAYS, scale=scale, delta=delta)
        y = test[N_DELAYS+1:N_DELAYS+h, :]
        base_rmse = sqrt(mean(abs2, ybase .- y))
        base_nrmse = base_rmse / max(sqrt(mean(abs2, y)), eps())
        push!(rows, (horizon=h, dmd_nrmse=pred.nrmse, phase_nrmse=base_nrmse))
    end
    return DataFrame(rows)
end

# Transient-growth horizon in operator steps spanning exactly ONE STRIDE.
# The operator advances one sample per step, so this is FS/f0. Reported
# alongside the metric so the horizon is auditable per trial.
function one_cycle_steps(f0; fallback::Int)
    isfinite(f0) && f0 > 0 || return fallback
    return clamp(round(Int, FS / f0), 8, 4000)
end

function metric_row(trial, meta_row; n_delays=N_DELAYS, rank_rule=FixedRank(PANEL_METRIC_RANK), stabilize=true)
    result = fit_trial_operator(trial; n_delays=n_delays, rank_rule=rank_rule, stabilize=stabilize)
    f0 = trial_stride_frequency(trial)
    harms = harmonic_gains(result, f0; n_harmonics=3, fs=FS)
    peak = resolvent_peak(result; n_angles=360, fs=FS, fmin=0.25)
    return (
        trial_index=Int(meta_row.trial_index),
        source=String(meta_row.source),
        subject=String(meta_row.subject),
        speed=Float64(meta_row.speed),
        group=String(meta_row.lf_or_hf),
        rank=result.rank,
        n_delays=n_delays,
        stride_frequency=f0,
        henrici=henrici_departure(result),
        log_tgmax=log10(transient_growth(result; kmax=one_cycle_steps(f0; fallback=3n_delays))),
        tg_horizon_steps=one_cycle_steps(f0; fallback=3n_delays),
        participation=effective_rank(result.singular_values),
        resolvent_peak_gain=peak.gain,
        resolvent_peak_frequency=peak.frequency,
        q_factor=q_factor(result, f0; fs=FS, step=0.01),
        log_q_factor=log10(max(q_factor(result, f0; fs=FS, step=0.01), eps())),
        gain2_over_gain1=length(harms) >= 2 ? harms[2] / max(harms[1], eps()) : NaN,
    )
end

function compute_or_load_metrics(ab_trials, ab_meta)
    path = joinpath(RESULTS_DIR, "figure5_ab_metrics_$(DMD_CONFIG_TAG).csv")
    if isfile(path) && !FORCE_RECOMPUTE
        println("Loading cached AB metrics: $path")
        return CSV.read(path, DataFrame)
    end
    rows = NamedTuple[]
    println("Fitting AB trial metrics ($(length(ab_trials)) trials)...")
    for i in eachindex(ab_trials)
        try
            push!(rows, metric_row(ab_trials[i], ab_meta[i, :]))
            print(".")
        catch err
            @warn "Failed AB metric fit" i err
        end
    end
    println()
    df = DataFrame(rows)
    CSV.write(path, df)
    println("Wrote AB metrics: $path")
    return df
end

function compute_or_load_prediction(ab_trials, ab_meta)
    path = joinpath(RESULTS_DIR, "figure5_prediction_curves_$(DMD_CONFIG_TAG).csv")
    if isfile(path) && !FORCE_RECOMPUTE
        println("Loading cached prediction curves: $path")
        return CSV.read(path, DataFrame)
    end
    rows = NamedTuple[]
    rng = MersenneTwister(505)
    subjects = unique(String.(ab_meta.subject))
    for subject in subjects
        idxs = findall(String.(ab_meta.subject) .== subject)
        isempty(idxs) && continue
        i = idxs[argmin(abs.(Float64.(ab_meta.speed[idxs]) .- median(Float64.(ab_meta.speed[idxs]))))]
        tr = center_trial(ab_trials[i])
        train = tr[1:min(TRAIN_ROWS, size(tr, 1)-300), :]
        test = tr[size(train, 1)+1:end, :]
        size(test, 1) <= TEST_HORIZON + N_DELAYS && continue
        result = fit_trial_operator(train)
        # Sub-sample stride period from the autocorrelation, not `round(FS/f0)`:
        # the FFT estimate is off by ~3 samples typically and by an octave for at
        # least one subject, and integer folding then accumulates that error over
        # every folded cycle. See `refine_stride_period` / `build_phase_template`.
        period_f = UNIFORM_STRIDE ? Float64(PHASE_STEPS) : refine_stride_period(train .- mean(train; dims=1))
        isfinite(period_f) || (period_f = FS / trial_stride_frequency(train))
        template = build_phase_template(train, period_f)
        df = prediction_curves(result, test, template)
        for row in eachrow(df)
            push!(rows, (
                subject=subject,
                speed=Float64(ab_meta.speed[i]),
                horizon=row.horizon,
                dmd_nrmse=row.dmd_nrmse,
                phase_nrmse=row.phase_nrmse,
            ))
        end
    end
    df = DataFrame(rows)
    CSV.write(path, df)
    println("Wrote prediction curves: $path")
    return df
end

function compute_or_load_robustness(ab_trials, ab_meta)
    path = joinpath(RESULTS_DIR, "figure5_robustness_$(SOURCE_TAG).csv")
    if isfile(path) && !FORCE_RECOMPUTE
        println("Loading cached robustness table: $path")
        return CSV.read(path, DataFrame)
    end
    rng = MersenneTwister(55)
    idxs = shuffle(rng, eachindex(ab_trials))[1:min(36, length(ab_trials))]
    rows = NamedTuple[]
    configs = [
        (label="d12 low rank", n_delays=12, rank=FixedRank(8)),
        (label="d12 energy", n_delays=12, rank=EnergyRank(1e-6)),
        (label="d26 energy", n_delays=26, rank=EnergyRank(1e-6)),
        (label="d40 energy", n_delays=40, rank=EnergyRank(1e-6)),
    ]
    println("Fitting Figure 5 robustness subset...")
    for cfg in configs, i in idxs
        try
            row = metric_row(ab_trials[i], ab_meta[i, :]; n_delays=cfg.n_delays, rank_rule=cfg.rank)
            push!(rows, merge((config=cfg.label,), row))
        catch err
            @warn "Failed robustness fit" cfg i err
        end
    end
    df = DataFrame(rows)
    CSV.write(path, df)
    println("Wrote robustness table: $path")
    return df
end

function speed_bin_indices(speeds)
    qs = quantile(Float64.(speeds), [1/3, 2/3])
    return [
        ("slow", findall(speeds .<= qs[1])),
        ("middle", findall((speeds .> qs[1]) .& (speeds .<= qs[2]))),
        ("fast", findall(speeds .> qs[2])),
    ]
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

function compute_or_load_resolvent_averages(ab_trials, ab_meta)
    profile_path = joinpath(RESULTS_DIR, "figure5_resolvent_profiles_$(DMD_CONFIG_TAG).csv")
    io_path = joinpath(RESULTS_DIR, "figure5_resolvent_io_by_speed_$(DMD_CONFIG_TAG).csv")
    if isfile(profile_path) && isfile(io_path) && !FORCE_RESOLVENT
        cached_io = CSV.read(io_path, DataFrame)
        if "all" in unique(String.(cached_io.speed_bin))
            println("Loading cached trial-averaged resolvent summaries")
            return CSV.read(profile_path, DataFrame), cached_io
        end
        println("Ignoring cached resolvent summaries missing the all-trials aggregate")
    end

    speeds = Float64.(ab_meta.speed)
    # Fit every trial exactly once, then slice the per-trial arrays for both
    # the speed-bin summaries and an "all" aggregate across every trial --
    # avoids refitting trials that belong to more than one grouping.
    bins = vcat(speed_bin_indices(speeds), [("all", 1:length(ab_trials))])
    # Same grid as Figure 6B: step 0.005 strides, landing exactly on every
    # harmonic. Gains are evaluated AT the grid (not interpolated from a coarse
    # full-circle scan, which on the phase grid sampled only every 0.1 stride and
    # turned each sharp harmonic peak into a triangular hump).
    grid = collect(range(0.5, 4.5; length=801))
    n = length(ab_trials)
    gains_all = fill(NaN, n, length(grid))
    io_stack_all = fill(NaN, n, 6, 6)
    println("Fitting trial-level resolvent summaries for Figure 5...")
    for i in 1:n
        try
            result = fit_trial_operator(ab_trials[i])
            f0 = trial_stride_frequency(ab_trials[i])
            gains_all[i, :] .= resolvent_gain_at(result, grid .* f0; fs=FS)
            # I/O structure read from the DELAY-EMBEDDED operator (opnorm over
            # each joint's Hankel block), matching Figure 6 D/E/F. The explicit
            # [angle, velocity] state space was tried here and is less
            # scale-sensitive, but it does not carry the clinical signatures the
            # stroke figure reports, so both figures use the same construction.
            M = hankel_block_resolvent_io_matrix(result, f0; fs=FS)
            M === nothing && continue
            # Normalize by TOTAL I/O gain, not by the single max pixel: the max
            # is one cell and moves with whichever pathway happens to peak, so
            # max-normalized maps are not comparable across trials. Dividing by
            # the sum makes each map a distribution of gain over pathways.
            io_stack_all[i, :, :] .= M ./ max(sum(M), eps())
            print(".")
        catch err
            @warn "Failed trial-level resolvent summary" i err
        end
    end
    println()

    profile_rows = NamedTuple[]
    io_rows = NamedTuple[]
    for (name, idxs) in bins
        idxs = collect(idxs)
        gains = gains_all[idxs, :]
        io_stack = io_stack_all[idxs, :, :]
        meanv, lo, hi = profile_stats(gains)
        for j in eachindex(grid)
            push!(profile_rows, (
                speed_bin=name,
                frequency_ratio=grid[j],
                mean_gain=meanv[j],
                lo=lo[j],
                hi=hi[j],
                n_trials=length(idxs),
            ))
        end
        for out in 1:6, input in 1:6
            vals = io_stack[:, out, input]
            vals = vals[isfinite.(vals)]
            push!(io_rows, (
                speed_bin=name,
                output_joint=JOINT_LABELS[out],
                input_joint=JOINT_LABELS[input],
                output_index=out,
                input_index=input,
                mean_gain=isempty(vals) ? NaN : mean(vals),
                sem_gain=length(vals) > 1 ? std(vals) / sqrt(length(vals)) : NaN,
                n_trials=length(vals),
            ))
        end
    end
    profile_df = DataFrame(profile_rows)
    io_df = DataFrame(io_rows)
    CSV.write(profile_path, profile_df)
    CSV.write(io_path, io_df)
    println("Wrote trial-averaged resolvent profiles: $profile_path")
    println("Wrote trial-averaged resolvent I/O: $io_path")
    return profile_df, io_df
end

function mean_sem_by_horizon(df)
    rows = NamedTuple[]
    for sub in groupby(df, :horizon)
        push!(rows, (
            horizon=first(sub.horizon),
            dmd_mean=mean(sub.dmd_nrmse),
            dmd_sem=std(sub.dmd_nrmse) / sqrt(nrow(sub)),
            phase_mean=mean(sub.phase_nrmse),
            phase_sem=std(sub.phase_nrmse) / sqrt(nrow(sub)),
        ))
    end
    return sort(DataFrame(rows), :horizon)
end

function scatter_metric!(ax, df, col)
    subjects = sort(unique(String.(df.subject)))
    palette = cgrad(:tab20, max(length(subjects), 3); categorical=true)
    subject_colors = Dict(subject => palette[i] for (i, subject) in enumerate(subjects))
    for sub in groupby(df, :subject)
        ord = sortperm(sub.speed)
        color = subject_colors[String(first(sub.subject))]
        lines!(ax, sub.speed[ord], sub[ord, col]; color=(color, 0.38), linewidth=1.4)
        scatter!(ax, sub.speed, sub[!, col]; color=(color, 0.80), markersize=9,
            strokecolor=color, strokewidth=1.0)
    end
end

# Trial points in subject colours with each subject's mixed-model line (fixed
# effect + that subject's random intercept and slope), and the population line
# with its 95% band in black. Same model as Figure 6, without a group term.
function mixed_metric!(ax, df, col)
    res = speed_mixed_model(df, col; grouped=false)
    pop, subj = mixed_model_lines(res)
    subjects = sort(unique(String.(df.subject)))
    palette = cgrad(:tab20, max(length(subjects), 3); categorical=true)
    subject_colors = Dict(subject => palette[i] for (i, subject) in enumerate(subjects))
    for sub in groupby(df, :subject)
        color = subject_colors[String(first(sub.subject))]
        scatter!(ax, sub.speed, sub[!, col]; color=(color, 0.55), markersize=8,
            strokecolor=color, strokewidth=0.8)
    end
    for s in groupby(subj, :subject)
        lines!(ax, s.speed, s.fit; color=(subject_colors[first(s.subject)], 0.9), linewidth=1.6)
    end
    band!(ax, pop.speed, pop.lo, pop.hi; color=(:black, 0.15))
    lines!(ax, pop.speed, pop.fit; color=:black, linewidth=3.2)
    text!(ax, 0.97, 0.96; space=:relative, align=(:right, :top), fontsize=13,
        text=@sprintf("speed: %.0f%% of variance\np = %s", 100res.share_speed, pstr(res.p_speed)))
    return res
end

pstr(p) = p < 1e-3 ? @sprintf("%.0e", p) : @sprintf("%.2g", p)

function metric_slope_by_subject(df, col)
    rows = NamedTuple[]
    for sub in groupby(df, :subject)
        nrow(sub) < 3 && continue
        x = Float64.(sub.speed)
        y = Float64.(sub[!, col])
        ok = isfinite.(x) .& isfinite.(y)
        count(ok) < 3 && continue
        xs, ys = x[ok], y[ok]
        b = sum((xs .- mean(xs)) .* (ys .- mean(ys))) / max(sum(abs2, xs .- mean(xs)), eps())
        push!(rows, (subject=first(sub.subject), metric=String(col), slope=b))
    end
    return DataFrame(rows)
end

function speed_pathway_summary(resolvent_io; n_each=3)
    wide = unstack(resolvent_io, [:output_joint, :input_joint, :output_index, :input_index], :speed_bin, :mean_gain)
    rows = NamedTuple[]
    for row in eachrow(wide)
        all(k -> k in propertynames(wide), [:slow, :middle, :fast]) || continue
        slow = Float64(row.slow)
        fast = Float64(row.fast)
        isfinite(slow) && isfinite(fast) || continue
        push!(rows, (
            output_joint=String(row.output_joint),
            input_joint=String(row.input_joint),
            output_index=Int(row.output_index),
            input_index=Int(row.input_index),
            pathway_type=row.output_index == row.input_index ? "self" : "cross",
            slow=slow,
            middle=Float64(row.middle),
            fast=fast,
            delta_fast_slow=fast - slow,
            abs_delta=abs(fast - slow),
        ))
    end
    df = DataFrame(rows)
    isempty(df) && return df
    up = df[df.delta_fast_slow .> 0, :]
    down = df[df.delta_fast_slow .< 0, :]
    up = isempty(up) ? up : up[sortperm(up.delta_fast_slow; rev=true)[1:min(n_each, nrow(up))], :]
    down = isempty(down) ? down : down[sortperm(down.delta_fast_slow)[1:min(n_each, nrow(down))], :]
    out = vcat(insertcols!(up, :direction => "up"), insertcols!(down, :direction => "down"); cols=:union)
    return out
end

function sink_source_by_speed(resolvent_io)
    bins = ["slow", "middle", "fast"]
    rows = NamedTuple[]
    for bin in bins
        sub = resolvent_io[resolvent_io.speed_bin .== bin, :]
        nrow(sub) == 0 && continue
        n = maximum(sub.output_index)
        M = fill(NaN, n, n)
        for r in eachrow(sub); M[Int(r.output_index), Int(r.input_index)] = Float64(r.mean_gain); end
        any(isnan, M) && continue
        F = svd(M)
        u = abs.(F.U[:, 1]); v = abs.(F.V[:, 1])
        sv = svdvals(M)
        for j in 1:n
            lbl = first(sub[sub.output_index .== j, :output_joint])
            push!(rows, (speed_bin=bin, index=j, joint=String(lbl),
                sink=u[j]/sum(u), source=v[j]/sum(v),
                energy_first=sv[1]^2/sum(abs2, sv)))
        end
    end
    return DataFrame(rows)
end

# The 6x6 resolvent I/O matrix is rank one (>=99.9% of its energy in the first
# singular value at every speed bin), so the individual "joint j -> joint i"
# pathways it contains are not independent: each is the product of a SINK
# weight for the receiving joint and a SOURCE weight for the driving one.
# Plotting six pathways implied six findings where there are at most eleven
# numbers. This panel shows the two vectors themselves. The source vector is
# flat in human walking, so the reception profile carries the speed effect:
# the knee dominates reception at slow speeds and the hip overtakes it at fast.
function plot_pathway_speed_summary!(ax, resolvent_io; n_each=3)
    summary = sink_source_by_speed(resolvent_io)
    if nrow(summary) == 0
        text!(ax, 0, 0; text="resolvent sink/source summary unavailable", align=(:center, :center))
        return summary
    end
    bins = ["slow", "middle", "fast"]
    xs = 1:3
    palette = [BLUE, RED, GREEN, GOLD, PURPLE, RGBf(0.35, 0.35, 0.35)]
    joints = unique(summary.joint)
    for (k, jt) in enumerate(joints)
        color = palette[mod1(k, length(palette))]
        sink = [only(summary[(summary.speed_bin .== b) .& (summary.joint .== jt), :sink]) for b in bins]
        src  = [only(summary[(summary.speed_bin .== b) .& (summary.joint .== jt), :source]) for b in bins]
        lines!(ax, xs, sink; color=color, linewidth=2.6, label=jt)
        scatter!(ax, xs, sink; color=color, markersize=9)
        lines!(ax, xs, src; color=(color, 0.45), linewidth=1.8, linestyle=:dash)
    end
    ax.xticks = (xs, ["slow", "middle", "fast"])
    xlims!(ax, 0.75, 3.25)
    text!(ax, 0.985, 0.97; space=:relative, align=(:right, :top), fontsize=14, color=GRAY,
        text="solid = reception   dashed = drive")
    axislegend(ax; position=:lt, framevisible=false)
    return summary
end

# ---------------------------------------------------------------------------
# Panel A skeleton schematic
#
# Sagittal-plane stick figure reconstructed by forward kinematics from the same
# six joint angles the operator is fitted to. Purely illustrative -- segment
# lengths are nominal fractions of leg length, not subject anthropometry -- but
# the POSES are real: they are the trial-mean gait cycle of the cohort, so the
# panel shows what the six channels actually encode.
#
# Channel order is [hip, knee, ankle] x [limb 1, limb 2], degrees, clinical
# sign conventions: hip flexion +, knee flexion +, ankle dorsiflexion +.
# ---------------------------------------------------------------------------

"""
    mean_gait_cycle(trial, f0; fs, n_phase)

Trial-mean gait cycle, `n_phase` x 6. Cycles are cut at successive peaks of hip
flexion (channel 1) spaced about one period apart, resampled to a common phase
grid, and averaged.
"""
function mean_gait_cycle(trial::AbstractMatrix, f0::Real; fs::Real=FS, n_phase::Int=60)
    P = round(Int, fs / f0)
    (isfinite(f0) && f0 > 0 && P >= 8 && size(trial, 1) > 2P) || return nothing
    h = @view trial[:, 1]
    # peaks of hip flexion, enforcing a refractory period of 0.6 * P
    peaks = Int[]
    minsep = max(4, round(Int, 0.6P))
    for t in 2:(length(h) - 1)
        if h[t] > h[t - 1] && h[t] >= h[t + 1]
            if isempty(peaks) || t - peaks[end] >= minsep
                push!(peaks, t)
            elseif h[t] > h[peaks[end]]
                peaks[end] = t
            end
        end
    end
    length(peaks) < 3 && return nothing
    grid = range(0, 1; length=n_phase)
    acc = zeros(n_phase, 6)
    n = 0
    for c in 1:(length(peaks) - 1)
        a, b = peaks[c], peaks[c + 1]
        len = b - a
        (len < round(Int, 0.5P) || len > round(Int, 1.8P)) && continue
        for j in 1:6, (k, u) in enumerate(grid)
            x = a + u * len
            i0 = clamp(floor(Int, x), 1, size(trial, 1) - 1)
            w = x - i0
            acc[k, j] += (1 - w) * trial[i0, j] + w * trial[i0 + 1, j]
        end
        n += 1
    end
    n == 0 && return nothing
    return acc ./ n
end

"""
    skeleton_points(angles; thigh, shank, foot)

Forward kinematics for one sagittal pose. `angles` is the 6-vector
[hip1, knee1, ankle1, hip2, knee2, ankle2] in degrees. Returns the hip origin
and, per limb, the (knee, ankle, toe) points in a hip-centred frame with +x
forward and +y up.
"""
function skeleton_points(angles::AbstractVector; thigh=0.62, shank=0.58, foot=0.24)
    limbs = NTuple{3,Point2f}[]
    for l in 0:1
        th = deg2rad(angles[3l + 1])          # hip flexion, from vertical
        tk = deg2rad(angles[3l + 2])          # knee flexion, shank lags thigh
        ta = deg2rad(angles[3l + 3])          # ankle dorsiflexion
        knee = Point2f(thigh * sin(th), -thigh * cos(th))
        a = th - tk                            # shank angle from vertical
        ankle = knee + Point2f(shank * sin(a), -shank * cos(a))
        af = a + ta                            # foot: perpendicular to shank, toe up on dorsiflexion
        toe = ankle + Point2f(foot * cos(af), foot * sin(af))
        push!(limbs, (knee, ankle, toe))
    end
    return limbs
end

"""
    draw_skeleton!(ax, angles; x0, ground, near, far, lw)

Draw one pose with its stance foot resting on `ground`. The hip height is set by
the pose rather than fixed, so the pelvis rises and falls across the cycle the
way it does in walking; pinning the hip instead makes the figure appear to float.
Limb 2 is drawn first and lighter so the near limb reads in front.
"""
function draw_skeleton!(ax, angles::AbstractVector; x0, ground, near=BLUE, far=(GRAY, 0.45), lw=2.6)
    limbs = skeleton_points(angles)
    # Rest the lowest point of the pose (the stance foot) on the ground line.
    lowest = minimum(min(l[2][2], l[3][2]) for l in limbs)
    y0 = ground - lowest
    hip = Point2f(x0, y0)
    shift(p) = Point2f(p[1] + x0, p[2] + y0)
    for (l, col, w) in ((2, far, lw * 0.8), (1, near, lw))
        knee, ankle, toe = limbs[l]
        lines!(ax, [hip, shift(knee), shift(ankle)]; color=col, linewidth=w)
        lines!(ax, [shift(ankle), shift(toe)]; color=col, linewidth=w)
        scatter!(ax, [shift(knee), shift(ankle)]; color=col, markersize=w * 1.8)
    end
    # trunk and head
    trunk = 0.52
    lines!(ax, [hip, Point2f(x0, y0 + trunk)]; color=near, linewidth=lw * 1.1)
    scatter!(ax, [Point2f(x0, y0 + trunk + 0.16)]; color=near, markersize=lw * 5.2)
    scatter!(ax, [hip]; color=near, markersize=lw * 2.0)
    return nothing
end

function make_figure5()
    ensure_dirs!()
    trials, meta = load_human_superset_trials()
    trials, meta = select_source(trials, meta, SOURCE)
    ab_idx = findall(String.(meta.lf_or_hf) .== "AB")
    ab_trials = trials[ab_idx]
    ab_meta = meta[ab_idx, :]
    @printf("[figure 5] source=%s -> %d able-bodied trials, %d subjects\n",
        SOURCE, length(ab_trials), length(unique(String.(ab_meta.subject))))

    metrics = compute_or_load_metrics(ab_trials, ab_meta)
    pred = compute_or_load_prediction(ab_trials, ab_meta)
    resolvent_profiles, resolvent_io = compute_or_load_resolvent_averages(ab_trials, ab_meta)

    fig = Figure(size=(1680, 1420), fontsize=PAPER_FONT_SIZE)

    axA = Axis(fig[1, 1], title="Human gait operator pipeline")
    panel_label!(fig, (1, 1), "A")
    hidespines!(axA); hidedecorations!(axA)
    xlims!(axA, 0, 10); ylims!(axA, 0.1, 5.9)

    # Top band: the six joint angles rendered as a sagittal stick figure, so the
    # reader can see what the state vector is before it is abstracted into an
    # operator. Poses are the cohort-mean gait cycle, not a cartoon.
    cycles = Matrix{Float64}[]
    for i in 1:length(ab_trials)
        f0 = trial_stride_frequency(ab_trials[i])
        C = mean_gait_cycle(ab_trials[i], f0; fs=FS, n_phase=60)
        C === nothing || push!(cycles, C)
    end
    if !isempty(cycles)
        Cbar = reduce(+, cycles) ./ length(cycles)
        n_pose = 6
        xs = range(1.15, 8.85; length=n_pose)
        ground = 3.95
        for (k, xk) in enumerate(xs)
            ph = round(Int, 1 + (k - 1) * (size(Cbar, 1) - 1) / (n_pose - 1))
            draw_skeleton!(axA, Cbar[ph, :]; x0=xk, ground=ground)
            text!(axA, xk, ground - 0.23; text=@sprintf("%d%%", round(Int, 100 * (k - 1) / (n_pose - 1))),
                align=(:center, :top), fontsize=14, color=GRAY)
        end
        lines!(axA, [0.6, 9.4], [ground, ground]; color=(GRAY, 0.5), linewidth=1.2)
        text!(axA, 5.0, ground - 0.62; text="one gait cycle", align=(:center, :top),
            fontsize=14, color=GRAY)
    end

    boxes = [(1.2, "6 joint\nangles"), (3.8, "delay\nembedding"), (6.4, "stabilized\nDMD"), (8.8, "response\nmetrics")]
    for (x, label) in boxes
        poly!(axA, Rect2f(x-0.8, 1.42, 1.6, 1.32); color=(:gray92, 1), strokecolor=GRAY, strokewidth=1.5)
        text!(axA, x, 2.08; text=label, align=(:center, :center), fontsize=17)
    end
    for x in (2.3, 4.9, 7.5)
        arrows!(axA, [x], [2.08], [0.85], [0]; arrowsize=16, linewidth=2, color=GRAY)
    end
    text!(axA, 0.45, 1.16; text=@sprintf("%d able-bodied trials, %d subjects\nYoung Adult study only",
        length(ab_trials), length(unique(String.(ab_meta.subject)))),
        align=(:left, :top), fontsize=15, color=GRAY)

    # B: representative held-out prediction
    example_i = argmin(abs.(Float64.(ab_meta.speed) .- median(Float64.(ab_meta.speed))))
    tr = center_trial(ab_trials[example_i])
    train = tr[1:min(TRAIN_ROWS, size(tr, 1)-300), :]
    test = tr[size(train, 1)+1:end, :]
    result = fit_trial_operator(train)
    pred_ex = rollout_prediction_error(result, test; n_steps=TEST_HORIZON)
    t = collect(0:TEST_HORIZON-1) ./ FS
    axB = Axis(fig[1, 2], title="Held-out able-bodied rollout", xlabel=UNIFORM_STRIDE ? "time (strides)" : "time (s)", ylabel="angle (deg)")
    panel_label!(fig, (1, 2), "B")
    vspan!(axB, t[1], t[min(N_DELAYS, length(t))]; color=(:gray70, 0.28))
    for (idx, color, label) in ((1, BLUE, "L hip"), (2, RED, "L knee"), (3, GREEN, "L ankle"))
        lines!(axB, t, pred_ex.y[:, idx]; color=color, linewidth=2, label=label)
        lines!(axB, t, pred_ex.yhat[:, idx]; color=color, linewidth=2, linestyle=:dash)
    end
    axislegend(axB; position=:rb, framevisible=false)

    axC = Axis(fig[1, 3], title="Forecast error across subjects", xlabel=UNIFORM_STRIDE ? "horizon (strides)" : "horizon (s)", ylabel="normalized RMSE")
    panel_label!(fig, (1, 3), "C")
    pred_sum = mean_sem_by_horizon(pred)
    xh = pred_sum.horizon ./ FS
    band!(axC, xh, pred_sum.dmd_mean .- pred_sum.dmd_sem, pred_sum.dmd_mean .+ pred_sum.dmd_sem; color=(BLUE, 0.18))
    lines!(axC, xh, pred_sum.dmd_mean; color=BLUE, linewidth=2.5, label="DMD")
    band!(axC, xh, pred_sum.phase_mean .- pred_sum.phase_sem, pred_sum.phase_mean .+ pred_sum.phase_sem; color=(GRAY, 0.18))
    lines!(axC, xh, pred_sum.phase_mean; color=GRAY, linewidth=2.5, linestyle=:dash, label="phase template")
    axislegend(axC; position=:lt, framevisible=false)

    lmm_fits = Tuple{Symbol,Any}[]
    glD = GridLayout(fig[2, 1:3])
    panel_label!(fig, (2, 1), "D")
    axD1 = Axis(glD[1, 1], title="Non-normality", xlabel="speed", ylabel="Henrici")
    axD2 = Axis(glD[1, 2], title="Transient growth", xlabel="speed", ylabel=L"\log_{10}\max_k \|A^k\|_2")
    axD3 = Axis(glD[1, 3], title="Effective dimensionality", xlabel="speed", ylabel="participation")
    push!(lmm_fits, (:henrici, mixed_metric!(axD1, metrics, :henrici)))
    push!(lmm_fits, (:log_tgmax, mixed_metric!(axD2, metrics, :log_tgmax)))
    push!(lmm_fits, (:participation, mixed_metric!(axD3, metrics, :participation)))

    axE = Axis(fig[3, 1], title="Trial-averaged resolvent profiles",
        xlabel="frequency / stride frequency", ylabel="resolvent gain", yscale=log10)
    panel_label!(fig, (3, 1), "E")
    colors = Dict("slow" => BLUE, "middle" => GOLD, "fast" => RED)
    for bin in ("slow", "middle", "fast")
        sub = resolvent_profiles[resolvent_profiles.speed_bin .== bin, :]
        isempty(sub) && continue
        band!(axE, sub.frequency_ratio, sub.lo, sub.hi; color=(colors[bin], 0.16))
        lines!(axE, sub.frequency_ratio, sub.mean_gain; color=colors[bin], linewidth=2.5, label=bin)
    end
    vlines!(axE, [1, 2, 3]; color=(:gray45, 0.45), linestyle=:dot)
    xlims!(axE, 0.5, 4.5)
    axislegend(axE; position=:rt, framevisible=false)

    axF = Axis(fig[3, 2], title="Q-factor", xlabel="speed", ylabel="log₁₀ Q-factor")
    panel_label!(fig, (3, 2), "F")
    push!(lmm_fits, (:log_q_factor, mixed_metric!(axF, metrics, :log_q_factor)))
    # log scale: on the phase grid the stride peak is very sharp and Q spans
    # two decades, so a linear axis with a cap hides most of the spread.

    function io_matrix_from_df(bin)
        M = fill(NaN, 6, 6)
        sub = resolvent_io[resolvent_io.speed_bin .== bin, :]
        for row in eachrow(sub)
            M[Int(row.output_index), Int(row.input_index)] = Float64(row.mean_gain)
        end
        return M
    end

    # G: the all-trial mean I/O map, shown as the two vectors it contains.
    # The map is rank one (>= 99.9% of its energy in the first singular value
    # against 0.87 for a shuffled null), so a 6x6 heatmap displays 36 cells
    # that are really the outer product of a reception weight and a drive
    # weight. Plotting the vectors is the same information without implying
    # 36 independent pathways.
    glG = GridLayout(fig[3, 3])
    axG = Axis(glG[1, 1], title="All-trial reception and drive", xticklabelrotation=0.9,
        xlabel="joint", ylabel="weight", xticks=(1:6, JOINT_LABELS))
    panel_label!(fig, (3, 3), "G")
    M_all = io_matrix_from_df("all")
    F_all = svd(M_all)
    sink_all = abs.(F_all.U[:, 1]); sink_all ./= sum(sink_all)
    src_all = abs.(F_all.V[:, 1]);  src_all ./= sum(src_all)
    sv_all = svdvals(M_all)
    # Same convention as Figure 6D-F: reception dark, drive light, one colour.
    barplot!(axG, (1:6) .- 0.19, sink_all; color=BLUE, width=0.36, label="reception")
    barplot!(axG, (1:6) .+ 0.19, src_all;  color=(BLUE, 0.35), width=0.36, label="drive")
    axislegend(axG; position=:lt, framevisible=false, labelsize=13)
    text!(axG, 0.98, 0.97; space=:relative, align=(:right, :top), fontsize=12, color=GRAY,
        text=@sprintf("rank one: %.1f%%", 100 * sv_all[1]^2 / sum(abs2, sv_all)))

    # H: one panel per joint with the three speed bins side by side, so the
    # speed effect on each joint is read off directly instead of across panels.
    pathway_summary = sink_source_by_speed(resolvent_io)
    bin_vecs = Dict{String,Any}()
    for bin in ("slow", "middle", "fast")
        Fb = svd(io_matrix_from_df(bin)); svb = Fb.S
        u = abs.(Fb.U[:, 1]); v = abs.(Fb.V[:, 1])
        bin_vecs[bin] = (sink=u ./ sum(u), source=v ./ sum(v), e1=svb[1]^2 / sum(abs2, svb))
    end
    vec_hi = max(maximum(sink_all), maximum(maximum(b.sink) for b in values(bin_vecs))) * 1.25
    ylims!(axG, 0, vec_hi)
    glH = GridLayout(fig[4, 1:3])
    panel_label!(fig, (4, 1), "H")
    bins = ("slow", "middle", "fast")
    bin_colors = Dict("slow" => BLUE, "middle" => GOLD, "fast" => RED)
    # Channel order is [R hip, R knee, R ankle, L hip, L knee, L ankle].
    for (k, joint) in enumerate(("hip", "knee", "ankle"))
        idx = (k, k + 3)                                  # (R, L) channels
        ax = Axis(glH[1, k], title=joint, ylabel=(k == 1 ? "weight" : ""),
            xticks=(1:4, ["reception R", "reception L", "drive R", "drive L"]),
            xticklabelrotation=0.5)
        for (b, bin) in enumerate(bins)
            v = bin_vecs[bin]
            ys = [v.sink[idx[1]], v.sink[idx[2]], v.source[idx[1]], v.source[idx[2]]]
            cols = [bin_colors[bin], bin_colors[bin], (bin_colors[bin], 0.4), (bin_colors[bin], 0.4)]
            barplot!(ax, (1:4) .+ (b - 2) * 0.26, ys; color=cols, width=0.24, label=bin)
        end
        ylims!(ax, 0, vec_hi)
        k > 1 && hideydecorations!(ax; grid=false)
        k == 3 && axislegend(ax, [PolyElement(color=bin_colors[b]) for b in bins],
            ["slow", "middle", "fast"], "walking speed"; position=:rt, framevisible=false,
            labelsize=13, orientation=:horizontal, titleposition=:left)
    end
    Label(glH[0, :], @sprintf("reception (dark) and drive (light) by joint and speed; rank one: %s",
            join([@sprintf("%.1f%%", 100 * bin_vecs[b].e1) for b in bins], ", ")),
        fontsize=13, color=GRAY)
    nrow(pathway_summary) > 0 &&
        CSV.write(joinpath(RESULTS_DIR, "figure5_speed_pathway_summary.csv"), pathway_summary)

    rowgap!(fig.layout, 18)
    colgap!(fig.layout, 18)
    # Row 1 carries panel A's skeleton schematic; give it enough height that
    # the poses are not vertically compressed against the x scale.
    rowsize!(fig.layout, 1, Relative(0.28))
    CSV.write(joinpath(RESULTS_DIR, "figure5_speed_mixed_models_$(DMD_CONFIG_TAG).csv"), mixed_model_summary(lmm_fits))
    savefig("Figure 5.svg", fig)
    return fig
end

make_figure5()

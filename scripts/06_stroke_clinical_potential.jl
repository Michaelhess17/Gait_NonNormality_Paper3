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
using ProgressMeter
using Random
using Statistics

const GOLD = RGBf(0.86, 0.56, 0.12)
const PURPLE = RGBf(0.42, 0.25, 0.62)
const GRAY = RGBf(0.25, 0.25, 0.25)
const HEATMAP_CMAP = :viridis
const DIFF_CMAP = :balance

const FS = 100.0
const N_DELAYS = parse(Int, get(ENV, "FIGURE6_N_DELAYS", "32"))
const PANEL_METRIC_RANK = parse(Int, get(ENV, "FIGURE6_PANEL_METRIC_RANK", "70"))

# Single-source figure: the PostStroke Gait Biomechanics study only, so the AB
# control group shares an acquisition and preprocessing pipeline with the HF/LF
# trials it is contrasted against. Pooling in the Healthy Young Adult able-bodied
# trials (as earlier drafts did) made the AB-vs-stroke contrast partly a contrast
# between preprocessing pipelines: those controls sit ~1.8 decades away in
# transient growth at rank 70, while this study's own controls sit with the
# stroke groups. See the SOURCE_* docstring in src/GaitNonNormalityPaper3.jl.
const SOURCE = get(ENV, "FIGURE6_SOURCE", SOURCE_PSGB)
const SOURCE_TAG = get(SOURCE_TAGS, SOURCE, "src")

# The source slug is part of the cache tag so a restricted run cannot silently
# read a cache built from the pooled superset.
const DMD_CONFIG_TAG = "d$(N_DELAYS)_fixedrank$(PANEL_METRIC_RANK)_$(SOURCE_TAG)$(PHASE_TAG)"
const PROFILE_RESOLVENT_ANGLES = 1000
const FORCE_RECOMPUTE = lowercase(get(ENV, "FORCE_FIGURE6", "false")) in ("1", "true", "yes")

# Pools AB with HF/LF. Channels 1-3 are the paretic limb for stroke trials
# and the right limb for AB; group-specific panels use `joint_labels(group)`.
const JOINT_LABELS = JOINT_LABELS_POOLED

function trial_stride_frequency(trial; fs=FS)
    UNIFORM_STRIDE && return fs / PHASE_STEPS        # exactly one stride per PHASE_STEPS samples
    f0 = estimate_stride_frequency(center_trial(trial); fs=fs, fmin=0.25, fmax=5.0)
    return isfinite(f0) && f0 > 0 ? f0 : 1.0
end

function fit_trial_operator(trial; n_delays=N_DELAYS, rank_rule=FixedRank(PANEL_METRIC_RANK), stabilize=true)
    h = HankelGenerator(n_delays=n_delays, delay_interval=1)
    model = DMD(rank=rank_rule, stabilize=stabilize)
    return fit_gait_operator(model, center_trial(trial), h)
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

function profile_stats(mat)
    meanv = fill(NaN, size(mat, 2))
    lo = similar(meanv)
    hi = similar(meanv)
    for j in axes(mat, 2)
        meanv[j], lo[j], hi[j] = positive_log_mean_sem(@view mat[:, j])
    end
    return meanv, lo, hi
end

finite_vec(v) = Float64[x for x in Float64.(v) if isfinite(x)]

function robust_limits(v; qlo=0.01, qhi=0.99, pad_frac=0.03)
    vals = finite_vec(v)
    isempty(vals) && return (NaN, NaN)
    lo, hi = quantile(vals, [qlo, qhi])
    lo == hi && return (lo - 1, hi + 1)
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad
end

function intercept_slope(x, y)
    xx = Float64.(x)
    yy = Float64.(y)
    ok = isfinite.(xx) .& isfinite.(yy)
    count(ok) < 2 && return (NaN, NaN)
    std(xx[ok]) == 0 && return (mean(yy[ok]), NaN)
    b = cov(xx[ok], yy[ok]) / var(xx[ok])
    a = mean(yy[ok]) - b * mean(xx[ok])
    return a, b
end

function pr_slope_berg_table(data)
    scores = load_clinical_scores()
    nrow(scores) == 0 && return DataFrame()
    stroke = data[data.group .!= "AB", :]
    rows = NamedTuple[]
    for sub in groupby(stroke, [:subject, :group])
        nrow(sub) >= 2 || continue
        _, slope = intercept_slope(sub.speed, sub.participation)
        isfinite(slope) || continue
        push!(rows, (subject=String(first(sub.subject)),
                     group=String(first(sub.group)),
                     pr_slope=slope,
                     median_speed=median(Float64.(sub.speed))))
    end
    isempty(rows) && return DataFrame()
    return innerjoin(DataFrame(rows), scores; on=:subject)
end

function bootstrap_pr_berg_fits(data; n_boot=1500, seed=307)
    scores = load_clinical_scores()
    nrow(scores) == 0 && return DataFrame(intercept=Float64[], slope=Float64[])
    d = innerjoin(data[data.group .!= "AB", :], scores; on=:subject)
    keep = isfinite.(Float64.(d.berg)) .& isfinite.(Float64.(d.participation)) .& isfinite.(Float64.(d.speed))
    d = d[keep, :]
    subjects = sort(unique(d.subject))
    length(subjects) < 3 && return DataFrame(intercept=Float64[], slope=Float64[])
    rng = MersenneTwister(seed)
    rows = NamedTuple[]
    for _ in 1:n_boot
        sampled_subjects = rand(rng, subjects, length(subjects))
        x = Float64[]
        y = Float64[]
        for s in sampled_subjects
            ds = d[d.subject .== s, :]
            nrow(ds) >= 2 || continue
            boot = ds[rand(rng, 1:nrow(ds), nrow(ds)), :]
            _, slope = intercept_slope(boot.speed, boot.participation)
            isfinite(slope) || continue
            push!(x, slope)
            push!(y, Float64(first(boot.berg)))
        end
        length(y) >= 3 && std(x) > 0 || continue
        β = hcat(ones(length(x)), x) \ y
        push!(rows, (intercept=β[1], slope=β[2]))
    end
    return DataFrame(rows)
end

function draw_regression_bootstrap_band!(ax, fits::DataFrame, xs; color=:black)
    nrow(fits) == 0 && return nothing
    pred = reduce(hcat, [fits.intercept[i] .+ fits.slope[i] .* xs for i in 1:nrow(fits)])
    μ = [mean(view(pred, j, :)) for j in 1:length(xs)]
    se = [std(view(pred, j, :)) for j in 1:length(xs)]
    band!(ax, xs, μ .- 1.96 .* se, μ .+ 1.96 .* se; color=(color, 0.18))
    lines!(ax, xs, μ; color=color, linewidth=2.8)
    return nothing
end

function pretty_feature(s)
    replacements = Dict(
        "median_log_tgmax" => "median log TG",
        "slope_log_tgmax" => "slope log TG",
        "median_henrici" => "median Henrici",
        "slope_henrici" => "slope Henrici",
        "median_participation" => "median PR",
        "slope_participation" => "slope PR",
        "median_stride_frequency" => "median stride freq.",
        "slope_stride_frequency" => "slope stride freq.",
        "median_log_q_factor" => "median log Q",
        "slope_log_q_factor" => "slope log Q",
        "median_resolvent_peak_gain" => "median peak gain",
        "slope_resolvent_peak_gain" => "slope peak gain",
        "median_gain2_over_gain1" => "median g2/g1",
        "slope_gain2_over_gain1" => "slope g2/g1",
    )
    haskey(replacements, s) && return replacements[s]
    for (prefix, gain_label) in (("norm_", " (norm.)"), ("abs_", " (abs.)"))
        startswith(s, prefix) || continue
        inner = s[(length(prefix) + 1):end]
        stat = startswith(inner, "median_") ? "median" : startswith(inner, "slope_") ? "slope" : ""
        body = replace(inner, r"^(median_|slope_)" => "")
        if startswith(body, "sink_")
            tag = split(body, "_")[2]
            return "$(stat) $(side_label(tag)) reception$(gain_label)"
        elseif startswith(body, "sidepath_")
            _, in_tag, out_tag = split(body, "_")
            return "$(stat) $(side_label(in_tag))→$(side_label(out_tag)) gain$(gain_label)"
        elseif startswith(body, "pathway_")
            _, in_joint, out_joint = split(body, "_")
            return "$(stat) $(in_joint)→$(out_joint) gain$(gain_label)"
        elseif startswith(body, "asym_")
            joint = split(body, "_")[2]
            return "$(stat) $(joint) asymmetry$(gain_label)"
        end
    end
    return replace(s, "_" => " ")
end

function clinical_feature_names()
    bases = [
        "stride_frequency",
        "henrici",
        "log_tgmax",
        "participation",
        "resolvent_peak_gain",
        "resolvent_peak_frequency",
        "log_q_factor",
        "gain2_over_gain1",
    ]
    return vcat(["median_$b" for b in bases], ["slope_$b" for b in bases])
end

function subject_feature_value(sub::DataFrame, feature::AbstractString)
    if startswith(feature, "median_")
        col = Symbol(feature[8:end])
        col in propertynames(sub) || return NaN
        vals = filter(isfinite, Float64.(sub[!, col]))
        return isempty(vals) ? NaN : median(vals)
    elseif startswith(feature, "slope_")
        col = Symbol(feature[7:end])
        col in propertynames(sub) || return NaN
        _, slope = intercept_slope(sub.speed, sub[!, col])
        return slope
    end
    return NaN
end

function clinical_original_screen(metrics)
    scores = load_clinical_scores()
    nrow(scores) == 0 && return DataFrame()
    stroke = innerjoin(metrics[metrics.group .!= "AB", :], scores; on=:subject)
    subjects = groupby(stroke, [:subject, :group])
    features = clinical_feature_names()
    rows = NamedTuple[]
    for outcome in ["berg", "fugl"]
        for feature in features
            x = Float64[]
            y = Float64[]
            for sub in subjects
                val = subject_feature_value(DataFrame(sub), feature)
                score = Float64(first(sub[!, Symbol(outcome)]))
                if isfinite(val) && isfinite(score)
                    push!(x, val)
                    push!(y, score)
                end
            end
            length(x) >= 4 && std(x) > 0 && std(y) > 0 || continue
            push!(rows, (
                outcome=outcome,
                feature=feature,
                n=length(x),
                pearson_r=cor(x, y),
                r2=cor(x, y)^2,
            ))
        end
    end
    return DataFrame(rows)
end

function clinical_hierarchical_bootstrap_screen(metrics; n_boot=1500, seed=611)
    boot_path = joinpath(RESULTS_DIR, "figure6_clinical_hierarchical_bootstrap_xlsxscores_$(DMD_CONFIG_TAG).csv")
    summary_path = joinpath(RESULTS_DIR, "figure6_clinical_hierarchical_summary_xlsxscores_$(DMD_CONFIG_TAG).csv")
    if isfile(boot_path) && isfile(summary_path) && !FORCE_RECOMPUTE
        return CSV.read(boot_path, DataFrame), CSV.read(summary_path, DataFrame)
    end

    scores = load_clinical_scores()
    nrow(scores) == 0 && return DataFrame(), DataFrame()
    stroke = innerjoin(metrics[metrics.group .!= "AB", :], scores; on=:subject)
    subjects = sort(unique(String.(stroke.subject)))
    features = clinical_feature_names()
    rng = MersenneTwister(seed)
    rows = NamedTuple[]

    for boot in 1:n_boot
        sampled_subjects = rand(rng, subjects, length(subjects))
        for outcome in ["berg", "fugl"]
            xs_by_feature = Dict(feature => Float64[] for feature in features)
            y = Float64[]
            for subject in sampled_subjects
                ds = stroke[String.(stroke.subject) .== subject, :]
                nrow(ds) == 0 && continue
                sampled_rows = rand(rng, 1:nrow(ds), nrow(ds))
                boot_sub = ds[sampled_rows, :]
                score = Float64(first(boot_sub[!, Symbol(outcome)]))
                isfinite(score) || continue
                push!(y, score)
                for feature in features
                    push!(xs_by_feature[feature], subject_feature_value(boot_sub, feature))
                end
            end
            for feature in features
                x = xs_by_feature[feature]
                ok = isfinite.(x) .& isfinite.(y)
                count(ok) >= 4 && std(x[ok]) > 0 && std(y[ok]) > 0 || continue
                push!(rows, (
                    boot=boot,
                    outcome=outcome,
                    feature=feature,
                    n_subjects=count(ok),
                    r2=cor(x[ok], y[ok])^2,
                ))
            end
        end
    end

    boot_df = DataFrame(rows)
    summary_rows = NamedTuple[]
    for sub in groupby(boot_df, [:outcome, :feature])
        vals = filter(isfinite, Float64.(sub.r2))
        isempty(vals) && continue
        push!(summary_rows, (
            outcome=String(first(sub.outcome)),
            feature=String(first(sub.feature)),
            mean_r2=mean(vals),
            sem_r2=length(vals) > 1 ? std(vals) / sqrt(length(vals)) : 0.0,
            n_boot=length(vals),
        ))
    end
    summary = DataFrame(summary_rows)
    original = clinical_original_screen(metrics)
    if nrow(original) > 0
        summary = leftjoin(summary, original[:, [:outcome, :feature, :r2, :pearson_r, :n]];
            on=[:outcome, :feature], renamecols="" => "_original")
    end
    CSV.write(boot_path, boot_df)
    CSV.write(summary_path, summary)
    return boot_df, summary
end

function selected_clinical_features(summary; n_each=4)
    rows = NamedTuple[]
    for outcome in ["berg", "fugl"]
        sub = summary[summary.outcome .== outcome, :]
        isempty(sub) && continue
        score_col = :r2_original in propertynames(sub) ? :r2_original : :mean_r2
        ord = sortperm(sub[!, score_col]; rev=true)[1:min(n_each, nrow(sub))]
        for row in eachrow(sub[ord, :])
            push!(rows, (
                outcome=String(row.outcome),
                feature=String(row.feature),
                label="$(row.outcome == "berg" ? "Berg" : "Fugl-Meyer"): $(pretty_feature(String(row.feature)))",
            ))
        end
    end
    return DataFrame(rows)
end

function rank_values(v)
    vals = Float64.(v)
    order = sortperm(vals)
    ranks = zeros(Float64, length(vals))
    i = 1
    while i <= length(vals)
        j = i
        while j < length(vals) && vals[order[j + 1]] == vals[order[i]]
            j += 1
        end
        ranks[order[i:j]] .= (i + j) / 2
        i = j + 1
    end
    return ranks
end

function clinical_subject_feature_table(metrics)
    scores = load_clinical_scores()
    nrow(scores) == 0 && return DataFrame()
    stroke = innerjoin(metrics[metrics.group .!= "AB", :], scores; on=:subject)
    nrow(stroke) == 0 && return DataFrame()
    rows = Dict{Symbol,Any}[]
    features = clinical_feature_names()
    for sub in groupby(stroke, [:subject, :group])
        row = Dict{Symbol,Any}(
            :subject => String(first(sub.subject)),
            :group => String(first(sub.group)),
            :berg => Float64(first(sub.berg)),
            :fugl => Float64(first(sub.fugl)),
            :median_speed => median(Float64.(sub.speed)),
        )
        for feature in features
            row[Symbol(feature)] = subject_feature_value(DataFrame(sub), feature)
        end
        push!(rows, row)
    end
    return DataFrame(rows)
end

function best_dmd_clinical_features(subjects::DataFrame;
        features=filter(f -> !occursin("stride_frequency", f), clinical_feature_names()))
    rows = NamedTuple[]
    for outcome in ["berg", "fugl"]
        y = Float64.(subjects[!, Symbol(outcome)])
        best_feature = ""
        best_score = -Inf
        best_pearson = NaN
        best_spearman = NaN
        best_n = 0
        for feature in features
            col = Symbol(feature)
            col in propertynames(subjects) || continue
            x = Float64.(subjects[!, col])
            ok = isfinite.(x) .& isfinite.(y)
            count(ok) >= 4 && std(x[ok]) > 0 && std(y[ok]) > 0 || continue
            pearson_r = cor(x[ok], y[ok])
            spearman_r = cor(rank_values(x[ok]), rank_values(y[ok]))
            score = outcome == "fugl" ? spearman_r^2 : pearson_r^2
            if score > best_score
                best_feature = feature
                best_score = score
                best_pearson = pearson_r
                best_spearman = spearman_r
                best_n = count(ok)
            end
        end
        !isempty(best_feature) && push!(rows, (
            outcome=outcome,
            feature=best_feature,
            selection_metric=outcome == "fugl" ? "spearman_r2" : "pearson_r2",
            pearson_r=best_pearson,
            pearson_r2=best_pearson^2,
            spearman_r=best_spearman,
            spearman_r2=best_spearman^2,
            n=best_n,
        ))
    end
    return DataFrame(rows)
end

"""Association statistics only, with no axis. Used for the walking-speed
baseline, which is still reported in the results table but no longer plotted
in panel I."""
function clinical_association_stats(subjects, xcol::Symbol, ycol::Symbol)
    x = Float64.(subjects[!, xcol])
    y = Float64.(subjects[!, ycol])
    ok = isfinite.(x) .& isfinite.(y)
    if count(ok) < 4 || std(x[ok]) == 0 || std(y[ok]) == 0
        return (pearson_r=NaN, pearson_r2=NaN, spearman_r=NaN, spearman_r2=NaN, n=count(ok))
    end
    pearson_r = cor(x[ok], y[ok])
    spearman_r = cor(rank_values(x[ok]), rank_values(y[ok]))
    return (pearson_r=pearson_r, pearson_r2=pearson_r^2,
        spearman_r=spearman_r, spearman_r2=spearman_r^2, n=count(ok))
end

function add_clinical_association!(ax, subjects, xcol::Symbol, ycol::Symbol; color, title, xlabel,
        ranked=false)
    x = Float64.(subjects[!, xcol])
    y = Float64.(subjects[!, ycol])
    ok = isfinite.(x) .& isfinite.(y)
    if count(ok) < 4 || std(x[ok]) == 0 || std(y[ok]) == 0
        text!(ax, 0, 0; text="insufficient data", align=(:center, :center))
        return (pearson_r=NaN, pearson_r2=NaN, spearman_r=NaN, spearman_r2=NaN, n=count(ok))
    end
    x_plot = ranked ? rank_values(x[ok]) : x[ok]
    y_plot = ranked ? rank_values(y[ok]) : y[ok]
    scatter!(ax, x_plot, y_plot; color=(color, 0.72), markersize=9)
    a, b = intercept_slope(x_plot, y_plot)
    xs = [minimum(x_plot), maximum(x_plot)]
    ys = a .+ b .* xs
    lines!(ax, xs, ys; color=color, linewidth=2.5)
    pearson_r = cor(x[ok], y[ok])
    spearman_r = cor(rank_values(x[ok]), rank_values(y[ok]))
    ax.title = title
    ax.xlabel = ranked ? "$(xlabel) rank" : xlabel
    ax.ylabel = ranked ? (ycol == :berg ? "Berg rank" : "Fugl-Meyer rank") :
        (ycol == :berg ? "Berg Balance Scale" : "Fugl-Meyer")
    stat_text = ranked ?
        @sprintf("ρ = %.2f, ρ² = %.2f", spearman_r, spearman_r^2) :
        @sprintf("r = %.2f, R² = %.2f", pearson_r, pearson_r^2)
    text!(ax, 0.03, 0.95; text=stat_text,
        space=:relative, align=(:left, :top), fontsize=12)
    return (pearson_r=pearson_r, pearson_r2=pearson_r^2,
        spearman_r=spearman_r, spearman_r2=spearman_r^2, n=count(ok))
end

function plot_clinical_best_associations!(gl, subjects::DataFrame, summary)
    if nrow(subjects) == 0
        ax = Axis(gl[1, 1])
        text!(ax, 0, 0; text="clinical scores unavailable", align=(:center, :center))
        return DataFrame()
    end
    best = best_overall_clinical_features(subjects)
    if nrow(best) == 0
        ax = Axis(gl[1, 1])
        text!(ax, 0, 0; text="insufficient clinical association data", align=(:center, :center))
        return DataFrame()
    end
    report_rows = NamedTuple[]
    for (icol, outcome) in enumerate(["berg", "fugl"])
        ycol = Symbol(outcome)
        color = outcome == "berg" ? GOLD : PURPLE
        outcome_label = outcome == "berg" ? "Berg" : "Fugl-Meyer"

        # The walking-speed baseline panels were removed from the figure: the
        # paper does not claim to outperform speed, and showing the comparison
        # here invited a horse race that is not the point of the panel. The
        # speed statistics are still computed and written to
        # figure6_panelI_clinical_best_associations.csv so the comparison
        # remains on record and can be quoted in the text.
        speed_stats = clinical_association_stats(subjects, :median_speed, ycol)
        push!(report_rows, (outcome=outcome, predictor="median_speed", predictor_type="speed_baseline",
            selection_metric="pearson_r2",
            pearson_r=speed_stats.pearson_r, pearson_r2=speed_stats.pearson_r2,
            spearman_r=speed_stats.spearman_r, spearman_r2=speed_stats.spearman_r2,
            n=speed_stats.n))

        row = best[best.outcome .== outcome, :][1, :]
        feature = String(row.feature)
        xcol = Symbol(feature)
        ranked = outcome == "fugl"
        rank_suffix = ranked ? " ranks" : ""
        ax_dmd = Axis(gl[1, icol], title="$(outcome_label): $(pretty_feature(feature))$(rank_suffix)",
            xlabel=pretty_feature(feature), ylabel="")
        dmd_stats = add_clinical_association!(ax_dmd, subjects, xcol, ycol;
            color=color, title="$(outcome_label): $(pretty_feature(feature))$(rank_suffix)",
            xlabel=pretty_feature(feature), ranked=ranked)
        push!(report_rows, (outcome=outcome, predictor=feature, predictor_type=predictor_type_label(feature),
            selection_metric=String(row.selection_metric),
            pearson_r=dmd_stats.pearson_r, pearson_r2=dmd_stats.pearson_r2,
            spearman_r=dmd_stats.spearman_r, spearman_r2=dmd_stats.spearman_r2,
            n=dmd_stats.n))
    end
    colgap!(gl, 14)
    rowgap!(gl, 12)
    return DataFrame(report_rows)
end

function plot_clinical_bootstrap_screen!(ax, boot, summary)
    selected = selected_clinical_features(summary; n_each=4)
    if nrow(selected) == 0
        text!(ax, 0, 0; text="clinical scores unavailable", align=(:center, :center))
        return selected
    end
    rng = MersenneTwister(612)
    for (k, row) in enumerate(eachrow(selected))
        vals = boot.r2[(boot.outcome .== row.outcome) .& (boot.feature .== row.feature)]
        all_vals = filter(isfinite, Float64.(vals))
        isempty(all_vals) && continue
        plot_vals = all_vals
        if length(plot_vals) > 400
            plot_vals = plot_vals[rand(rng, 1:length(plot_vals), 400)]
        end
        color = row.outcome == "berg" ? GOLD : PURPLE
        xs = fill(k, length(plot_vals)) .+ 0.08 .* randn(rng, length(plot_vals))
        scatter!(ax, xs, plot_vals; color=(color, 0.22), markersize=3)
        meanv = mean(all_vals)
        semv = length(all_vals) > 1 ? std(all_vals) / sqrt(length(all_vals)) : 0.0
        lines!(ax, [k, k], [meanv - semv, meanv + semv]; color=:black, linewidth=2.2)
        scatter!(ax, [k], [meanv]; color=:black, markersize=9)
    end
    ax.xticks = (1:nrow(selected), selected.label)
    ax.xticklabelrotation = 0.55
    ylims!(ax, 0, 1)
    return selected
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

# Transient-growth horizon in operator steps spanning exactly ONE STRIDE.
# The operator advances one sample per step, so this is FS/f0. Reported
# alongside the metric so the horizon is auditable per trial.
function one_cycle_steps(f0; fallback::Int)
    isfinite(f0) && f0 > 0 || return fallback
    return clamp(round(Int, FS / f0), 8, 4000)
end

function metric_row(trial, meta_row)
    result = fit_trial_operator(trial)
    f0 = trial_stride_frequency(trial)
    harms = harmonic_gains(result, f0; n_harmonics=3, fs=FS)
    peak = resolvent_peak(result; n_angles=360, fs=FS, fmin=0.25)
    q = q_factor(result, f0; fs=FS, step=0.01)
    return (
        trial_index=Int(meta_row.trial_index),
        source=String(meta_row.source),
        subject=String(meta_row.subject),
        speed=Float64(meta_row.speed),
        group=String(meta_row.lf_or_hf),
        rank=result.rank,
        stride_frequency=f0,
        henrici=henrici_departure(result),
        log_tgmax=log10(transient_growth(result; kmax=one_cycle_steps(f0; fallback=3N_DELAYS))),
        tg_horizon_steps=one_cycle_steps(f0; fallback=3N_DELAYS),
        participation=effective_rank(result.singular_values),
        resolvent_peak_gain=peak.gain,
        resolvent_peak_frequency=peak.frequency,
        q_factor=q,
        log_q_factor=log10(max(q, eps())),
        gain2_over_gain1=length(harms) >= 2 ? harms[2] / max(harms[1], eps()) : NaN,
    )
end

function compute_or_load_metrics(trials, meta)
    path = joinpath(RESULTS_DIR, "figure6_metrics_$(DMD_CONFIG_TAG).csv")
    if isfile(path) && !FORCE_RECOMPUTE
        println("Loading Figure 6 metrics: $path")
        return CSV.read(path, DataFrame)
    end
    rows = NamedTuple[]
    println("Fitting Figure 6 fixed-rank trial metrics ($(length(trials)) trials)...")
    for i in eachindex(trials)
        try
            push!(rows, metric_row(trials[i], meta[i, :]))
            print(".")
        catch err
            @warn "Failed Figure 6 metric fit" i err
        end
    end
    println()
    df = DataFrame(rows)
    CSV.write(path, df)
    println("Wrote Figure 6 metrics: $path")
    return df
end

function compute_or_load_profiles_and_io(trials, meta)
    # Keep this separate from the older stride-normalized profile cache.
    profile_path = joinpath(RESULTS_DIR, "figure6_resolvent_profiles_raw_$(DMD_CONFIG_TAG).csv")
    io_path = joinpath(RESULTS_DIR, "figure6_resolvent_io_$(DMD_CONFIG_TAG).csv")
    if isfile(profile_path) && isfile(io_path) && !FORCE_RECOMPUTE
        println("Loading Figure 6 resolvent summaries")
        return CSV.read(profile_path, DataFrame), CSV.read(io_path, DataFrame)
    end

    # Step 0.005 strides, landing exactly on every harmonic (shared with Figure 5E).
    grid = collect(range(0.5, 4.5; length=801))
    profile_rows = NamedTuple[]
    io_rows = NamedTuple[]
    cached_io = isfile(io_path) && !FORCE_RECOMPUTE ? CSV.read(io_path, DataFrame) : nothing
    total_profiles = sum(count(String.(meta.lf_or_hf) .== group) for group in GROUP_ORDER)
    progress = Progress(total_profiles; desc="Raw resolvent profiles", dt=1.0, showspeed=true)
    for group in GROUP_ORDER
        idxs = findall(String.(meta.lf_or_hf) .== group)
        gains = fill(NaN, length(idxs), length(grid))
        io_stack = cached_io === nothing ? fill(NaN, length(idxs), 6, 6) : nothing
        for (row, i) in enumerate(idxs)
            try
                result = fit_trial_operator(trials[i])
                f0 = trial_stride_frequency(trials[i])
                gains[row, :] .= resolvent_gain_at(result, grid .* f0; fs=FS)
                if io_stack !== nothing
                # I/O structure read from the DELAY-EMBEDDED operator (opnorm over
                # each joint's Hankel block). The explicit [angle, velocity] state
                # space was tried here and is less scale-sensitive, but it does not
                # carry the two clinical signatures: LF directional asymmetry
                # (P hip -> NP knee vs reverse) goes from p=0.031 to p=0.155, and HF
                # NP-knee self-coupling reverses sign. See the 2026-09-21 comparison.
                    M = hankel_block_resolvent_io_matrix(result, f0; fs=FS)
                    M === nothing && continue
                    # Total-gain (not max-pixel) normalization -- see figure 5.
                    io_stack[row, :, :] .= M ./ max(sum(M), eps())
                end
            catch err
                @warn "Failed Figure 6 resolvent summary" group i err
            finally
                next!(progress; showvalues=[(:group, group), (:trial, i)])
            end
        end
        meanv, lo, hi = profile_stats(gains)
        for j in eachindex(grid)
            push!(profile_rows, (
                group=group,
                frequency_ratio=grid[j],
                mean_gain=meanv[j],
                lo=lo[j],
                hi=hi[j],
                n_trials=length(idxs),
            ))
        end
        if io_stack !== nothing
            for out in 1:6, input in 1:6
                vals = io_stack[:, out, input]
                vals = vals[isfinite.(vals)]
                push!(io_rows, (
                    group=group,
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
    end
    finish!(progress)
    profiles = DataFrame(profile_rows)
    io = cached_io === nothing ? DataFrame(io_rows) : cached_io
    CSV.write(profile_path, profiles)
    CSV.write(io_path, io)
    println("Wrote Figure 6 resolvent profiles: $profile_path")
    println("Wrote Figure 6 resolvent I/O: $io_path")
    return profiles, io
end

# Trial points by group, each subject's mixed-model line (thin), and each
# group's population line with its 95% band (thick), drawn only over the speeds
# that group actually walked at -- the AB and LF speed ranges do not overlap,
# so extending the lines would present an extrapolation as data. Model:
# y ~ group * log2(speed) + (1 + log2(speed) | subject); see src/.
function mixed_metric!(ax, df, col)
    res = speed_mixed_model(df, col; grouped=true)
    pop, subj = mixed_model_lines(res)
    for group in GROUP_ORDER
        sub = df[df.group .== group, :]
        isempty(sub) && continue
        color = GROUP_COLORS[group]
        scatter!(ax, sub.speed, sub[!, col]; color=(color, 0.55), markersize=8,
            marker=GROUP_MARKERS[group], label=group)
    end
    for s in groupby(subj, :subject)
        lines!(ax, s.speed, s.fit; color=(GROUP_COLORS[first(s.group)], 0.45), linewidth=1.2)
    end
    for g in groupby(pop, :group)
        color = GROUP_COLORS[first(g.group)]
        band!(ax, g.speed, g.lo, g.hi; color=(color, 0.18))
        lines!(ax, g.speed, g.fit; color=color, linewidth=3.4)
    end
    text!(ax, 0.97, 0.96; space=:relative, align=(:right, :top), fontsize=13,
        text=@sprintf("group p = %s\nspeed p = %s", pstr(res.p_group), pstr(res.p_speed)))
    return res
end

pstr(p) = p < 1e-3 ? @sprintf("%.0e", p) : @sprintf("%.2g", p)

function io_matrix_from_df(io, group)
    M = fill(NaN, 6, 6)
    sub = io[io.group .== group, :]
    for row in eachrow(sub)
        M[Int(row.output_index), Int(row.input_index)] = Float64(row.mean_gain)
    end
    return M
end

# Which LIMB SLOT a channel belongs to. Channels 1-3 are the first limb in the
# source export, 4-6 the second. For stroke trials the export already ordered
# them paretic-first, so the slot IS the role -- see the limb convention in
# GaitNonNormalityPaper3.jl.
function joint_side(idx)
    idx <= 3 ? "first" : "second"
end

function joint_type(idx)
    JOINT_LABELS[mod1(idx, 3)][3:end]
end

cell_string(v) = v isa Missing ? "" : String(v)

# Role of a channel's limb. This is read off the SLOT, not by comparing a
# derived L/R label against `paretic_side`.
#
# The previous version did the latter: `joint_side` called channels 1-3 "L",
# and a trial's channels counted as paretic only when its recorded paretic side
# was also "L". But the published SpeedSigs_data.csv is already paretic-first
# for every stroke trial (GS00_ProcessGaitSignatures.m:136-141 does the swap,
# and 16 of 20 stroke subjects show channels 1-3 as the lower-ROM limb
# regardless of recorded side), so that test inverted the paretic and
# non-paretic roles on all 48 R-paretic stroke trials (24 HF + 24 LF of 119).
# `paretic_side` records which anatomical side it was; it is not an instruction
# to reorder, because the reordering already happened upstream.
function side_role(side, paretic_side, group)
    group == "AB" && return side == "first" ? "right" : "left"
    return side == "first" ? "paretic" : "non-paretic"
end

function compute_or_load_trial_io(trials, meta)
    path = joinpath(RESULTS_DIR, "figure6_trial_resolvent_io_$(DMD_CONFIG_TAG).csv")
    if isfile(path) && !FORCE_RECOMPUTE
        df = CSV.read(path, DataFrame)
        if all(group -> group in unique(String.(df.group)), GROUP_ORDER) &&
                length(unique(df.trial_index)) == nrow(meta)
            println("Loading Figure 6 trial-level resolvent I/O: $path")
            return df
        end
        println("Rebuilding Figure 6 trial-level resolvent I/O because cached trials are incomplete")
    end

    rows = NamedTuple[]
    println("Fitting Figure 6 trial-level resolvent I/O...")
    for i in eachindex(trials)
        meta_row = meta[i, :]
        try
            result = fit_trial_operator(trials[i])
            f0 = trial_stride_frequency(trials[i])
            # I/O structure read from the DELAY-EMBEDDED operator (opnorm over
            # each joint's Hankel block). The explicit [angle, velocity] state
            # space was tried here and is less scale-sensitive, but it does not
            # carry the two clinical signatures: LF directional asymmetry
            # (P hip -> NP knee vs reverse) goes from p=0.031 to p=0.155, and HF
            # NP-knee self-coupling reverses sign. See the 2026-09-21 comparison.
            M = hankel_block_resolvent_io_matrix(result, f0; fs=FS)
            M === nothing && continue
            M ./= max(sum(M), eps())
            group = String(meta_row.lf_or_hf)
            paretic_side = cell_string(meta_row.paretic_side)
            for out in 1:6, input in 1:6
                output_side = joint_side(out)
                input_side = joint_side(input)
                push!(rows, (
                    trial_index=Int(meta_row.trial_index),
                    subject=String(meta_row.subject),
                    speed=Float64(meta_row.speed),
                    group=group,
                    paretic_side=paretic_side,
                    output_joint=JOINT_LABELS[out],
                    input_joint=JOINT_LABELS[input],
                    output_index=out,
                    input_index=input,
                    output_side=output_side,
                    input_side=input_side,
                    output_role=side_role(output_side, paretic_side, group),
                    input_role=side_role(input_side, paretic_side, group),
                    output_joint_type=joint_type(out),
                    input_joint_type=joint_type(input),
                    gain=M[out, input],
                ))
            end
            print(".")
        catch err
            @warn "Failed Figure 6 trial-level I/O" i err
        end
    end
    println()
    df = DataFrame(rows)
    CSV.write(path, df)
    println("Wrote Figure 6 trial-level resolvent I/O: $path")
    return df
end

function compute_or_load_trial_io_absolute(trials, meta)
    path = joinpath(RESULTS_DIR, "figure6_trial_resolvent_io_absolute_$(DMD_CONFIG_TAG).csv")
    if isfile(path) && !FORCE_RECOMPUTE
        df = CSV.read(path, DataFrame)
        if all(group -> group in unique(String.(df.group)), GROUP_ORDER) &&
                length(unique(df.trial_index)) == nrow(meta)
            println("Loading absolute trial-level resolvent I/O: $path")
            return df
        end
        println("Rebuilding absolute trial-level resolvent I/O because cached trials are incomplete")
    end

    rows = NamedTuple[]
    println("Fitting absolute (unnormalized) trial-level resolvent I/O...")
    for i in eachindex(trials)
        meta_row = meta[i, :]
        try
            result = fit_trial_operator(trials[i])
            f0 = trial_stride_frequency(trials[i])
            # I/O structure read from the DELAY-EMBEDDED operator (opnorm over
            # each joint's Hankel block). The explicit [angle, velocity] state
            # space was tried here and is less scale-sensitive, but it does not
            # carry the two clinical signatures: LF directional asymmetry
            # (P hip -> NP knee vs reverse) goes from p=0.031 to p=0.155, and HF
            # NP-knee self-coupling reverses sign. See the 2026-09-21 comparison.
            M = hankel_block_resolvent_io_matrix(result, f0; fs=FS)
            M === nothing && continue
            group = String(meta_row.lf_or_hf)
            paretic_side = cell_string(meta_row.paretic_side)
            for out in 1:6, input in 1:6
                output_side = joint_side(out)
                input_side = joint_side(input)
                push!(rows, (
                    trial_index=Int(meta_row.trial_index),
                    subject=String(meta_row.subject),
                    speed=Float64(meta_row.speed),
                    group=group,
                    paretic_side=paretic_side,
                    output_index=out,
                    input_index=input,
                    output_role=side_role(output_side, paretic_side, group),
                    input_role=side_role(input_side, paretic_side, group),
                    output_joint_type=joint_type(out),
                    input_joint_type=joint_type(input),
                    gain=M[out, input],
                ))
            end
            print(".")
        catch err
            @warn "Failed absolute trial-level I/O" i err
        end
    end
    println()
    df = DataFrame(rows)
    CSV.write(path, df)
    println("Wrote absolute trial-level resolvent I/O: $path")
    return df
end

# Do the panel G/H hip-knee pathway and paretic-asymmetry gains (absolute or per-trial
# max-normalized) predict Berg/Fugl-Meyer better than the single-metric predictors used
# in panel I?

const CLINICAL_PATHWAYS = [("hip", "hip"), ("hip", "knee"), ("knee", "hip"), ("knee", "knee")]
const CLINICAL_ASYM_JOINTS = ["hip", "knee", "ankle"]

# SIDE-RESOLVED pathways. The four pathways above average over sides -- their
# "hip_hip" pools every hip-input/hip-output cell regardless of which limb
# either end sits on -- so they cannot express a directed cross-body route.
# Paretic-hip-to-non-paretic-hip is elevated specifically in the lower-
# functioning group (1.14x able-bodied, against 0.93x in the higher-functioning
# group), the pattern a hip-hiking or circumduction strategy would produce, so
# it is carried through the clinical screen as its own candidate. Added as a
# full base -- median and slope, normalised and absolute -- like every other
# pathway, rather than only the variant that happened to score best.
const CLINICAL_SIDE_PATHWAYS = [("paretic", "hip", "non-paretic", "hip")]

side_tag(role, joint) = (role == "paretic" ? "P" : "NP") * joint
side_label(tag) = startswith(tag, "NP") ? "NP " * tag[3:end] : "P " * tag[2:end]

# The 6x6 resolvent I/O matrix is rank one (>= 99.9% of its energy in the first
# singular value, against 0.87 for a shuffled null; scripts/S56_A5_rank_one.jl),
# so every "joint j -> joint i" pathway it contains is the product of a SINK
# weight for the receiving joint and a SOURCE weight for the driving one. The
# four pathway means, the side pathway and the three asymmetries that used to be
# screened here were therefore not independent candidates -- they were products
# of two 6-vectors. This now extracts those vectors directly. In human walking
# the source vector is flat (0.15-0.20 across all six joints in every group), so
# the reception profile is what carries the group structure.
function pathway_trial_table(trial_io)
    stroke = trial_io[trial_io.group .!= "AB", :]
    rows = Dict{Symbol,Any}[]
    for sub in groupby(stroke, [:trial_index, :subject, :speed, :group])
        row = Dict{Symbol,Any}(
            :trial_index => Int(first(sub.trial_index)),
            :subject => String(first(sub.subject)),
            :speed => Float64(first(sub.speed)),
            :group => String(first(sub.group)),
        )
        n = maximum(sub.output_index)
        M = fill(NaN, n, n)
        tag = Vector{String}(undef, n)
        for r in eachrow(sub)
            M[Int(r.output_index), Int(r.input_index)] = Float64(r.gain)
            tag[Int(r.output_index)] = side_tag(String(r.output_role), String(r.output_joint_type))
        end
        if any(isnan, M)
            for j in 1:n; row[Symbol("sink_" * (isassigned(tag, j) ? tag[j] : string(j)))] = NaN; end
        else
            F = svd(M)
            u = abs.(F.U[:, 1]); u ./= sum(u)
            for j in 1:n; row[Symbol("sink_" * tag[j])] = u[j]; end
        end
        push!(rows, row)
    end
    return DataFrame(rows)
end

const CLINICAL_SINK_TAGS = ["Phip", "Pknee", "Pankle", "NPhip", "NPknee", "NPankle"]

function pathway_feature_names()
    bases = ["sink_$(t)" for t in CLINICAL_SINK_TAGS]
    return vcat(["median_$b" for b in bases], ["slope_$b" for b in bases])
end

function pathway_subject_feature_table(trial_table; prefix::AbstractString="")
    scores = load_clinical_scores()
    nrow(scores) == 0 && return DataFrame()
    stroke = innerjoin(trial_table, scores; on=:subject)
    nrow(stroke) == 0 && return DataFrame()
    rows = Dict{Symbol,Any}[]
    features = pathway_feature_names()
    for sub in groupby(stroke, [:subject, :group])
        row = Dict{Symbol,Any}(
            :subject => String(first(sub.subject)),
            :group => String(first(sub.group)),
            :berg => Float64(first(sub.berg)),
            :fugl => Float64(first(sub.fugl)),
            :median_speed => median(Float64.(sub.speed)),
        )
        for feature in features
            row[Symbol(prefix * feature)] = subject_feature_value(DataFrame(sub), feature)
        end
        push!(rows, row)
    end
    return DataFrame(rows)
end

# Merge the DMD-metric subject table (panel I's original predictors) with the panel G/H
# pathway and paretic-asymmetry gains, in both gain conventions, so the panel I "best
# predictor" can be chosen across all of them rather than just the DMD metrics.
function build_full_clinical_subject_table(metrics, trial_io_norm, trial_io_abs)
    subjects = clinical_subject_feature_table(metrics)
    nrow(subjects) == 0 && return subjects
    norm_tab = pathway_subject_feature_table(pathway_trial_table(trial_io_norm); prefix="norm_")
    abs_tab = pathway_subject_feature_table(pathway_trial_table(trial_io_abs); prefix="abs_")
    for tab in (norm_tab, abs_tab)
        nrow(tab) == 0 && continue
        extra = select(tab, Not([:group, :berg, :fugl, :median_speed]))
        subjects = leftjoin(subjects, extra; on=:subject)
        for feature_col in propertynames(extra)
            feature_col == :subject && continue
            if Missing <: eltype(subjects[!, feature_col])
                subjects[!, feature_col] = coalesce.(subjects[!, feature_col], NaN)
            end
        end
    end
    return subjects
end

function pathway_feature_names_prefixed()
    return vcat(["norm_$f" for f in pathway_feature_names()], ["abs_$f" for f in pathway_feature_names()])
end

# The exact candidate set panel I searches over. Stride-frequency features are
# excluded because they are a speed proxy and speed is the baseline being beaten.
# Exposed separately so the panel's own footnote can COUNT it rather than quote a
# hard-coded number -- the previous footnote said 44 while the search set was
# actually 42, and adding a pathway would have silently widened the gap.
function best_overall_clinical_feature_set()
    return vcat(filter(f -> !occursin("stride_frequency", f), clinical_feature_names()),
        pathway_feature_names_prefixed())
end

function best_overall_clinical_features(subjects::DataFrame)
    return best_dmd_clinical_features(subjects; features=best_overall_clinical_feature_set())
end

function predictor_type_label(feature::AbstractString)
    startswith(feature, "norm_") && return "reception_weight_normalized"
    startswith(feature, "abs_") && return "reception_weight_absolute"
    return "best_dmd_metric"
end

function screen_pathway_features(subjects::DataFrame, label::AbstractString)
    features = pathway_feature_names()
    rows = NamedTuple[]
    for outcome in ["berg", "fugl"]
        y = Float64.(subjects[!, Symbol(outcome)])
        for feature in features
            col = Symbol(feature)
            col in propertynames(subjects) || continue
            x = Float64.(subjects[!, col])
            ok = isfinite.(x) .& isfinite.(y)
            count(ok) >= 4 && std(x[ok]) > 0 && std(y[ok]) > 0 || continue
            pearson_r = cor(x[ok], y[ok])
            spearman_r = cor(rank_values(x[ok]), rank_values(y[ok]))
            push!(rows, (
                gain_type=label,
                outcome=outcome,
                feature=feature,
                n=count(ok),
                pearson_r=pearson_r,
                pearson_r2=pearson_r^2,
                spearman_r=spearman_r,
                spearman_r2=spearman_r^2,
            ))
        end
    end
    return DataFrame(rows)
end

function pathway_clinical_correlation_screen(trial_io_norm, trial_io_abs, metrics)
    subj_norm = pathway_subject_feature_table(pathway_trial_table(trial_io_norm))
    subj_abs = pathway_subject_feature_table(pathway_trial_table(trial_io_abs))
    screen = vcat(
        screen_pathway_features(subj_norm, "normalized"),
        screen_pathway_features(subj_abs, "absolute"),
    )
    CSV.write(joinpath(RESULTS_DIR, "figure6_pathway_clinical_correlation.csv"), screen)

    current_best = best_dmd_clinical_features(clinical_subject_feature_table(metrics))
    println()
    println("=== Panel I best single-metric predictors (baseline) ===")
    for row in eachrow(current_best)
        if row.selection_metric == "spearman_r2"
            @printf("  %-5s  best=%-28s  ρ=%.3f  ρ²=%.3f  n=%d\n",
                row.outcome, row.feature, row.spearman_r, row.spearman_r2, row.n)
        else
            @printf("  %-5s  best=%-28s  r=%.3f  R²=%.3f  n=%d\n",
                row.outcome, row.feature, row.pearson_r, row.pearson_r2, row.n)
        end
    end
    for outcome in ["berg", "fugl"]
        score_col = outcome == "fugl" ? :spearman_r2 : :pearson_r2
        stat_col = outcome == "fugl" ? :spearman_r : :pearson_r
        println("=== Panel G/H pathway & asymmetry screen: $(outcome) (ranked by $(score_col)) ===")
        sub = screen[screen.outcome .== outcome, :]
        ord = sortperm(sub[!, score_col]; rev=true)
        for i in ord[1:min(8, length(ord))]
            row = sub[i, :]
            @printf("  [%-10s] %-22s  r/ρ=%+.3f  R²/ρ²=%.3f  n=%d\n",
                row.gain_type, row.feature, row[stat_col], row[score_col], row.n)
        end
    end
    println("Wrote Figure 6 pathway/clinical correlation screen: $(joinpath(RESULTS_DIR, "figure6_pathway_clinical_correlation.csv"))")
    return screen
end

function mean_sem(vals)
    v = filter(isfinite, Float64.(vals))
    isempty(v) && return (NaN, NaN)
    return mean(v), length(v) > 1 ? std(v) / sqrt(length(v)) : 0.0
end

function trial_mean_rows(df, keys)
    rows = Dict{Symbol,Any}[]
    for sub in groupby(df, keys)
        m, _ = mean_sem(sub.gain)
        row = Dict{Symbol,Any}(:mean_gain => m)
        for k in keys
            row[k] = first(sub[!, k])
        end
        push!(rows, row)
    end
    return DataFrame(rows)
end

function asymmetry_summary(trial_io)
    stroke = trial_io[(trial_io.group .!= "AB") .& (trial_io.output_role .!= "unknown"), :]
    per_trial = trial_mean_rows(stroke, [:trial_index, :subject, :group, :output_joint_type, :output_role])
    rows = NamedTuple[]
    for sub in groupby(per_trial, [:trial_index, :subject, :group, :output_joint_type])
        p = sub.mean_gain[sub.output_role .== "paretic"]
        np = sub.mean_gain[sub.output_role .== "non-paretic"]
        (isempty(p) || isempty(np)) && continue
        push!(rows, (
            trial_index=Int(first(sub.trial_index)),
            subject=String(first(sub.subject)),
            group=String(first(sub.group)),
            joint_type=String(first(sub.output_joint_type)),
            asymmetry=mean(p) - mean(np),
        ))
    end
    per = DataFrame(rows)
    summary_rows = NamedTuple[]
    for sub in groupby(per, [:group, :joint_type])
        m, sem = mean_sem(sub.asymmetry)
        push!(summary_rows, (
            group=String(first(sub.group)),
            joint_type=String(first(sub.joint_type)),
            mean_asymmetry=m,
            sem_asymmetry=sem,
            n_trials=nrow(sub),
        ))
    end
    return DataFrame(summary_rows), per
end

function pathway_delta_summary(trial_io)
    pathway_specs = [("hip", "hip"), ("hip", "knee"), ("knee", "hip"), ("knee", "knee")]
    per_trial = trial_mean_rows(trial_io, [:trial_index, :subject, :group, :input_joint_type, :output_joint_type])
    rows = NamedTuple[]
    for (input_type, output_type) in pathway_specs
        base = per_trial[(per_trial.group .== "AB") .&
            (per_trial.input_joint_type .== input_type) .&
            (per_trial.output_joint_type .== output_type), :]
        base_mean, _ = mean_sem(base.mean_gain)
        for group in ["HF", "LF"]
            sub = per_trial[(per_trial.group .== group) .&
                (per_trial.input_joint_type .== input_type) .&
                (per_trial.output_joint_type .== output_type), :]
            deltas = Float64.(sub.mean_gain) .- base_mean
            m, sem = mean_sem(deltas)
            push!(rows, (
                group=group,
                pathway="$(input_type) -> $(output_type)",
                input_joint_type=input_type,
                output_joint_type=output_type,
                mean_delta=m,
                sem_delta=sem,
                n_trials=nrow(sub),
                ab_mean=base_mean,
            ))
        end
    end
    return DataFrame(rows)
end

function plot_asymmetry_summary!(ax, summary)
    joints = ["hip", "knee", "ankle"]
    offsets = Dict("HF" => -0.18, "LF" => 0.18)
    width = 0.30
    hlines!(ax, [0]; color=(:gray35, 0.65), linewidth=1.2)
    for group in ["HF", "LF"]
        xs = Float64[]
        ys = Float64[]
        lo = Float64[]
        hi = Float64[]
        for (j, joint) in enumerate(joints)
            sub = summary[(summary.group .== group) .& (summary.joint_type .== joint), :]
            isempty(sub) && continue
            m = Float64(first(sub.mean_asymmetry))
            sem = Float64(first(sub.sem_asymmetry))
            push!(xs, j + offsets[group])
            push!(ys, m)
            push!(lo, m - sem)
            push!(hi, m + sem)
        end
        barplot!(ax, xs, ys; width=width, color=(GROUP_COLORS[group], 0.72), label=group)
        rangebars!(ax, xs, lo, hi; color=:black, linewidth=1.4)
    end
    ax.xticks = (1:3, ["hip", "knee", "ankle"])
    ax.xlabel = "output joint"
    ax.ylabel = "paretic - non-paretic gain"
    axislegend(ax; position=:rb, framevisible=false)
end

function plot_pathway_delta_summary!(ax, summary)
    pathways = ["hip -> hip", "hip -> knee", "knee -> hip", "knee -> knee"]
    offsets = Dict("HF" => -0.18, "LF" => 0.18)
    width = 0.30
    hlines!(ax, [0]; color=(:gray35, 0.65), linewidth=1.2)
    for group in ["HF", "LF"]
        xs = Float64[]
        ys = Float64[]
        lo = Float64[]
        hi = Float64[]
        for (j, pathway) in enumerate(pathways)
            sub = summary[(summary.group .== group) .& (summary.pathway .== pathway), :]
            isempty(sub) && continue
            m = Float64(first(sub.mean_delta))
            sem = Float64(first(sub.sem_delta))
            push!(xs, j + offsets[group])
            push!(ys, m)
            push!(lo, m - sem)
            push!(hi, m + sem)
        end
        barplot!(ax, xs, ys; width=width, color=(GROUP_COLORS[group], 0.72), label=group)
        rangebars!(ax, xs, lo, hi; color=:black, linewidth=1.4)
    end
    ax.xticks = (1:4, replace.(pathways, " -> " => "\n-> "))
    ax.xlabel = "input -> output pathway"
    ax.ylabel = "gain difference from AB"
    axislegend(ax; position=:rt, framevisible=false)
end

function add_io_heatmap!(layout, M; title, colorrange, colormap=HEATMAP_CMAP, colorbar=false, label="")
    ax = Axis(layout[1, 1], title=title, xticklabelrotation=0.9,
        xlabel="input joint", ylabel="output joint",
        xticks=(1:6, JOINT_LABELS), yticks=(1:6, JOINT_LABELS))
    hm = heatmap!(ax, 1:6, 1:6, M'; colormap=colormap, colorrange=colorrange)
    colorbar && Colorbar(layout[1, 2], hm; label=label, width=8)
    return ax
end

function make_figure6()
    ensure_dirs!()
    trials, meta = load_human_superset_trials()
    trials, meta = select_source(trials, meta, SOURCE)
    @printf("[figure 6] source=%s -> %d trials; %s\n", SOURCE, length(trials),
        join([@sprintf("%s: %d trials / %d subj", g,
                       count(String.(meta.lf_or_hf) .== g),
                       length(unique(String.(meta.subject[String.(meta.lf_or_hf) .== g]))))
              for g in GROUP_ORDER], ", "))
    metrics = compute_or_load_metrics(trials, meta)
    # Existing cached metric tables predate the raw Q-factor column.
    :q_factor in propertynames(metrics) ||
        (metrics.q_factor = 10.0 .^ Float64.(metrics.log_q_factor))
    profiles, io = compute_or_load_profiles_and_io(trials, meta)
    trial_io = compute_or_load_trial_io(trials, meta)
    trial_io_abs = compute_or_load_trial_io_absolute(trials, meta)

    fig = Figure(size=(1680, 2184), fontsize=PAPER_FONT_SIZE)

    lmm_fits = Tuple{Symbol,Any}[]
    glA = GridLayout(fig[1, 1:3])
    panel_label!(fig, (1, 1), "A")
    axA1 = Axis(glA[1, 1], title="Non-normality", xlabel="speed", ylabel="Henrici")
    axA2 = Axis(glA[1, 2], title="Transient growth", xlabel="speed", ylabel=L"\log_{10}\max_k \|A^k\|_2")
    axA3 = Axis(glA[1, 3], title="Effective dimensionality", xlabel="speed", ylabel="participation")
    push!(lmm_fits, (:henrici, mixed_metric!(axA1, metrics, :henrici)))
    push!(lmm_fits, (:log_tgmax, mixed_metric!(axA2, metrics, :log_tgmax)))
    push!(lmm_fits, (:participation, mixed_metric!(axA3, metrics, :participation)))
    axislegend(axA1; position=:lt, framevisible=false)

    axB = Axis(fig[2, 1:2], title="Group mean resolvent profiles",
        xlabel="frequency / stride frequency", ylabel="resolvent gain", yscale=log10)
    panel_label!(fig, (2, 1), "B")
    for group in GROUP_ORDER
        sub = profiles[profiles.group .== group, :]
        isempty(sub) && continue
        color = GROUP_COLORS[group]
        ok = isfinite.(sub.mean_gain) .& isfinite.(sub.lo) .& isfinite.(sub.hi)
        band!(axB, sub.frequency_ratio[ok], sub.lo[ok], sub.hi[ok]; color=(color, 0.14))
        lines!(axB, sub.frequency_ratio[ok], sub.mean_gain[ok]; color=color, linewidth=2.5, label=group)
    end
    vlines!(axB, [1, 2, 3, 4]; color=(:gray45, 0.45), linestyle=:dot)
    xlims!(axB, 0.5, 4.5)
    axislegend(axB; position=:rt, framevisible=false)

    axC = Axis(fig[2, 3], title="Q-factor", xlabel="speed", ylabel="log₁₀ Q-factor")
    panel_label!(fig, (2, 3), "C")
    push!(lmm_fits, (:log_q_factor, mixed_metric!(axC, metrics, :log_q_factor)))
    # log scale: on the phase grid the stride peak is very sharp and Q spans
    # two decades, so a linear axis with a cap hides most of the spread.

    # D/E/F: each group's mean I/O map as the two vectors it contains. These
    # maps are rank one (>= 99.9% of energy in the first singular value against
    # 0.87 for a shuffled null), so the 6x6 heatmaps were showing the outer
    # product of a reception weight and a drive weight. The group difference
    # lives entirely in reception: able-bodied walkers are knee-symmetric,
    # while in the lower-functioning group the paretic knee drops out and the
    # non-paretic knee takes it up.
    mats = Dict(group => io_matrix_from_df(io, group) for group in GROUP_ORDER)
    vecs = Dict{String,Any}()
    for (group, M) in mats
        F = svd(M)
        u = abs.(F.U[:, 1]); v = abs.(F.V[:, 1]); sv = svdvals(M)
        vecs[group] = (sink=u ./ sum(u), source=v ./ sum(v),
            e1=sv[1]^2 / sum(abs2, sv))
    end
    vec_hi = maximum(maximum(v.sink) for v in values(vecs)) * 1.25
    for (pos, label, group) in [((3, 1), "D", "AB"), ((3, 2), "E", "HF"), ((3, 3), "F", "LF")]
        ax = Axis(fig[pos...], title="$(group) reception and drive",
            xticklabelrotation=0.9, xlabel="joint",
            ylabel=(group == "AB" ? "weight" : ""),
            xticks=(1:6, joint_labels(group)))
        panel_label!(fig, pos, label)
        barplot!(ax, (1:6) .- 0.19, vecs[group].sink;
            color=GROUP_COLORS[group], width=0.36, label="reception")
        barplot!(ax, (1:6) .+ 0.19, vecs[group].source;
            color=(GROUP_COLORS[group], 0.35), width=0.36, label="drive")
        ylims!(ax, 0, vec_hi)
        group == "AB" && axislegend(ax; position=:lt, framevisible=false, labelsize=13)
        text!(ax, 0.98, 0.97; space=:relative, align=(:right, :top), fontsize=12, color=GRAY,
            text=@sprintf("rank one: %.1f%%", 100 * vecs[group].e1))
    end

    asym_summary, asym_trials = asymmetry_summary(trial_io)
    pathway_summary = pathway_delta_summary(trial_io)
    CSV.write(joinpath(RESULTS_DIR, "figure6_paretic_asymmetry_summary.csv"), asym_summary)
    CSV.write(joinpath(RESULTS_DIR, "figure6_paretic_asymmetry_trials.csv"), asym_trials)
    CSV.write(joinpath(RESULTS_DIR, "figure6_hip_knee_pathway_delta_summary.csv"), pathway_summary)

    axG = Axis(fig[4, 1], title="Paretic reception asymmetry")
    panel_label!(fig, (4, 1), "G")
    plot_asymmetry_summary!(axG, asym_summary)

    # H was "Hip/knee pathway changes". Those directed pathway means are
    # reproduced by the rank-one reconstruction of the same maps at r = 0.999
    # (scripts/S56_A5_rank_one.jl), so they are a projection of the two vectors
    # in D/E/F rather than independent evidence for a pathway. The space now
    # carries the control that licenses the vector presentation: how rank one
    # each trial's map actually is, against a null that keeps the same gains
    # and destroys the outer-product structure.
    axH = Axis(fig[4, 2:3], title="Each map is rank one",
        xlabel="share of energy in the first singular value", ylabel="trials")
    panel_label!(fig, (4, 2), "H")
    let rng = MersenneTwister(5)
        e1_real = Float64[]; e1_null = Float64[]
        for g in groupby(trial_io, :trial_index)
            M = fill(NaN, 6, 6)
            for r in eachrow(g); M[Int(r.output_index), Int(r.input_index)] = Float64(r.gain); end
            any(isnan, M) && continue
            sv = svdvals(M); push!(e1_real, sv[1]^2 / sum(abs2, sv))
            svn = svdvals(reshape(shuffle(rng, vec(M)), 6, 6))
            push!(e1_null, svn[1]^2 / sum(abs2, svn))
        end
        if !isempty(e1_real)
            hist!(axH, e1_null; bins=range(0.4, 1.0; length=40), color=(GRAY, 0.55), label="shuffled null")
            hist!(axH, e1_real; bins=range(0.4, 1.0; length=40), color=(GROUP_COLORS["AB"], 0.85), label="real maps")
            axislegend(axH; position=:lt, framevisible=false, labelsize=13)
            text!(axH, 0.98, 0.95; space=:relative, align=(:right, :top), fontsize=13, color=GRAY,
                text=@sprintf("median %.3f vs %.3f", median(e1_real), median(e1_null)))
        end
    end

    glI = GridLayout(fig[5, 1:3])
    panel_label!(fig, (5, 1), "I")
    # Deliberately calibrated wording. The trends here are real in direction and
    # comparable in magnitude to the walking-speed baseline, but each DMD feature
    # is the best of the whole candidate screen at n = 13 (Berg) / 20 (Fugl-Meyer),
    # so the r values are optimistic and none survives correction for that screen.
    # The panel is shown to indicate which metrics are worth testing in a
    # powered sample, not to report an established association.
    Label(glI[0, 1:2],
        "Clinical associations: suggestive trends, not established effects",
        fontsize=PAPER_TITLE_SIZE, font=:bold, tellwidth=false)
    Label(glI[3, 1:2],
        @sprintf("Each DMD feature is the strongest of a %d-feature screen (n = 13 Berg, 20 Fugl-Meyer); ",
            length(best_overall_clinical_feature_set())) *
        "r values are therefore optimistic and none survives correction for the screen.",
        fontsize=15, color=GRAY, tellwidth=false, justification=:left)
    boot, clinical_summary = clinical_hierarchical_bootstrap_screen(metrics; n_boot=1500)
    selected = selected_clinical_features(clinical_summary; n_each=4)
    if nrow(selected) > 0
        CSV.write(joinpath(RESULTS_DIR, "figure6_panelI_clinical_bootstrap_selected.csv"), selected)
    end
    full_subjects = build_full_clinical_subject_table(metrics, trial_io, trial_io_abs)
    report = plot_clinical_best_associations!(glI, full_subjects, clinical_summary)
    if nrow(report) > 0
        CSV.write(joinpath(RESULTS_DIR, "figure6_panelI_clinical_best_associations.csv"), report)
        println("Figure 6 clinical best associations:")
        for row in eachrow(report)
            if row.selection_metric == "spearman_r2"
                println("  $(row.outcome) $(row.predictor_type) $(row.predictor): ρ=$(round(row.spearman_r; digits=3)), ρ²=$(round(row.spearman_r2; digits=3))")
            else
                println("  $(row.outcome) $(row.predictor_type) $(row.predictor): r=$(round(row.pearson_r; digits=3)), R²=$(round(row.pearson_r2; digits=3))")
            end
        end
    end

    pathway_clinical_correlation_screen(trial_io, trial_io_abs, metrics)

    rowgap!(fig.layout, 18)
    colgap!(fig.layout, 18)
    rowsize!(fig.layout, 5, Relative(0.26))   # panel I lost its speed row
    CSV.write(joinpath(RESULTS_DIR, "figure6_speed_mixed_models_$(DMD_CONFIG_TAG).csv"), mixed_model_summary(lmm_fits))
    savefig("Figure 6.svg", fig)
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figure6()
end

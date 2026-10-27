module GaitNonNormalityPaper3

using CairoMakie
using CSV
using DataFrames
using NPZ
using Statistics
using DMDAnalysis
using LinearAlgebra
import MixedModels
using MixedModels: MixedModel, @formula, DummyCoding, VarCorr, varest, fixef, vcov,
    coefnames, modelmatrix, raneftables, coeftable

export ROOT, FIG_DIR, RESULTS_DIR, SLIP_REPO_ROOT, PAPER_FONT_SIZE,
    PAPER_TITLE_SIZE, PAPER_LABEL_SIZE, PAPER_TICK_SIZE,
    PAPER_LEGEND_SIZE, PAPER_COLORBAR_LABEL_SIZE, PAPER_COLORBAR_TICK_SIZE,
    paper_theme,
    ensure_dirs!, savefig,
    panel_label!, load_human_superset_trials, load_human_superset_array,
    PHASE_GRID, PHASE_STEPS, PHASE_TAG, STRIDE_RESCALE, UNIFORM_STRIDE, RIDGE_ALPHA,
    fit_gait_operator, stride_rescale_trial,
    center_trial, effective_rank, finite_vals, sem, robust_cor,
    r2_identity, rmse, standardize_train, loocv_linear,
    load_clinical_scores, clinical_score_mapping_report,
    GROUP_ORDER, GROUP_COLORS, GROUP_MARKERS, GROUP_ALPHAS, GROUP_MARKERSCALE,
    JOINT_LABELS_PARETIC, JOINT_LABELS_AB, JOINT_LABELS_POOLED, joint_labels,
    LIMB_CONVENTION_NOTE,
    SOURCE_HYA, SOURCE_PSGB, SOURCE_ALL, SOURCE_TAGS, select_source,
    with_velocity, posvel_io_matrix,
    speed_mixed_model, mixed_model_lines, mixed_model_summary

const ROOT = normpath(joinpath(@__DIR__, ".."))
const FIG_DIR = joinpath(ROOT, "figures")
const RESULTS_DIR = joinpath(ROOT, "results")
const SLIP_REPO_ROOT = normpath(get(ENV, "SLIP_REPO_ROOT", joinpath(ROOT, "vendor", "SLIP.jl")))
const PAPER_FONT_SIZE = 22
const PAPER_TITLE_SIZE = 24
const PAPER_LABEL_SIZE = 22
const PAPER_TICK_SIZE = 18
const PAPER_LEGEND_SIZE = 17
const PAPER_COLORBAR_LABEL_SIZE = 18
const PAPER_COLORBAR_TICK_SIZE = 16

"""Shared AB/HF/LF group palette used by every figure that plots these groups.

Blue/gold/purple rather than the more conventional red/green so HF-vs-LF
(and AB-vs-HF) contrasts stay distinguishable under red-green color
blindness. Defined once here so the mapping from group label to color/marker
stays consistent across figures even as the analysis choices change.
"""
const GROUP_ORDER = ["AB", "HF", "LF"]
const GROUP_COLORS = Dict(
    "AB" => RGBf(0.18, 0.43, 0.72),
    "HF" => RGBf(0.86, 0.56, 0.12),
    "LF" => RGBf(0.42, 0.25, 0.62),
)
const GROUP_MARKERS = Dict("AB" => :circle, "HF" => :rect, "LF" => :utriangle)

"""Per-group scatter opacity. AB is drawn at lower opacity than HF/LF because
able-bodied trials are the densest cluster in every speed scatter and a solid
blue mass hides both its own spread and the stroke points underneath it.
`GROUP_MARKERSCALE` compensates so the fainter points stay legible."""
const GROUP_ALPHAS = Dict("AB" => 0.32, "HF" => 0.62, "LF" => 0.62)
const GROUP_MARKERSCALE = Dict("AB" => 1.35, "HF" => 1.0, "LF" => 1.0)

"""
Limb convention for the 6 kinematic channels of the human gait superset.

Channels are ordered [limb-A hip, knee, ankle, limb-B hip, knee, ankle], and
which anatomical limb "limb A" is depends on the group, because the source
export already reorganized the stroke data:

  - HF / LF (stroke): channels 1-3 are the PARETIC limb, 4-6 non-paretic.
    `MATLAB/Preprocessing/GS00_ProcessGaitSignatures.m:136-141` in
    bermanlabemory/GaitSignatures_PostStrokeGaitBiomechanics documents the
    swap ("Left = paretic -> Swap so it goes (left/paretic)-then-(right/
    non-paretic)"), and it is confirmed in the published SpeedSigs_data.csv:
    16 of 20 stroke subjects show channels 1-3 as the lower-ROM limb
    regardless of their recorded paretic side (e.g. ST06 knee ROM 19.2 deg
    vs 70.8), and the 4 exceptions are all near-symmetric mild cases.
    `SpeedSigs_isParetic.csv` records which anatomical side that was; it is
    NOT an instruction to reorder, the reordering already happened.
  - AB: no swap is applied, so channels stay in the source's native order,
    which the same MATLAB implies is RIGHT-then-LEFT. The able-bodied limbs
    are symmetric, so this cannot be confirmed from the data itself and is
    taken as an assumption.

Use `joint_labels(group)` for anything group-specific and
`JOINT_LABELS_POOLED` where trials from several groups share one axis.
"""
const JOINT_LABELS_PARETIC = ["P hip", "P knee", "P ankle", "NP hip", "NP knee", "NP ankle"]
const JOINT_LABELS_AB = ["R hip", "R knee", "R ankle", "L hip", "L knee", "L ankle"]
# Shared axes pool AB (R/L) with stroke (P/NP) trials in the same columns; the
# stroke reading is the clinically meaningful one, so it names the axis and the
# able-bodied caveat goes in the caption.
const JOINT_LABELS_POOLED = JOINT_LABELS_PARETIC
joint_labels(group) = group == "AB" ? JOINT_LABELS_AB : JOINT_LABELS_PARETIC
const LIMB_CONVENTION_NOTE =
    "channels 1-3 = paretic limb (HF/LF) or right limb (AB); see GS00_ProcessGaitSignatures.m"

"""
Source datasets in the human gait superset, and the one-figure-one-dataset rule.

The superset pools two studies whose kinematics were preprocessed differently.
`S13_dataset_provenance_audit.jl` established that the difference is NOT
biomechanical -- every standard gait metric overlaps between them -- but that
the PostStroke study's signals were smoothed onto a low-dimensional basis
(locally deterministic to ~1e-6 degrees, far below any mocap system's physical
resolution), which collapses their delay-embedded singular spectrum. At the
ranks this paper fits, the two datasets therefore produce systematically
different operators: the HYA-vs-PostStroke gap in log transient growth is
Cohen's d = -0.05 at rank 10 but +3.40 at rank 70.

Consequence: rank-sensitive operator metrics are only comparable WITHIN a
dataset. Figures that compare trials must therefore draw those trials from a
single source. Figure 5 (able-bodied baseline) uses HYA only; Figures 6 and 7
(stroke) use the PostStroke study only, so their able-bodied controls come from
the same acquisition and preprocessing pipeline as the stroke trials they are
contrasted against.
"""
const SOURCE_HYA = "GaitSignatures_HealthyYoungAdultStudy"
const SOURCE_PSGB = "GaitSignatures_PostStrokeGaitBiomechanics"

# Deliberate opt-out, for the one analysis whose subject IS the cross-dataset
# difference (S13_dataset_provenance_audit.jl). Every other figure should name a
# single source; pooling is what this constant exists to make explicit rather
# than accidental.
const SOURCE_ALL = "all"

# Short slugs for cache filenames, so a dataset-restricted run can never load a
# cache built from the pooled superset (the failure mode would be silent).
const SOURCE_TAGS = Dict(SOURCE_HYA => "hya", SOURCE_PSGB => "psgb", SOURCE_ALL => "all")

"""
    select_source(trials, meta, source)

Restrict a loaded superset to one source dataset. Returns `(trials, meta)` with
`meta` re-indexed from 1, and errors rather than returning an empty selection so
a typo in `source` cannot silently produce an empty figure. Pass `SOURCE_ALL` to
keep every trial (see the constant's docstring for when that is appropriate).
"""
function select_source(trials, meta, source::AbstractString)
    source == SOURCE_ALL && return trials, meta
    idx = findall(String.(meta.source) .== source)
    isempty(idx) && error("no trials for source \"$source\"; available: " *
                          join(sort(unique(String.(meta.source))), ", ") *
                          " (or \"$SOURCE_ALL\" to pool)")
    return trials[idx], meta[idx, :]
end


"""Shared Makie styling for manuscript figures.

Axis text is intentionally bold so labels remain legible after figures are
scaled to manuscript-column width.  Individual figures can still override a
property when a compact nested panel needs it.
"""
function paper_theme()
    Theme(
        fontsize=PAPER_FONT_SIZE,
        Axis=(
            titlesize=PAPER_TITLE_SIZE,
            titlefont=:bold,
            xlabelsize=PAPER_LABEL_SIZE,
            ylabelsize=PAPER_LABEL_SIZE,
            xlabelfont=:bold,
            ylabelfont=:bold,
            xticklabelsize=PAPER_TICK_SIZE,
            yticklabelsize=PAPER_TICK_SIZE,
            xticklabelfont=:bold,
            yticklabelfont=:bold,
        ),
        Colorbar=(
            labelsize=PAPER_COLORBAR_LABEL_SIZE,
            ticklabelsize=PAPER_COLORBAR_TICK_SIZE,
            labelfont=:bold,
            ticklabelfont=:bold,
        ),
        Legend=(
            labelsize=PAPER_LEGEND_SIZE,
            titlefont=:bold,
        ),
    )
end

set_theme!(paper_theme())

function ensure_dirs!()
    mkpath(FIG_DIR)
    mkpath(RESULTS_DIR)
    return nothing
end

function savefig(name::AbstractString, fig)
    ensure_dirs!()
    path = joinpath(FIG_DIR, name)
    CairoMakie.save(path, fig)
    println("Saved: ", path)
    return path
end

function panel_label!(fig, pos, label; fontsize=32, padding=(0, 8, 0, 6))
    Label(fig[pos..., TopLeft()], label;
        fontsize=fontsize, font=:bold, halign=:right, valign=:bottom,
        padding=padding, tellwidth=false, tellheight=false)
end

function load_human_superset_array()
    data = npzread(joinpath(ROOT, "data", "human_gait_superset.npy"))
    meta = CSV.read(joinpath(ROOT, "data", "human_gait_superset_metadata.csv"), DataFrame)
    return data, meta
end

"""
Phase-grid analysis switch (`GAIT_PHASE_GRID=true`).

When on, every human trial is resampled onto a uniform gait-phase grid of
`PHASE_STEPS` samples per stride (`DMDAnalysis.phase_normalize_trial`), so the
operator advances a fixed fraction of a stride per step for every walker
regardless of cadence. With the scripts' FS = 100 this makes the stride
frequency exactly FS / PHASE_STEPS = 1, so one-cycle horizons, resolvent
evaluation at f0 and Q need no per-trial frequency estimate. Rank is chosen by
BIC on one-step prediction of the observed angles (k = r^2, n = time steps;
`scripts/S59_rank_selection.jl`), which selects rank 20 on this grid.
"""
const PHASE_GRID = lowercase(get(ENV, "GAIT_PHASE_GRID", "false")) in ("1", "true", "yes")
const PHASE_STEPS = parse(Int, get(ENV, "GAIT_PHASE_STEPS", "100"))

"""
Stride-rescaling switch (`GAIT_STRIDE_RESCALE=true`).

Each trial is uniformly resampled in time so that its MEAN stride spans
`PHASE_STEPS` samples. Unlike the phase grid, nothing is warped within a stride
(stance/swing timing is kept) and no instantaneous phase estimate is needed: the
stride frequency comes from the total phase advance over the trial, which is
robust even where the Hilbert clock briefly runs backwards. Removes the cadence
dependence of a window fixed in samples (see the clock/cadence test, 2026-10-07).
"""
const STRIDE_RESCALE = lowercase(get(ENV, "GAIT_STRIDE_RESCALE", "false")) in ("1", "true", "yes")
const UNIFORM_STRIDE = PHASE_GRID || STRIDE_RESCALE

"""
Ridge strength for operator fits (`GAIT_RIDGE_ALPHA`, default 0 = plain exact DMD).

A = U_r' Y V_r S_r (S_r^2 + alpha s_1^2)^-1: Tikhonov regularisation in POD
coordinates, scaled by the leading singular value so alpha is dimensionless.
alpha = 1e-4 acts as a soft cutoff near s_j / s_1 ~ 1e-2, which makes transient
growth independent of the retained rank above ~40 while matching or improving
held-out forecasts.
"""
const RIDGE_ALPHA = parse(Float64, get(ENV, "GAIT_RIDGE_ALPHA", "0"))

const PHASE_TAG = (PHASE_GRID ? "_phase$(PHASE_STEPS)" : STRIDE_RESCALE ? "_stride$(PHASE_STEPS)" : "") *
                  (RIDGE_ALPHA > 0 ? "_ridge$(RIDGE_ALPHA)" : "")

"Mean stride frequency (cycles per sample) from the total Hilbert-phase advance of the trial."
function mean_stride_cycles_per_sample(X::AbstractMatrix)
    sig = DMDAnalysis.pca_phase_signal(Matrix{Float64}(X))
    phi = DMDAnalysis.unwrap_phase(angle.(DMDAnalysis.analytic_signal(sig .- mean(sig))))
    adv = abs(phi[end] - phi[1]) / (2pi)
    return adv / (length(phi) - 1)
end

"Uniformly resample `X` (linear interpolation) so one mean stride spans `steps` samples."
function stride_rescale_trial(X::AbstractMatrix; steps::Int=PHASE_STEPS)
    c = mean_stride_cycles_per_sample(X)
    (isfinite(c) && c > 0) || throw(ArgumentError("no stride rhythm found"))
    T = size(X, 1)
    n = max(2, round(Int, (T - 1) * c * steps) + 1)
    out = zeros(n, size(X, 2))
    for (k, t) in enumerate(range(1, T; length=n))
        j = clamp(floor(Int, t), 1, T - 1); a = t - j
        out[k, :] = (1 - a) .* X[j, :] .+ a .* X[j+1, :]
    end
    return out
end

"""
    fit_gait_operator(model, X, h)

`fit(model, X, h)` from DMDAnalysis when `RIDGE_ALPHA == 0`; otherwise the same
exact-DMD fit with the ridge filter above, returned as a `DMDResult` so every
downstream function (rollout, transient growth, resolvent) is unchanged.
"""
function fit_gait_operator(model::DMDAnalysis.DMD, X, h::DMDAnalysis.HankelGenerator; alpha::Real=RIDGE_ALPHA)
    alpha > 0 || return DMDAnalysis.fit(model, X, h)
    hr = DMDAnalysis.transform(h, X)
    Xm, Ym = DMDAnalysis.pooled_xy(hr)
    F = svd(Xm)
    r = DMDAnalysis._select_rank(model.rank, F.S, Xm)
    Ur = F.U[:, 1:r]; Sr = F.S[1:r]; Vr = F.V[:, 1:r]
    A = Ur' * Ym * Vr * Diagonal(Sr ./ (Sr .^ 2 .+ alpha * F.S[1]^2))
    model.stabilize && (A = DMDAnalysis._stabilize_schur(A))
    A = real.(A)
    return DMDAnalysis.DMDResult(A, Ur, r, Float64.(F.S), ComplexF64.(eigvals(A)), model,
        hr.generator, hr.n_features, size(Xm, 2))
end

function load_human_superset_trials()
    data, meta = load_human_superset_array()
    trials = [Matrix{Float64}(@view data[i, :, :]) for i in axes(data, 1)]
    if STRIDE_RESCALE
        keep = Int[]; out = Matrix{Float64}[]
        for (i, X) in enumerate(trials)
            Z = try stride_rescale_trial(X) catch; nothing end
            Z === nothing && continue
            push!(out, Z); push!(keep, i)
        end
        length(keep) < length(trials) &&
            @warn "stride rescale: dropped $(length(trials) - length(keep)) trial(s) without a stride rhythm"
        return out, meta[keep, :]
    end
    PHASE_GRID || return trials, meta
    keep = Int[]
    out = Matrix{Float64}[]
    for (i, X) in enumerate(trials)
        Z = try
            Matrix{Float64}(DMDAnalysis.phase_normalize_trial(X; steps=PHASE_STEPS))
        catch
            nothing
        end
        Z === nothing && continue
        push!(out, Z); push!(keep, i)
    end
    length(keep) < length(trials) &&
        @warn "phase grid: dropped $(length(trials) - length(keep)) trial(s) without a full cycle"
    return out, meta[keep, :]
end

center_trial(X) = Matrix{Float64}(X .- mean(X; dims=1))

"""
    with_velocity(X, fs)

Append a centred finite-difference derivative of every column of `X`, giving the
explicit `[angle, angular velocity]` state. Endpoints repeat their neighbour so
the result keeps the same number of rows.
"""
function with_velocity(X::AbstractMatrix, fs::Real)
    A = Matrix{Float64}(X)
    n, m = size(A)
    n >= 3 || return hcat(A, zeros(n, m))
    V = zeros(n, m)
    V[2:end-1, :] = (A[3:end, :] .- A[1:end-2, :]) .* (fs / 2)
    V[1, :] = V[2, :]
    V[end, :] = V[end-1, :]
    return hcat(A, V)
end

"""
    posvel_io_matrix(trial, f0; fs, rank_mult=1)

Resolvent input/output map over joints, fitted in the explicit
`[angle, angular velocity]` state space with NO delay embedding, grouping each
joint's position and velocity channels together.

Preferred over a delay embedding of angles alone for I/O structure: on the human
cohorts the group trial-mean map is more reproducible (split-half 0.95-0.999 vs
0.82-0.996) and far less sensitive to channel scaling (raw-vs-z-scored agreement
0.77-0.84 vs 0.53-0.70 for AB and HF). Channels are deliberately NOT z-scored --
the relative strength of flow between joints in their own units is the quantity
of interest.
"""
function posvel_io_matrix(trial::AbstractMatrix, f0::Real; fs::Real, stabilize::Bool=true)
    isfinite(f0) && f0 > 0 || return nothing
    n_joints = size(trial, 2)
    Y = with_velocity(trial, fs)
    Y = Y .- mean(Y; dims=1)
    result = try
        DMDAnalysis.fit(DMDAnalysis.DMD(rank=DMDAnalysis.FixedRank(2n_joints), stabilize=stabilize),
            [Matrix{Float64}(Y)],
            DMDAnalysis.HankelGenerator(n_delays=1, delay_interval=1))
    catch
        return nothing
    end
    return DMDAnalysis.resolvent_io_matrix(result, f0; fs=fs,
        groups=[[j, j + n_joints] for j in 1:n_joints])
end


function effective_rank(v)
    vals = Float64.(v)
    return sum(vals)^2 / max(sum(abs2, vals), eps())
end

finite_vals(v) = Float64[x for x in Float64.(v) if isfinite(x)]

function sem(v)
    vals = finite_vals(v)
    length(vals) <= 1 && return 0.0
    return std(vals) / sqrt(length(vals))
end

function robust_cor(x, y)
    xx = Float64.(x)
    yy = Float64.(y)
    ok = isfinite.(xx) .& isfinite.(yy)
    count(ok) >= 3 && std(xx[ok]) > 0 && std(yy[ok]) > 0 || return NaN
    return cor(xx[ok], yy[ok])
end

r2_identity(pred, truth) = 1 - sum(abs2, pred .- truth) / sum(abs2, truth .- mean(truth))
rmse(pred, truth) = sqrt(mean(abs2, pred .- truth))

function standardize_train(X)
    mu = vec(mean(X; dims=1))
    sd = vec(std(X; dims=1))
    sd .= max.(sd, eps())
    return mu, sd
end

function loocv_linear(Xall, yall)
    pred = fill(NaN, length(yall))
    for i in eachindex(yall)
        train = setdiff(eachindex(yall), [i])
        mu, sd = standardize_train(Xall[train, :])
        Xtr = (Xall[train, :] .- mu') ./ sd'
        Xte = (reshape(Xall[i, :], 1, :) .- mu') ./ sd'
        beta = hcat(ones(length(train)), Xtr) \ yall[train]
        pred[i] = only(hcat(ones(1), Xte) * beta)
    end
    return pred
end

function try_parse_float(v)
    v isa Missing && return NaN
    v isa Real && return Float64(v)
    s = strip(String(v))
    isempty(s) && return NaN
    s in ("-", "NA", "NaN", "nan") && return NaN
    try
        return parse(Float64, s)
    catch
        return NaN
    end
end

score_column(df::DataFrame, col::Symbol) =
    col in propertynames(df) ? [try_parse_float(v) for v in df[!, col]] : fill(NaN, nrow(df))

function normalize_side_label(v)
    v isa Missing && return ""
    s = uppercase(strip(String(v)))
    isempty(s) && return ""
    startswith(s, "R") && return "R"
    startswith(s, "L") && return "L"
    return s
end

function load_combined_clinical_scores(path)
    raw = CSV.read(path, DataFrame)
    out = DataFrame(subject=strip.(String.(raw.subject)))
    out.berg = score_column(raw, :berg)
    out.fugl = score_column(raw, :lower_extremity_fugl_meyer)
    out.fugl_legacy = score_column(raw, :legacy_fugl_meyer)
    out.age_years = score_column(raw, :age_years)
    out.months_post_stroke = score_column(raw, :months_post_stroke)
    out.ss_overground_speed_mps = score_column(raw, :ss_overground_speed_mps)
    out.ss_treadmill_speed_mps = score_column(raw, :ss_treadmill_speed_mps)
    out.fastest_treadmill_speed_mps = score_column(raw, :fastest_treadmill_speed_mps)
    out.sex = :sex in propertynames(raw) ? String.(raw.sex) : fill("", nrow(raw))
    out.docx_paretic_side = :docx_paretic_side in propertynames(raw) ?
        normalize_side_label.(raw.docx_paretic_side) : fill("", nrow(raw))
    out.score_source = :score_source in propertynames(raw) ? String.(raw.score_source) : fill("", nrow(raw))
    out.notes = :notes in propertynames(raw) ? String.(raw.notes) : fill("", nrow(raw))
    out.fugl = ifelse.(isfinite.(out.fugl), out.fugl, out.fugl_legacy)
    keep = isfinite.(out.berg) .| isfinite.(out.fugl)
    return out[keep, :]
end

function load_legacy_clinical_scores(path)
    raw = CSV.read(path, DataFrame; header=false)
    ncol(raw) < 3 && return DataFrame(subject=String[], berg=Float64[], fugl=Float64[])
    rename!(raw, [:subject_raw, :fugl_raw, :berg_raw])
    scores = DataFrame(
        subject=strip.(String.(raw.subject_raw)),
        berg=[try_parse_float(v) for v in raw.berg_raw],
        fugl=[try_parse_float(v) for v in raw.fugl_raw],
        fugl_legacy=[try_parse_float(v) for v in raw.fugl_raw],
        score_source=fill("legacy_csv", nrow(raw)),
    )
    keep = isfinite.(scores.berg) .| isfinite.(scores.fugl)
    return scores[keep, :]
end

function load_clinical_scores()
    combined_path = joinpath(ROOT, "data", "clinical_scores_combined.csv")
    isfile(combined_path) && return load_combined_clinical_scores(combined_path)

    score_paths = [
        joinpath(ROOT, "data", "subject_scores.csv"),
        normpath(joinpath(ROOT, "..", "Gait_NonNormality_Paper", "data", "subject_scores.csv")),
        expanduser("~/Documents/Synology_local/Python/Gait-Signatures/data/subject_scores.csv"),
        expanduser("~/Synology/Python/Gait-Signatures/data/subject_scores.csv"),
    ]
    idx = findfirst(isfile, score_paths)
    isnothing(idx) && return DataFrame(subject=String[], berg=Float64[], fugl=Float64[])
    return load_legacy_clinical_scores(score_paths[idx])
end

function clinical_score_mapping_report(meta_path=joinpath(ROOT, "data", "human_gait_superset_metadata.csv"))
    scores = load_clinical_scores()
    isfile(meta_path) || return DataFrame()
    meta = CSV.read(meta_path, DataFrame)
    subjects = unique(DataFrame(subject=String.(meta.subject),
        group=String.(meta.lf_or_hf),
        metadata_paretic_side=[ismissing(v) ? "" : String(v) for v in meta.paretic_side]))
    report = outerjoin(subjects, scores; on=:subject)
    report.has_gait_data = .!ismissing.(report.group)
    report.has_docx_or_legacy_score = isfinite.(coalesce.(report.berg, NaN)) .|
        isfinite.(coalesce.(report.fugl, NaN))
    if :docx_paretic_side in propertynames(report)
        report.side_matches_metadata = [ismissing(row.group) || row.group == "AB" ||
            ismissing(row.docx_paretic_side) || row.docx_paretic_side == "" ||
            ismissing(row.metadata_paretic_side) || row.metadata_paretic_side == "" ||
            row.docx_paretic_side == row.metadata_paretic_side for row in eachrow(report)]
    end
    return report
end


# ---------------------------------------------------------------------------
# Speed mixed-effects models (Figures 5 and 6)
#
# Speed is a dynamical factor, not a nuisance: each measure is modelled as
#   y ~ [group *] speed + (1 + speed | subject)
# by maximum likelihood, with speed entering as log2(speed), centred, so slopes
# are changes per doubling of speed. Log speed fits the young-adult data far
# better than linear speed (AIC lower by 8-101 across measures; the measures
# flatten above ~100 cm/s) and the stroke data equally well. Variance is
# partitioned Nakagawa-style into fixed effects (var of X*beta), individuals
# (mean over trials of z' Sigma z, z = [1, speed]) and residual. With a group
# term, the fixed share is split into parts unique to group, unique to speed,
# and shared (the drop in marginal R^2 when each term is removed).
# ---------------------------------------------------------------------------
function speed_mixed_model(df::AbstractDataFrame, col::Symbol; grouped::Bool=true)
    d = DataFrame(y=Float64.(df[!, col]), speed=Float64.(df.speed),
        subject=String.(df.subject), group=String.(df.group))
    d = d[isfinite.(d.y) .& isfinite.(d.speed), :]
    center = mean(log2.(d.speed))
    d.spd = log2.(d.speed) .- center
    grouped &= length(unique(d.group)) > 1
    contr = grouped ? Dict(:group => DummyCoding(base=first(filter(in(unique(d.group)), GROUP_ORDER)))) : Dict{Symbol,Any}()
    fitm(f) = MixedModels.fit(MixedModel, f, d; contrasts=contr, REML=false, progress=false)
    fixvar(m) = var(modelmatrix(m) * fixef(m))
    function ranvar(m)
        s = VarCorr(m).σρ.subject
        sd = collect(s.σ); rho = (isempty(s.ρ) || !isfinite(s.ρ[1])) ? 0.0 : s.ρ[1]
        length(sd) == 1 && return sd[1]^2           # intercept-only comparison models
        S = [sd[1]^2 rho*sd[1]*sd[2]; rho*sd[1]*sd[2] sd[2]^2]
        Z = hcat(ones(nrow(d)), d.spd)
        mean(sum((Z * S) .* Z; dims=2))
    end
    r2m(m) = (v = fixvar(m); v / (v + ranvar(m) + varest(m)))
    lrt_p(small, big) = MixedModels.likelihoodratiotest(small, big).pval[2]
    if grouped
        full = fitm(@formula(y ~ 1 + group * spd + (1 + spd | subject)))
        nogrp = fitm(@formula(y ~ 1 + spd + (1 + spd | subject)))
        noint = fitm(@formula(y ~ 1 + group + spd + (1 + spd | subject)))
        nospd = fitm(@formula(y ~ 1 + group + (1 | subject)))
        p_group, p_speed, p_inter = lrt_p(nogrp, full), lrt_p(nospd, full), lrt_p(noint, full)
        uniq_group = r2m(full) - r2m(nogrp); uniq_speed = r2m(full) - r2m(nospd)
    else
        full = fitm(@formula(y ~ 1 + spd + (1 + spd | subject)))
        nospd = fitm(@formula(y ~ 1 + (1 | subject)))
        noint = nothing
        p_group, p_inter = NaN, NaN
        p_speed = lrt_p(nospd, full)
        uniq_group = 0.0; uniq_speed = r2m(full)
    end
    vf, vr, ve = fixvar(full), ranvar(full), varest(full); tot = vf + vr + ve
    return (; model=full, additive=noint, data=d, center, grouped,
        p_group, p_speed, p_inter,
        share_fixed=vf / tot, share_group=uniq_group, share_speed=uniq_speed,
        share_shared=vf / tot - uniq_group - uniq_speed,
        share_subject=vr / tot, share_residual=ve / tot)
end

# Fixed-effect design row for one group at one centred speed, in the order of
# the model's coefficients.
function _design_row(m, group, spd)
    map(coefnames(m)) do name
        name == "(Intercept)" && return 1.0
        name == "spd" && return spd
        g = match(r"^group: (\w+)$", name)
        g !== nothing && return Float64(g[1] == group)
        gi = match(r"^group: (\w+) & spd$", name)
        gi !== nothing && return Float64(gi[1] == group) * spd
        error("unexpected coefficient $name")
    end
end

"""Fitted lines from `speed_mixed_model`: a population line per group with a
95% Wald band, drawn only over that group's recorded speeds, and one line per
subject (fixed effects plus that subject's random intercept and slope), drawn
over that subject's recorded speeds."""
function mixed_model_lines(res; n=60)
    m, d = res.model, res.data
    V = vcov(m); beta = fixef(m)
    groups = res.grouped ? [g for g in GROUP_ORDER if g in d.group] : [first(d.group)]
    pop = NamedTuple[]
    for g in groups
        sp = res.grouped ? d.speed[d.group .== g] : d.speed
        for s in range(minimum(sp), maximum(sp); length=n)
            x = _design_row(m, g, log2(s) - res.center)
            yhat = dot(x, beta); se = sqrt(max(dot(x, V * x), 0.0))
            push!(pop, (group=g, speed=s, fit=yhat, lo=yhat - 1.96se, hi=yhat + 1.96se))
        end
    end
    re = DataFrame(only(raneftables(m)))
    subj = NamedTuple[]
    for r in eachrow(re)
        sub = d[d.subject .== r.subject, :]
        g = first(sub.group)
        for s in range(minimum(sub.speed), maximum(sub.speed); length=max(2, n ÷ 6))
            spd = log2(s) - res.center
            yhat = dot(_design_row(m, res.grouped ? g : first(d.group), spd), beta) + r[2] + r[3] * spd
            push!(subj, (subject=String(r.subject), group=g, speed=s, fit=yhat))
        end
    end
    return DataFrame(pop), DataFrame(subj)
end


"""One row per coefficient of each fitted model (full and, if grouped, additive)
with the model-level p values and variance shares repeated on every row."""
function mixed_model_summary(fits)
    rows = NamedTuple[]
    for (metric, res) in fits
        for (label, m) in (("full", res.model), ("additive", res.additive))
            m === nothing && continue
            ct = coeftable(m)
            for (name, b, se, p) in zip(coefnames(m), fixef(m), ct.cols[2], ct.cols[4])
                push!(rows, (metric=String(metric), model=label, coef=name, estimate=b, se=se, p=p,
                    p_group=res.p_group, p_speed=res.p_speed, p_interaction=res.p_inter,
                    share_fixed=res.share_fixed, share_group=res.share_group, share_speed=res.share_speed,
                    share_shared=res.share_shared, share_subject=res.share_subject,
                    share_residual=res.share_residual, speed_center=res.center))
            end
        end
    end
    return DataFrame(rows)
end

end

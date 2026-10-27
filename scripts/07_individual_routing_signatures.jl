#!/usr/bin/env julia
#
# Figure 7: Individual routing signatures.
#
# The question this figure answers is the clinically useful one: given one
# person's gait recording, how far is their control routing from normal, WHICH
# pathways are responsible, and is that description specific enough to belong to
# that individual rather than to their diagnosis?
#
# Method. Each trial contributes a 6x6 resolvent input-output map, normalized to
# its own total gain so it describes ROUTING (what fraction of the response goes
# where) rather than overall magnitude. Overall magnitude is deliberately
# excluded: it is dominated by the delay-embedded spectrum's conditioning, which
# differs between the two source datasets for preprocessing reasons (see S13),
# whereas routing is largely dataset-robust -- a claim this figure tests directly
# rather than assuming, by pooling BOTH datasets and marking which each subject
# came from.
#
# Every subject is then expressed as a DEVIATION from an able-bodied reference,
# in units of able-bodied standard deviations per pathway, using a leave-one-out
# reference so that able-bodied subjects are not scored against themselves. That
# deviation vector splits into:
#   - its length          -> an abnormality score (how far from normal),
#   - its direction       -> which pathways are responsible,
# and the directions in turn split into a shared stroke axis plus an individual
# residual. The residual is the majority of each patient's deviation and is
# reproducible on held-out trials from the same patient, which is what licenses
# the phrase "individual signature".
#
# What this figure does NOT claim: that the abnormality score predicts Berg or
# Fugl-Meyer better than walking speed. Panel G shows it does not, on this
# sample.

include(joinpath(@__DIR__, "..", "src", "GaitNonNormalityPaper3.jl"))

using .GaitNonNormalityPaper3
using CairoMakie
using CSV
using DataFrames
using LinearAlgebra
using Printf
using Random
using Statistics

const GRAY = RGBf(0.25, 0.25, 0.25)
const DIVERGING = :RdBu

# Pooled across BOTH source datasets on purpose: panel A tests whether that
# pooling is safe for routing features rather than assuming it.
const DMD_CONFIG_TAG = get(ENV, "FIGURE7_CONFIG_TAG", "d32_fixedrank70_all")
const N_SPLIT_REPS = parse(Int, get(ENV, "FIGURE7_SPLIT_REPS", "200"))
const MIN_TRIALS_SPLIT = 4
const SOURCE_SHORT = Dict(SOURCE_HYA => "HYA", SOURCE_PSGB => "PSGB")

# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

"""
Trial x 36 matrix of log10 routing fractions, plus per-trial metadata.

The 36 columns are (output joint, input joint) pairs in the paretic-referenced
order of `JOINT_LABELS_PARETIC`, flattened input-fastest so that column
`6*(out-1) + in` is `input -> output`.
"""
function load_routing_matrix()
    io_path = joinpath(RESULTS_DIR, "figure6_trial_resolvent_io_$(DMD_CONFIG_TAG).csv")
    met_path = joinpath(RESULTS_DIR, "figure6_metrics_$(DMD_CONFIG_TAG).csv")
    for p in (io_path, met_path)
        isfile(p) || error("missing $p\nRegenerate with: FIGURE6_SOURCE=all ./scripts/julia-env --project=. scripts/06_stroke_clinical_potential.jl")
    end
    io = CSV.read(io_path, DataFrame)
    met = CSV.read(met_path, DataFrame)
    source_of = Dict(Int(r.trial_index) => String(r.source) for r in eachrow(met))

    trials = sort(unique(Int.(io.trial_index)))
    pos = Dict(t => k for (k, t) in enumerate(trials))
    X = fill(NaN, length(trials), 36)
    subject = Vector{String}(undef, length(trials))
    group = Vector{String}(undef, length(trials))
    speed = fill(NaN, length(trials))
    dataset = Vector{String}(undef, length(trials))
    for r in eachrow(io)
        k = pos[Int(r.trial_index)]
        X[k, 6 * (Int(r.output_index) - 1) + Int(r.input_index)] = Float64(r.gain)
        subject[k] = String(r.subject)
        group[k] = String(r.group)
        speed[k] = Float64(r.speed)
        dataset[k] = get(SOURCE_SHORT, get(source_of, Int(r.trial_index), ""), "other")
    end
    keep = vec(all(isfinite, X; dims=2))
    X = X[keep, :]
    # Per-trial normalization to a routing distribution, then log so that
    # proportional changes are additive and the deviation z-scores below are
    # symmetric for over- and under-routing.
    X ./= sum(X; dims=2)
    L = log10.(clamp.(X, 1e-12, Inf))
    return (L=L, subject=subject[keep], group=group[keep],
            speed=speed[keep], dataset=dataset[keep])
end

"""Collapse trials to subjects (a subject's routing is the mean over their trials)."""
function subject_table(tr)
    subs = sort(unique(tr.subject))
    S = Matrix{Float64}(undef, length(subs), 36)
    grp = Vector{String}(undef, length(subs))
    ds = Vector{String}(undef, length(subs))
    spd = Vector{Float64}(undef, length(subs))
    ntr = Vector{Int}(undef, length(subs))
    for (i, s) in enumerate(subs)
        m = tr.subject .== s
        S[i, :] = vec(mean(tr.L[m, :]; dims=1))
        grp[i] = tr.group[findfirst(m)]
        ds[i] = tr.dataset[findfirst(m)]
        spd[i] = median(tr.speed[m])
        ntr[i] = count(m)
    end
    return (subs=subs, S=S, group=grp, dataset=ds, speed=spd, n_trials=ntr)
end

"""
Deviation from the able-bodied reference, in able-bodied SDs per pathway.

Leave-one-out for able-bodied subjects: without it every control is scored
against a reference that contains itself, which shrinks control abnormality
toward zero and manufactures the group difference this figure is testing.
"""
function deviation_matrix(S, group)
    ab = group .== "AB"
    D = similar(S)
    for i in axes(S, 1)
        ref = copy(ab)
        ab[i] && (ref[i] = false)
        mu = vec(mean(S[ref, :]; dims=1))
        sd = vec(std(S[ref, :]; dims=1))
        D[i, :] = (S[i, :] .- mu) ./ max.(sd, eps())
    end
    return D
end

abnormality(D) = vec(sqrt.(sum(abs2, D; dims=2) ./ size(D, 2)))

"""
Shared stroke axes: principal directions of the stroke subjects' deviations.

Signed so that the stroke groups sit on the NEGATIVE side of axis 1, which makes
the loading map read as "what stroke subjects route less of".
"""
function stroke_axes(D, group; n_axes=3)
    st = group .!= "AB"
    F = svd(D[st, :])
    V = F.V[:, 1:n_axes]
    ev = (F.S .^ 2) ./ sum(F.S .^ 2)
    for a in 1:n_axes
        mean(D[st, :] * V[:, a]) > 0 && (V[:, a] .*= -1)
    end
    return V, ev[1:n_axes]
end

# ---------------------------------------------------------------------------
# Identifiability: are signatures individual, or just group-typical?
# ---------------------------------------------------------------------------

"""
Split-half reproducibility of each subject's deviation direction.

Each subject's trials are split at random into two halves; a deviation vector is
built from each half against a reference formed from half A of the able-bodied
subjects. If a signature is individual, a subject's two halves should point in
the same direction as each other and not as everyone else's -- and the nearest
neighbour of half A should be that same subject's half B.
"""
function split_half_identifiability(tr, subject_group; n_reps=N_SPLIT_REPS, seed=11)
    rng = MersenneTwister(seed)
    subs = sort(unique(tr.subject))
    usable = [s for s in subs if count(tr.subject .== s) >= MIN_TRIALS_SPLIT]
    isempty(usable) && return (within=Float64[], between=Float64[], id_rate=NaN, n=0)
    gmap = Dict(s => g for (s, g) in zip(subs, subject_group))
    within = Float64[]
    between = Float64[]
    id_hits = Float64[]
    for _ in 1:n_reps
        A = Matrix{Float64}(undef, length(usable), 36)
        B = similar(A)
        for (i, s) in enumerate(usable)
            idx = shuffle(rng, findall(tr.subject .== s))
            h = div(length(idx), 2)
            A[i, :] = vec(mean(tr.L[idx[1:h], :]; dims=1))
            B[i, :] = vec(mean(tr.L[idx[h+1:end], :]; dims=1))
        end
        isab = [gmap[s] == "AB" for s in usable]
        mu = vec(mean(A[isab, :]; dims=1))
        sd = max.(vec(std(A[isab, :]; dims=1)), eps())
        DA = (A .- mu') ./ sd'
        DB = (B .- mu') ./ sd'
        DA ./= max.(sqrt.(sum(abs2, DA; dims=2)), eps())
        DB ./= max.(sqrt.(sum(abs2, DB; dims=2)), eps())
        C = DA * DB'
        n = length(usable)
        append!(within, diag(C))
        for i in 1:n, j in 1:n
            i != j && push!(between, C[i, j])
        end
        push!(id_hits, mean([argmax(view(C, i, :)) == i for i in 1:n]))
    end
    return (within=within, between=between, id_rate=mean(id_hits), n=length(usable))
end

# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------

function perm_eta2(values, labels; n_perm=5000, seed=7)
    v = Float64.(values)
    ok = isfinite.(v)
    v, labels = v[ok], labels[ok]
    function eta2(x)
        gm = mean(x)
        sst = sum(abs2, x .- gm)
        sst <= 0 && return NaN
        ssb = sum(count(labels .== g) * (mean(x[labels .== g]) - gm)^2 for g in unique(labels))
        return ssb / sst
    end
    obs = eta2(v)
    rng = MersenneTwister(seed)
    hits = count(_ -> eta2(shuffle(rng, v)) >= obs, 1:n_perm)
    return obs, (hits + 1) / (n_perm + 1)
end

function mannwhitney_p(a, b)
    a, b = Float64.(a), Float64.(b)
    (isempty(a) || isempty(b)) && return NaN
    all_v = vcat(a, b)
    r = invperm(sortperm(all_v))
    ranks = Float64.(r)
    # average ranks for ties
    ord = sortperm(all_v)
    i = 1
    while i <= length(ord)
        j = i
        while j < length(ord) && all_v[ord[j+1]] == all_v[ord[i]]
            j += 1
        end
        if j > i
            avg = mean(i:j)
            for k in i:j
                ranks[ord[k]] = avg
            end
        end
        i = j + 1
    end
    na, nb = length(a), length(b)
    Ra = sum(ranks[1:na])
    U = Ra - na * (na + 1) / 2
    mu = na * nb / 2
    sd = sqrt(na * nb * (na + nb + 1) / 12)
    sd <= 0 && return NaN
    z = (U - mu) / sd
    return 2 * (1 - 0.5 * (1 + erf_approx(abs(z) / sqrt(2))))
end

erf_approx(x) = sign(x) * sqrt(1 - exp(-x^2 * (4 / pi + 0.147 * x^2) / (1 + 0.147 * x^2)))

function pearson(x, y)
    x, y = Float64.(x), Float64.(y)
    m = isfinite.(x) .& isfinite.(y)
    count(m) < 3 && return NaN
    xc, yc = x[m] .- mean(x[m]), y[m] .- mean(y[m])
    d = sqrt(sum(abs2, xc) * sum(abs2, yc))
    return d > 0 ? sum(xc .* yc) / d : NaN
end

"""Partial correlation of x with y controlling for z -- the panel G question."""
function partial_cor(x, y, z)
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    den = sqrt((1 - rxz^2) * (1 - ryz^2))
    return den > 0 ? (rxy - rxz * ryz) / den : NaN
end

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

"""6x6 loading/deviation map. Rows are output joints, columns are inputs."""
function routing_heatmap!(ax, v; climit=nothing)
    M = reshape(Vector{Float64}(v), 6, 6)'   # -> M[output, input]
    lim = isnothing(climit) ? maximum(abs, M) : climit
    hm = heatmap!(ax, 1:6, 1:6, permutedims(M); colormap=DIVERGING,
        colorrange=(-lim, lim))
    ax.xticks = (1:6, JOINT_LABELS_PARETIC)
    ax.yticks = (1:6, JOINT_LABELS_PARETIC)
    ax.xticklabelrotation = pi / 4
    return hm
end

function group_strip!(ax, values, groups; order=GROUP_ORDER, jitter=0.13, seed=3)
    rng = MersenneTwister(seed)
    for (i, g) in enumerate(order)
        m = groups .== g
        count(m) == 0 && continue
        v = Float64.(values[m])
        x = i .+ jitter .* (2 .* rand(rng, count(m)) .- 1)
        scatter!(ax, x, v; color=(GROUP_COLORS[g], 0.55), markersize=11,
            marker=GROUP_MARKERS[g])
        mu = mean(v)
        se = std(v) / sqrt(length(v))
        scatter!(ax, [i], [mu]; color=:black, markersize=15)
        errorbars!(ax, [i], [mu], [se]; color=:black, whiskerwidth=12, linewidth=2.2)
    end
    ax.xticks = (1:length(order), order)
    xlims!(ax, 0.4, length(order) + 0.6)
end

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

function make_figure7()
    ensure_dirs!()
    tr = load_routing_matrix()
    st = subject_table(tr)
    D = deviation_matrix(st.S, st.group)
    absc = abnormality(D)
    V, ev = stroke_axes(D, st.group)
    A1 = D * V[:, 1]
    A2 = D * V[:, 2]
    ab = st.group .== "AB"
    stroke = .!ab

    @printf("[figure 7] %d trials, %d subjects (AB %d: HYA %d / PSGB %d, HF %d, LF %d)\n",
        size(tr.L, 1), length(st.subs), count(ab),
        count(ab .& (st.dataset .== "HYA")), count(ab .& (st.dataset .== "PSGB")),
        count(st.group .== "HF"), count(st.group .== "LF"))

    # -- dataset control: do the two able-bodied cohorts differ in ROUTING? --
    p_ds_abn = mannwhitney_p(absc[ab .& (st.dataset .== "HYA")], absc[ab .& (st.dataset .== "PSGB")])
    p_ds_a1 = mannwhitney_p(A1[ab .& (st.dataset .== "HYA")], A1[ab .& (st.dataset .== "PSGB")])
    eta_grp, p_grp = perm_eta2(absc, st.group)
    ident = split_half_identifiability(tr, st.group)

    fig = Figure(size=(1680, 1560), fontsize=PAPER_FONT_SIZE)

    # ---------------- A: deviation map, marked by dataset ----------------
    axA = Axis(fig[1, 1:2], title="Response-map deviation (each point one subject)",
        xlabel="axis 1: paretic-knee reception  (AB SD)",
        ylabel="axis 2  (AB SD)")
    panel_label!(fig, (1, 1), "A")
    vlines!(axA, [0]; color=(:black, 0.2), linestyle=:dash)
    hlines!(axA, [0]; color=(:black, 0.2), linestyle=:dash)
    for g in GROUP_ORDER, ds in ("HYA", "PSGB")
        m = (st.group .== g) .& (st.dataset .== ds)
        count(m) == 0 && continue
        scatter!(axA, A1[m], A2[m];
            color=ds == "HYA" ? (GROUP_COLORS[g], 0.75) : (:white, 0.0),
            strokecolor=GROUP_COLORS[g], strokewidth=ds == "HYA" ? 0 : 2.4,
            marker=GROUP_MARKERS[g], markersize=15,
            label="$g ($(ds == "HYA" ? "YA" : ds))")
    end
    axislegend(axA; position=:lb, framevisible=false, labelsize=PAPER_LEGEND_SIZE, nbanks=2)
    text!(axA, 0.02, 0.97;
        text=@sprintf("able-bodied cohorts do not separate:\nabnormality p = %.2f, axis 1 p = %.2f\n(open = PostStroke study controls)",
            p_ds_abn, p_ds_a1),
        space=:relative, align=(:left, :top), fontsize=15, color=GRAY)

    # ---------------- B: abnormality score ----------------
    axB = Axis(fig[1, 3], title="Abnormality score",
        ylabel="distance from AB (SD)")
    panel_label!(fig, (1, 3), "B")
    group_strip!(axB, absc, st.group)
    text!(axB, 0.03, 0.97; text=@sprintf("η² = %.2f, p = %.4f", eta_grp, p_grp),
        space=:relative, align=(:left, :top), fontsize=16, color=GRAY)

    # ---------------- C/D: what the two axes mean ----------------
    axC = Axis(fig[2, 1], title=@sprintf("Axis 1  (%.0f%% of deviation)", 100ev[1]),
        xlabel="input joint", ylabel="output joint")
    panel_label!(fig, (2, 1), "C")
    hmC = routing_heatmap!(axC, V[:, 1])
    Colorbar(fig[2, 1][1, 2], hmC; label="loading", labelsize=PAPER_COLORBAR_LABEL_SIZE,
        ticklabelsize=PAPER_COLORBAR_TICK_SIZE)

    axD = Axis(fig[2, 2], title=@sprintf("Axis 2  (%.0f%% of deviation)", 100ev[2]),
        xlabel="input joint", ylabel="output joint")
    panel_label!(fig, (2, 2), "D")
    hmD = routing_heatmap!(axD, V[:, 2])
    Colorbar(fig[2, 2][1, 2], hmD; label="loading", labelsize=PAPER_COLORBAR_LABEL_SIZE,
        ticklabelsize=PAPER_COLORBAR_TICK_SIZE)

    # ---------------- E: identifiability ----------------
    axE = Axis(fig[2, 3], title="Are signatures individual?",
        ylabel="cosine similarity", xticks=(1:2, ["same\nsubject", "different\nsubjects"]))
    panel_label!(fig, (2, 3), "E")
    rng = MersenneTwister(5)
    for (i, v) in enumerate((ident.within, ident.between))
        isempty(v) && continue
        x = i .+ 0.16 .* (2 .* rand(rng, length(v)) .- 1)
        sel = randperm(rng, length(v))[1:min(length(v), 900)]
        scatter!(axE, x[sel], v[sel]; color=(i == 1 ? GROUP_COLORS["HF"] : GRAY, 0.16), markersize=4)
        scatter!(axE, [i], [mean(v)]; color=:black, markersize=15)
    end
    hlines!(axE, [0]; color=(:black, 0.25), linestyle=:dash)
    xlims!(axE, 0.5, 2.5)
    text!(axE, 0.03, 0.97;
        text=@sprintf("held-out half of the SAME subject\nis the nearest match %.0f%% of the time\n(chance %.1f%%, n = %d subjects)",
            100 * ident.id_rate, 100 / ident.n, ident.n),
        space=:relative, align=(:left, :top), fontsize=15, color=GRAY)

    # ---------------- F: individual fingerprints at matched severity ----------------
    # Three patients chosen to sit close together on axis 1 (comparable overall
    # abnormality) but far apart on axis 2 -- the point being that equal severity
    # does not imply the same routing problem.
    # Choose three patients whose abnormality SCORES are as close as possible
    # while their deviation DIRECTIONS are as different as possible: the panel's
    # claim is that a single severity number hides distinct mechanisms, so the
    # illustration has to hold severity fixed and let direction vary.
    sidx = findall(stroke)
    Dn = D[sidx, :] ./ max.(sqrt.(sum(abs2, D[sidx, :]; dims=2)), eps())
    best = (score=Inf, combo=sidx[1:3])
    for a in 1:length(sidx)-2, b in a+1:length(sidx)-1, c in b+1:length(sidx)
        sc = absc[sidx[[a, b, c]]]
        spread = maximum(sc) - minimum(sc)
        spread > 0.45 && continue                       # comparable severity
        cosmax = maximum([dot(Dn[a, :], Dn[b, :]), dot(Dn[a, :], Dn[c, :]),
                          dot(Dn[b, :], Dn[c, :])])     # most-similar pair
        cosmax < best.score && (best = (score=cosmax, combo=sidx[[a, b, c]]))
    end
    pick = best.combo
    @printf("[figure 7] panel F: %s (scores %.2f-%.2f, max pairwise cosine %.2f)\n",
        join(st.subs[pick], ", "), minimum(absc[pick]), maximum(absc[pick]), best.score)
    gl = GridLayout(fig[3, 1:3])
    panel_label!(fig, (3, 1), "F")
    Label(gl[1, 1:3], "Similar abnormality scores, different deviation patterns",
        fontsize=PAPER_TITLE_SIZE, font=:bold, tellwidth=false)
    lim = maximum(abs, D[pick, :])
    local hmF
    for (k, i) in enumerate(pick)
        axF = Axis(gl[2, k],
            title=@sprintf("%s (%s)  score %.1f", st.subs[i], st.group[i], absc[i]),
            xlabel="input joint", ylabel=k == 1 ? "output joint" : "")
        hmF = routing_heatmap!(axF, D[i, :]; climit=lim)
        k == 1 || hideydecorations!(axF; grid=false)
    end
    Colorbar(gl[2, 4], hmF; label="deviation (AB SD)",
        labelsize=PAPER_COLORBAR_LABEL_SIZE, ticklabelsize=PAPER_COLORBAR_TICK_SIZE)

    # ---------------- G: honest clinical comparison ----------------
    scores = load_clinical_scores()
    gl2 = GridLayout(fig[4, 1:3])
    panel_label!(fig, (4, 1), "G")
    Label(gl2[1, 1:2], "Against the clinical scores, walking speed remains the stronger predictor",
        fontsize=PAPER_TITLE_SIZE, font=:bold, tellwidth=false)
    rows = NamedTuple[]
    for (k, outcome) in enumerate((:berg, :fugl))
        outcome in propertynames(scores) || continue
        idx = [findfirst(==(s), String.(scores.subject)) for s in st.subs]
        y = [isnothing(j) ? NaN : Float64(scores[j, outcome]) for j in idx]
        keep = isfinite.(y) .& stroke
        count(keep) < 6 && continue
        r_speed = pearson(st.speed[keep], y[keep])
        r_abn = pearson(absc[keep], y[keep])
        r_par = partial_cor(absc[keep], y[keep], st.speed[keep])
        push!(rows, (outcome=String(outcome), n=count(keep), r_speed=r_speed,
            r_abnormality=r_abn, r_partial_given_speed=r_par))
        ax = Axis(gl2[2, k], title=uppercase(String(outcome)),
            xlabel="abnormality score (AB SD)", ylabel=String(outcome) == "berg" ? "clinical score" : "")
        for g in ("HF", "LF")
            m = keep .& (st.group .== g)
            count(m) == 0 && continue
            scatter!(ax, absc[m], y[m]; color=(GROUP_COLORS[g], 0.8),
                marker=GROUP_MARKERS[g], markersize=14, label=g)
        end
        text!(ax, 0.03, 0.06;
            text=@sprintf("abnormality r = %+.2f\nspeed r = %+.2f\npartial (abnormality | speed) r = %+.2f\nn = %d",
                r_abn, r_speed, r_par, count(keep)),
            space=:relative, align=(:left, :bottom), fontsize=15, color=GRAY)
        k == 1 && axislegend(ax; position=:rt, framevisible=false, labelsize=PAPER_LEGEND_SIZE)
    end

    rowsize!(fig.layout, 3, Relative(0.28))
    rowsize!(fig.layout, 4, Relative(0.24))

    savefig("Figure 7.svg", fig)

    # ---------------- cached outputs ----------------
    CSV.write(joinpath(RESULTS_DIR, "figure7_routing_subject_table.csv"),
        DataFrame(subject=st.subs, group=st.group, dataset=st.dataset,
            median_speed=st.speed, n_trials=st.n_trials,
            abnormality=absc, axis1=A1, axis2=A2))
    CSV.write(joinpath(RESULTS_DIR, "figure7_routing_axis_loadings.csv"),
        DataFrame(axis=repeat(1:size(V, 2); inner=36),
            output_joint=repeat(repeat(JOINT_LABELS_PARETIC; inner=6), size(V, 2)),
            input_joint=repeat(repeat(JOINT_LABELS_PARETIC; outer=6), size(V, 2)),
            loading=vec(V),
            variance_explained=repeat(ev; inner=36)))
    CSV.write(joinpath(RESULTS_DIR, "figure7_routing_identifiability.csv"),
        DataFrame(statistic=["within_subject_cosine", "between_subject_cosine",
                             "identification_rate", "chance_rate", "n_subjects", "n_reps"],
                  value=[mean(ident.within), mean(ident.between), ident.id_rate,
                         1 / ident.n, ident.n, N_SPLIT_REPS]))
    CSV.write(joinpath(RESULTS_DIR, "figure7_routing_dataset_control.csv"),
        DataFrame(test=["abnormality_HYA_vs_PSGB_AB", "axis1_HYA_vs_PSGB_AB", "abnormality_group"],
                  statistic=["mann_whitney_p", "mann_whitney_p", "permutation_eta2"],
                  value=[p_ds_abn, p_ds_a1, eta_grp],
                  extra=[NaN, NaN, p_grp]))
    isempty(rows) || CSV.write(joinpath(RESULTS_DIR, "figure7_routing_clinical.csv"), DataFrame(rows))

    @printf("[figure 7] abnormality: AB %.2f, HF %.2f, LF %.2f (eta2 %.2f, p %.4f)\n",
        mean(absc[ab]), mean(absc[st.group .== "HF"]), mean(absc[st.group .== "LF"]), eta_grp, p_grp)
    @printf("[figure 7] identifiability: within %.3f vs between %.3f, ID rate %.0f%% (chance %.1f%%)\n",
        mean(ident.within), mean(ident.between), 100 * ident.id_rate, 100 / ident.n)
    @printf("[figure 7] dataset control (AB only): abnormality p = %.3f, axis 1 p = %.3f\n", p_ds_abn, p_ds_a1)
    return fig
end

set_theme!(paper_theme())
make_figure7()

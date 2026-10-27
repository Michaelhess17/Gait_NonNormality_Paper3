#!/usr/bin/env julia
#
# S60 figure: does anything the paper claims depend on the ridge strength?
# Rows: transient growth (paper norm), transient growth relative to the data spread
# (whitened), Q-factor, Henrici departure. Columns: group medians, subject-level
# effect sizes, and the healthy-speed effect (fraction of healthy young adults whose
# within-subject slope against speed is negative). Line style = retained rank.

include(joinpath(@__DIR__, "..", "src", "GaitNonNormalityPaper3.jl"))
using .GaitNonNormalityPaper3
using CSV, DataFrames, Statistics, CairoMakie

df = CSV.read(joinpath(RESULTS_DIR, "S60_ridge_sweep.csv"), DataFrame)
paper_theme()
const ALPHA0 = 3e-7                                  # where alpha = 0 is drawn
xpos(a) = a == 0 ? ALPHA0 : a
const RSTYLE = Dict(40 => :dash, 70 => :solid, 100 => :dot)
const METRICS = [(:log_tg, "log₁₀ transient growth\n(joint-angle units)"),
                 (:log_tg_white, "log₁₀ transient growth\n(relative to data spread)"),
                 (:log_q, "log₁₀ Q-factor"), (:henrici, "Henrici departure")]
const CONTRASTS = [("AB", "HF", "HF − AB"), ("HF", "LF", "LF − HF"), ("AB", "LF", "LF − AB")]
const CONTRAST_COLORS = [:darkorange, :purple, :black]

cohend(a, b) = (mean(b) - mean(a)) / sqrt((var(a) + var(b)) / 2)
function slope(x, y)
    xm = mean(x); sum((x .- xm) .* (y .- mean(y))) / sum(abs2, x .- xm)
end
alphas = sort(unique(df.alpha)); ranks = sort(unique(df.rank))
const SUP = Dict(zip("-0123456789.", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹·"))
sup(x) = join(SUP[c] for c in string(x))
alabel(a) = a == 0 ? "0" : (e = log10(a); isinteger(round(e, digits=6)) ? "10" * sup(round(Int, e)) : "")
xt = (xpos.(alphas), alabel.(alphas))

fig = Figure(size=(1900, 520 * length(METRICS)))
for (row, (col, lab)) in enumerate(METRICS)
    axm = Axis(fig[row, 1], xscale=log10, xticks=xt, ylabel=lab, title=row == 1 ? "group medians" : "",
        xlabel=row == length(METRICS) ? "ridge strength α" : "")
    axd = Axis(fig[row, 2], xscale=log10, xticks=xt, ylabel="Cohen's d (subject level)", title=row == 1 ? "group differences" : "",
        xlabel=row == length(METRICS) ? "ridge strength α" : "")
    axs = Axis(fig[row, 3], xscale=log10, xticks=xt, ylabel="healthy subjects with\nnegative speed slope", title=row == 1 ? "healthy speed effect" : "",
        xlabel=row == length(METRICS) ? "ridge strength α" : "", limits=(nothing, (-0.05, 1.05)))
    for ax in (axm, axd, axs); xlims!(ax, ALPHA0 / 2, 2e-2); end
    for ax in (axm, axd, axs); vlines!(ax, [1e-4], color=(:gray, 0.5), linewidth=6); end
    hlines!(axd, [0.0], color=:gray, linestyle=:dash); hlines!(axs, [0.5], color=:gray, linestyle=:dash)
    for r in ranks
        sub = df[df.rank .== r, :]
        for g in GROUP_ORDER
            y = [median(filter(isfinite, sub[(sub.alpha .== a) .& (sub.group .== g), col])) for a in alphas]
            lines!(axm, xpos.(alphas), y, color=GROUP_COLORS[g], linestyle=RSTYLE[r], linewidth=3,
                label=(r == 70 ? g : nothing))
        end
        for (k, (g1, g2, cl)) in enumerate(CONTRASTS)
            dd = map(alphas) do a
                s = sub[sub.alpha .== a, :]
                subj = combine(groupby(s, [:subject, :group]), col => (x -> mean(filter(isfinite, x))) => :v)
                cohend(subj.v[subj.group .== g1], subj.v[subj.group .== g2])
            end
            lines!(axd, xpos.(alphas), dd, color=CONTRAST_COLORS[k], linestyle=RSTYLE[r], linewidth=3, label=(r == 70 ? cl : nothing))
        end
        fr = map(alphas) do a
            s = sub[(sub.alpha .== a) .& (sub.source .== SOURCE_HYA), :]
            sl = [slope(g.speed, g[!, col]) for g in groupby(s, :subject) if length(unique(g.speed)) >= 3]
            mean(sl .< 0)
        end
        lines!(axs, xpos.(alphas), fr, color=:black, linestyle=RSTYLE[r], linewidth=3)
    end
    if row == 1
        axislegend(axm, position=:rt, framevisible=false)
        axislegend(axd, position=:rt, framevisible=false)
        axislegend(axs, [LineElement(color=:black, linestyle=RSTYLE[r], linewidth=3) for r in ranks],
            ["rank $r" for r in ranks], position=:rb, framevisible=false)
    end
end
Label(fig[0, :], "Ridge sensitivity (grey band: α = 10⁻⁴ used in the paper)", fontsize=PAPER_TITLE_SIZE, font=:bold)
savefig("S60_ridge_sweep.png", fig)

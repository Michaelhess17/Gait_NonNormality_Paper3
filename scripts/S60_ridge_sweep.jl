#!/usr/bin/env julia
#
# S60: sensitivity of the human operator metrics to the ridge strength alpha.
#
# Same fit and metric code as Figure 6 (fit_gait_operator, stabilised, d = 32,
# stride-rescaled to 100 samples/stride), swept over alpha and retained rank.
# alpha = 0 is plain exact DMD. Also records transient growth in whitened POD
# coordinates (S^-1 A^k S), i.e. growth relative to the data's own spread, which a
# stationary process bounds near 1.
#
# Writes results/S60_ridge_sweep.csv; plot with S60_ridge_sweep_figure.jl.

ENV["GAIT_STRIDE_RESCALE"] = get(ENV, "GAIT_STRIDE_RESCALE", "true")
include(joinpath(@__DIR__, "..", "src", "GaitNonNormalityPaper3.jl"))
using .GaitNonNormalityPaper3
using CSV, DataFrames, DMDAnalysis, LinearAlgebra, Statistics

const FS = 100.0
const D = 32
const ALPHAS = [0.0, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2]
const RANKS = [40, 70, 100]
const FORCE = lowercase(get(ENV, "FORCE_S60", "false")) in ("1", "true", "yes")
ensure_dirs!()
STRIDE_RESCALE || error("S60 is defined on the stride-rescaled pipeline (GAIT_STRIDE_RESCALE=true)")

function whitened_tg(result; kmax)
    A = result.A; s = result.singular_values[1:size(A, 1)]
    S = Diagonal(s); Si = Diagonal(1 ./ s)
    best = 1.0; Ak = Matrix(1.0I, size(A)...)
    for _ in 1:kmax; Ak = A * Ak; best = max(best, opnorm(Si * Ak * S)); end
    best
end

fp = joinpath(RESULTS_DIR, "S60_ridge_sweep.csv")
if isfile(fp) && !FORCE
    println("cached: $fp"); exit(0)
end

trials, meta = load_human_superset_trials()
f0 = FS / PHASE_STEPS
kmax = PHASE_STEPS                                # one stride
h = HankelGenerator(n_delays=D, delay_interval=1)
rows = [NamedTuple[] for _ in eachindex(trials)]
Threads.@threads for i in eachindex(trials)
    X = center_trial(trials[i])
    for r in RANKS, a in ALPHAS
        res = try
            fit_gait_operator(DMD(rank=FixedRank(r), stabilize=true), X, h; alpha=a)
        catch err
            @warn "fit failed" i r a err; continue
        end
        q = q_factor(res, f0; fs=FS, step=0.01)
        push!(rows[i], (; idx=i, subject=String(meta.subject[i]), source=String(meta.source[i]),
            group=String(meta.lf_or_hf[i]), speed=Float64(meta.speed[i]), rank=r, alpha=a,
            log_tg=log10(transient_growth(res; kmax=kmax)), log_tg_white=log10(whitened_tg(res; kmax=kmax)),
            log_q=log10(max(q, eps())), henrici=henrici_departure(res)))
    end
    i % 25 == 0 && println("trial $i")
end
df = DataFrame(reduce(vcat, rows)); sort!(df, [:idx, :rank, :alpha])
CSV.write(fp, df); println("Wrote $fp ($(nrow(df)) rows)")

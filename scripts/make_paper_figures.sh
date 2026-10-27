#!/usr/bin/env bash
# Rebuild every figure in the manuscript (Figs 1-7 and Supplementary Fig. S60).
#
# Human analyses (Figs 5-7, S60) use the current pipeline: each trial is
# uniformly resampled to 100 samples per mean stride, and operators are fitted
# with ridge-regularised DMD (alpha = 1e-4, rank 70, 32 delays). Scripts reuse
# cached CSVs in results/; set FORCE_FIGURE5/6=true (or FORCE_S60=true) to refit.
#
# Order matters: Figure 7 reads the pooled ("all") Figure 6 caches, and the
# final Figure 6.svg is the post-stroke-dataset (PSGB) run, so it goes last of
# the two Figure 6 runs.
set -euo pipefail
cd "$(dirname "$0")/.."
JULIA="${JULIA:-julia}"           # on Nix systems: JULIA=./scripts/julia-env
run() { echo ">> $*"; "$JULIA" --project=. "$@"; }

run scripts/01_pipeline_schematic.jl
run scripts/02_benchmark_oscillators.jl
run scripts/03_slip_locomotor_intuition.jl
[ -d external/loco_mujoco_mocap/processed ] || external/loco_mujoco_mocap/fetch_and_convert.sh
run scripts/04_bipedal_joint_structure.jl

export GAIT_STRIDE_RESCALE=true GAIT_RIDGE_ALPHA=1e-4
run scripts/05_able_bodied_baseline.jl
FIGURE6_SOURCE=all run scripts/06_stroke_clinical_potential.jl
run scripts/06_stroke_clinical_potential.jl
FIGURE7_CONFIG_TAG=d32_fixedrank70_all_stride100_ridge0.0001 run scripts/07_individual_routing_signatures.jl

run -t auto scripts/S60_ridge_sweep.jl
run scripts/S60_ridge_sweep_figure.jl

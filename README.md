# Gait Non-Normality Paper 3

Analysis and figure code for the manuscript on delay-embedded DMD as a
"mathematical microscope" for gait dynamics: operator geometry, transient
growth and resolvent structure across benchmark oscillators, SLIP, bipedal
robots, able-bodied walkers and stroke survivors.

The current manuscript is `draft/draft.docx`; its figures are in `figures/`.

## Quick start

Requirements: [Julia 1.12](https://julialang.org/downloads/), `git`, `curl`,
and Python 3 with `numpy` (only to fetch the robot data for Figure 4).

```bash
git clone git@github.com:Michaelhess17/Gait_NonNormality_Paper3.git
cd Gait_NonNormality_Paper3
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # one-time, ~10-20 min
./scripts/make_paper_figures.sh                        # rebuilds Figs 1-7 and S60
```

`Pkg.instantiate()` installs the exact package versions in `Manifest.toml`,
including the lab's DMD package, fetched from
[github.com/Michaelhess17/DMDAnalysis](https://github.com/Michaelhess17/DMDAnalysis)
at the pinned commit. Nothing else needs to be cloned.

Each figure can also be built on its own, e.g.
`julia --project=. scripts/03_slip_locomotor_intuition.jl`. For Figures 5-7
and S60, set the pipeline variables first (see below) or use
`make_paper_figures.sh`, which sets them and runs everything in the right order.

The first run computes everything from scratch and caches the results as CSVs
in `results/`; later runs reuse the caches and only redraw. Set
`FORCE_FIGURE5=true`, `FORCE_FIGURE6=true` or `FORCE_S60=true` to refit.

## One file that is not in the repository

`data/clinical_scores_combined.csv` (Berg Balance and Fugl-Meyer scores per
stroke participant) is participant data and is shared separately. Copy it into
`data/` before running Figures 6 and 7; without it their clinical panels have
no scores to correlate against. Everything else runs from the clone alone.

## What is in the repository

```text
draft/draft.docx                current manuscript
figures/                        paper figures (Figs 1-7 as SVG, S60 ridge sweep)
scripts/01-07_*.jl              one script per manuscript figure
scripts/S60_ridge_sweep*.jl     Supplementary: sensitivity to ridge strength and rank
scripts/make_paper_figures.sh   rebuilds every figure in order
scripts/05_build_human_gait_superset.py
                                how data/human_gait_superset.npy was assembled
                                from the public GaitSignatures repositories
src/GaitNonNormalityPaper3.jl   shared code: data loading, stride rescaling,
                                ridge-regularised operator fit, plotting theme
vendor/SLIP.jl                  biped SLIP simulator used by Figure 3
data/human_gait_superset.*      joint-angle trajectories for all human trials
                                (299 trials x 1500 samples x 6 joints) + metadata
external/loco_mujoco_mocap/     downloads and converts the LocoMuJoCo robot mocap
                                for Figure 4 (dataset is CC-BY-NC-ND, so it is
                                fetched from Hugging Face rather than stored here)
```

`results/` (caches) is generated and not tracked.

## Human-data pipeline settings

Figures 5-7 and S60 use the current pipeline, switched on by two environment
variables read in `src/` (`make_paper_figures.sh` sets both):

| variable | value | meaning |
|---|---|---|
| `GAIT_STRIDE_RESCALE` | `true` | resample each trial uniformly to 100 samples per mean stride |
| `GAIT_RIDGE_ALPHA` | `1e-4` | ridge (Tikhonov) strength of the DMD fit, relative to the leading singular value squared |

Without them the scripts fall back to raw-time, unregularised DMD. Cache file
names carry the configuration (e.g. `*_d32_fixedrank70_all_stride100_ridge0.0001.csv`),
so results from different settings never mix.

## Nix systems

On NixOS, Julia binary artifacts can fail with
`libquadmath.so.0: cannot open shared object file`. Use the wrapper, which
exposes the Fortran runtime and keeps its own package depot in `.julia_depot/`:

```bash
./scripts/julia-env --project=. -e 'using Pkg; Pkg.instantiate()'
JULIA=./scripts/julia-env ./scripts/make_paper_figures.sh
```

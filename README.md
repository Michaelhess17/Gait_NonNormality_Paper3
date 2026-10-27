# Gait Non-Normality Paper 3

Analysis and figure code for the manuscript on delay-embedded DMD as a
"mathematical microscope" for gait dynamics: operator geometry, transient
growth and resolvent structure across benchmark oscillators, SLIP, bipedal
robots, able-bodied walkers and stroke survivors.

## Layout

```text
src/GaitNonNormalityPaper3.jl   shared helpers: data loading, stride rescaling,
                                ridge-regularised operator fit, plotting theme
scripts/01-07_*.jl              one script per manuscript figure (Figs 1-7)
scripts/S60_ridge_sweep*.jl     Supplementary ridge/rank sensitivity figure
scripts/05_build_human_gait_superset.py
                                builds data/human_gait_superset.npy from the
                                public GaitSignatures repositories
scripts/make_paper_figures.sh   rebuilds every paper figure in the right order
scripts/julia-env               Julia wrapper for Nix systems (see below)
figures/                        paper figures (tracked)
results/                        cached CSVs (generated, not tracked)
draft/                          manuscript (draft_v2.docx is current)
data/, external/                inputs, not tracked (see below)
archive/                        supplemental/exploratory analyses, old figures,
                                notes and backups; local only, not tracked
```

## Rebuilding the figures

```bash
./scripts/make_paper_figures.sh
```

The human analyses (Figs 5-7, S60) are switched to the current pipeline by
environment variables read in `src/`:

| variable | value used | meaning |
|---|---|---|
| `GAIT_STRIDE_RESCALE` | `true` | resample each trial to 100 samples per mean stride |
| `GAIT_RIDGE_ALPHA` | `1e-4` | Tikhonov strength (relative to s1^2) of the DMD fit |

Without them the scripts fall back to raw-time, unregularised DMD. Cached
results are tagged with the configuration (e.g.
`*_d32_fixedrank70_all_stride100_ridge0.0001.csv`), so settings never mix.

## Inputs (not in the repository)

- `../DMDAnalysis` - the reusable DMD package (path source in `Project.toml`).
- `external/SLIP.jl` - vendored SLIP simulator used by Figure 3.
- `external/loco_mujoco_mocap` - robot/mocap trajectories for Figure 4.
- `data/human_gait_superset.npy` and `data/human_gait_superset_metadata.csv` -
  built by `scripts/05_build_human_gait_superset.py`.
- `data/clinical_scores_combined.csv`, `data/Table_Demogrpahics.xlsx` - clinical
  scores; participant data, kept out of version control.

## Julia runtime

On Nix-based systems Julia JLL artifacts can fail with
`libquadmath.so.0: cannot open shared object file`. Use the wrapper, which
exposes the Fortran runtime and keeps the depot inside the workspace:

```bash
./scripts/julia-env --project=. -e 'using Pkg; Pkg.instantiate()'
./scripts/julia-env --project=. scripts/05_able_bodied_baseline.jl
```

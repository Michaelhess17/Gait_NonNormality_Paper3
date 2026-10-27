# Gait manuscript revision plan for Michael

**Purpose:** A self-contained analysis and writing plan for a possible PNAS Research Article.  
**Prepared:** September 25, 2026.  
**Scope:** Reanalysis of existing recordings and new computational validation; no new human or robot experiments.

This plan is based on the 30-page draft *A Practical Approach for Data-Driven Modeling and Characterization of Bipedal Walking Dynamics* by Michael Hess, Michael Rosenberg, and Gordon J. Berman, available as `Human Gait DMD Paper.pdf`, and the revision discussion titled *Paper Review Submission Advice*. Page and figure references below refer to that draft. The local PDF matches the manuscript and discrepancies described in the discussion. Older drafts with different titles were not treated as the current results.

**Status:** This is a proposed revision program, not a report of completed reanalyses. Existing numerical results remain provisional where the draft is inconsistent. “Supportive outcome” below means a result that would justify a claim, not an outcome to optimize the analysis to obtain. The proposed title, significance statement, abstract, and paper outline at the end are conditional on the relevant tests succeeding.

## 1. The scientific question and the change in emphasis

The central question should become:

> Can short recordings of walking reveal reproducible organization in how deviations evolve across a multijoint body, beyond what is explained by the average gait cycle?

For a locomotion researcher, the value is concrete: two people may produce similar movement and have similar scalar stability summaries, yet differ in where deviations appear, how large they become, and how they evolve over the following steps. Establishing that distinction would provide a useful intermediate description between average kinematics and a detailed neuromusculoskeletal model.

The paper should therefore move away from a contest in which “the classic stability number fails” and a collection of operator metrics wins. Average kinematics, local divergence estimates, Floquet multipliers, and multivariate response models answer different questions. A scalar summary cannot describe a full joint-resolved response, but that does not make established stability analysis incorrect or uninformative.

The intended chain of evidence is:

**Validated perturbation prediction in a multijoint simulation → structure beyond the mean waveform → healthy state dependence → stroke-associated asymmetry → reproducible individual variation.**

There are three levels of interpretation to keep separate:

| Evidence | Defensible description | What it does not establish |
| --- | --- | --- |
| An operator fitted to observational kinematics | Model-implied response organization under specified coordinates and input/output definitions | A neural controller, muscle force pathway, or causal anatomical connection |
| Accurate predictions of imposed perturbations in a simulated walker | Validated perturbation-response prediction in that simulation and tested regime | Validated human intervention responses |
| Human effects that survive waveform, sampling, and nuisance controls | Reproducible kinematic dynamics associated with speed, stroke, and identity | A treatment target, fall-risk predictor, or rehabilitation benefit |

“Response organization” is the public-facing term. Define “finite-time response geometry” mathematically if useful, but do not ask that phrase to carry the biological motivation.

## 2. What the present draft provides

| Material | Current scope | How it should contribute |
| --- | --- | --- |
| Healthy young adults (HYA) | 150 trials from 17 people | Within-person speed effects and cross-speed identification |
| Post-stroke study | 149 trials from 25 people; 30 control, 65 higher-functioning, and 54 lower-functioning trials | Stroke asymmetry and heterogeneity; these are trial counts, not participant counts |
| Post-stroke participants | Five able-bodied controls and 20 stroke participants; the draft states 9–11 participants per stroke group | Modest independent sample; verify exact HF/LF assignment and inclusion counts |
| Combined human data | 299 trials from 42 people | Reproducibility, with study and diagnostic-group controls |
| Human measurements | Six bilateral sagittal hip, knee, and ankle angles; nominally 100 Hz and 15 seconds per trial | Kinematics only; no direct measurement of muscle recruitment, force transfer, or frontal-plane circumduction |
| Existing validation | Stuart–Landau oscillators, one-way coupled oscillators, SLIP, and three robots | Supporting mathematical and cross-system evidence; currently no direct multijoint perturbation validation |

The current operator uses 32-sample histories and a retained rank of 70. Eigenvalues outside the unit circle are projected onto it before response calculations. Joint “routing” is a normalized gain between entire delayed joint-coordinate blocks at the stride frequency. These choices are central to the proposed checks, not minor implementation details.

The most promising human observations are the paretic-side output asymmetry, particularly at the knee (current Fig. 6G), and within-person similarity of response maps (current Fig. 7). Neither should yet be called a validated control mechanism or clinical biomarker.

## 3. Execution order and shared analysis rules

### Recommended sequence

| Stage | Work | Decision it enables |
| --- | --- | --- |
| 0 | Reconcile the source data, figure computations, metric definitions, and forecast splits (A1–A3) | Establish a trustworthy analysis pipeline |
| 1 | Mean-cycle surrogates, response-matrix rank, and fixed-stride refits (A4–A6) | Determine what structure the existing human results actually contain |
| 2 | Define physical input/output mappings, then validate in a multijoint walker (A7–A8) | Determine whether physical perturbation language is supported |
| 3 | Human deviation forecasts, stroke asymmetry, optional normative transfer, and individual specificity (A9–A12) | Establish the biological claims and their limits |
| 4 | Recording-length reliability, final robustness, and manuscript reconstruction (A13–A14) | Earn the “short recordings” claim and assemble the paper |

Settle the basic rank and numerical conventions in Stage 0, use them throughout, and run the broader robustness checks alongside later analyses. Do not postpone those choices until after selecting the most attractive biological result. The multijoint simulation need not become a large project until the surrogate and sampling tests show there is a response-specific result worth pursuing.

### Rules that apply throughout

1. **People are the independent units for human inference.** Bootstrap subjects and then trials or stride blocks within subjects as appropriate. Do not count overlapping windows, trials, or repeated random splits as independent people. Report usable participants and trials for each endpoint.
2. **Separate fitting, tuning, and testing.** Fit scaling, waveform templates, singular-vector bases, operator parameters, nuisance regressions, reference distributions, and feature selection using training data only for predictive evaluations. Development findings from this already-explored dataset remain exploratory even after cleaner validation.
3. **Avoid overlapping-data leakage.** Delay windows and forecast targets assigned to different splits must not reuse samples. When splitting a continuous recording, leave enough separation for the history and prediction horizon; split independent recordings when possible. Filtering must also respect the prediction boundary.
4. **Keep a common comparison contract.** Real data, surrogates, and baselines use the same available channels, training budget, sampling convention, horizons, and admissible model-selection rule. A baseline must receive the same observed initialization information as the dynamical model.
5. **Predefine a small set of endpoints.** Prioritize the knee output asymmetry, incremental prediction of deviations, and residual identification. Treat selection among 36 pathways or many metrics as a multiple-testing problem. Show effect sizes, uncertainty, and individual data alongside corrected tests.
6. **Report negative and inconclusive results.** A wide interval is not evidence of equivalence. If a control removes an effect, revise the claim; if estimates are too noisy, state that the current recordings cannot decide it.

## 4. Additional analyses and required changes

### A1. Reconcile figures, numbers, labels, and metric definitions

**Why this is needed.** Some discrepancies reverse the scientific conclusion. Others change the apparent strength or meaning of the individual-signature result. Text editing alone cannot resolve them.

**How to do it.** Rebuild each panel from a versioned results table, recording the data subset, estimator, model settings, target quantity, and uncertainty calculation. Generate caption values and prose values from the same table. Audit at least the following:

| Location | Discrepancy to resolve |
| --- | --- |
| Fig. 2F, pp. 7–10 | Prose and artwork report approximately 0.90, 0.92, and 0.996; the caption reports 0.88, 0.45, and 0.979, plus an eigenvalue decoder at 0.994. Identify which analysis and artwork are current. |
| Fig. 3E/J, pp. 10–13 | The plotted return-map comparison has correlation about 0.95 and identity-line R² about 0.91, with small errors. The prose says about 4% explained and MAE 0.47 versus 0.13; the caption gives return-map R² = 0.16. Identify the estimator and recompute every comparison. |
| Fig. 6I, pp. 19–21 | The text, caption, and artwork refer to screens of 45, 44, and 46 features. The named best Berg feature also differs. Reconcile the candidates, selection, missing clinical scores, and correction. |
| Fig. 7C–E, pp. 22–24 | Prose/caption give 44% and 21% for shared axes; artwork gives 40% and 22%. Prose says 41% identification; artwork says 39%. The abstract describes a held-out trial, while the Methods describe a held-out half of trials. These are different evaluation units. |
| Response definitions, pp. 27–29 | Figures use “Henrici” while Methods give a normalized commutator norm; some plotted formulas appear squared or square-rooted differently from the prose definition of transient gain. Verify formulas, norms, log transforms, and labels against code. |

Define whether Fig. 3 “Return map” is an empirical Poincaré-map fit to simulated observations, a numerical derivative of a known map, or an analytical quantity. An empirical estimate is a conventional-estimator benchmark; an analytical result checked against the simulator is a consistency check. Neither should be described as a failure if it performs well.

Separate correlation, fitted-regression R², identity-line R², cross-validated R², and mean absolute error. Specify the Floquet multiplier versus the continuous-time decay rate and its sign convention. Also verify whether participation ratio uses singular values or variance eigenvalues: the draft's `(sum sigma)^2 / sum sigma^2` is a singular-value participation ratio, not automatically a variance-based effective dimension. It describes the delay data, not the number of neural control modules.

**Supportive outcome.** One reproducible set of numbers, correct estimator labels, and conclusions that follow from the verified panels. Conventional stability estimation is allowed to work well.

**If it fails or changes the result.** Replace or withdraw the unsupported comparison. If the source computation cannot be reconstructed, remove that result until it can. Do not choose the numerical version that best supports the old story.

**Deliverable:** An internal audit table and regenerated current Figs. 2–7, including their captions.

### A2. Establish what “held out” means and verify genuine forecasting

**Why this is needed.** The Methods describe fitting on all time-shifted states from a trial, while several figure descriptions show only an initial history window and call the remainder a forecast. Supplying an initial history is not the same as restricting model fitting to that history. Reconstruction of a trajectory used to estimate the operator does not validate future prediction.

**How to do it.** For each forecast panel, explicitly state the samples used to estimate the operator, tune the model, initialize the state, and score the future. Prefer independent training and test trials at comparable conditions when available. For a within-trial test, fit on a clearly marked prefix and predict a strictly later suffix; report how much of the 15-second record remains for each purpose. Avoid estimating the test-trial mean, stride frequency, phase trajectory, or normalization from the unseen suffix.

Compare with a training-derived mean-cycle predictor initialized from observed history, persistence where relevant, and a modest regularized autoregressive predictor. Distinguish free-running multi-step prediction from predictions repeatedly updated with observed samples. Evaluate overall angles and deviations from the mean cycle separately, with errors by joint and horizon. State whether normalization uses training scale or a fixed evaluation convention; provide an absolute-error summary as well.

**Supportive outcome.** Future joint trajectories, and preferably stride-to-stride deviations, are predicted better than a phase-aware mean-cycle baseline across subjects, with an improvement that persists beyond the first few samples.

**If it fails.** Relabel in-sample demonstrations as reconstruction. If genuine forecasts reproduce only the mean cycle, narrow the predictive claim and treat the response interpretation as unvalidated until A8/A9 provide stronger evidence. Do not claim the model forecast ten strides from a short record unless the split actually permits it.

**Deliverable:** A split diagram, corrected forecast Methods, and error-versus-horizon curves with subject-level uncertainty.

### A3. Choose defensible model complexity and test numerical sensitivity

**Why this is needed.** Multi-cycle forecasting reportedly plateaus near rank 15, yet the primary rank is 70. Extra poorly determined directions can dominate operator norms without improving useful predictions. Eigenvalue clipping also changes the object being interpreted, and clipping to magnitude one does not establish asymptotic decay.

**How to do it.** Use blocked or trial-level validation to choose the smallest rank within a predefined forecast plateau, such as a one-standard-error rule, on development data. Retain rank 70 as a sensitivity analysis. If a common rank is used, select it without clinical group labels; if ranks vary, use the same selection rule for everyone and report them. Ensure every retained direction is numerically supported by the data. The standard DMD reference itself discusses pitfalls with rank-deficient data. [Tu et al., 2014](https://www.aimsciences.org/article/doi/10.3934/jcd.2014.1.391).

Compare raw operators, the existing clipping procedure, and an independently regularized fit with an explicit stability constraint or damping convention. Ordinary ridge regression alone does not guarantee stability. State what happens to oscillatory/neutral modes. For finite horizons, raw unstable fits can still be evaluated; report blow-ups rather than silently dropping them. Show eigenvalue spectra, fractions modified, reconstruction error, conditioning, and changes in headline response summaries.

Unit-circle resolvents can become singular or grid-dominated near modes placed exactly on that circle. Specify frequency resolution, damping/radial offset if used, and treatment of unresolved Q peaks. Repeat at reasonable alternative settings; never turn an undefined bandwidth into a physiological ceiling. Report Q values hidden by the current plotting cap.

Test delay duration, rank, reasonable channel scaling, and response horizon. Gains and non-normality depend on the coordinate metric: the norm of a reduced delay-state perturbation is not directly physical joint-angle amplification or energy. Use output-space response measures when making physical comparisons. Preserve absolute gains as diagnostics even if normalized response maps are primary; normalization does not remove all preprocessing sensitivity.

Report raw spectral rates in comparable units. For a relevant sampled mode, `|lambda|^(samples per stride)` is a per-stride conversion, but the dominant DMD eigenvalue need not be a transverse Floquet multiplier: periodic/phase modes may be neutral. Define the mode and comparison rather than identifying these objects by name alone.

**Supportive outcome.** The main waveform-excess, speed, asymmetry, and identity results survive parsimonious ranks and reasonable fitting conventions. Exact global gain values need not agree across every coordinate system.

**If it fails.** Drop unstable metrics from the main argument. If normalized maps or biological effects depend on high rank or one clipping rule, report that dependence and narrow the paper. Do not call clipping a conservative bound on physical response.

**Deliverable:** A compact robustness summary for each headline claim, with full parameter grids and numerical diagnostics in SI.

### A4. Test the mean-cycle explanation with a surrogate hierarchy

**Why this is needed.** Near the stride frequency, an operator fitted to periodic kinematics may largely rediscover the Fourier structure of the mean gait cycle. A joint response map could therefore reflect ordinary waveform amplitude and phase rather than information about deviations. This is the earliest decisive test of the proposed biological interpretation.

**How to do it.** Estimate each trial's multijoint phase-dependent mean cycle and residuals, documenting event detection and alignment. Build repeated realizations of the following controls:

| Control | What it preserves | What it tests |
| --- | --- | --- |
| Repeated multijoint mean cycle | Joint waveform, relative phases, and chosen cadence schedule | Whether the average periodic motion explains the map |
| Mean cycle plus independently generated joint residuals | Mean waveform plus each joint's residual variance and autocorrelation or spectrum | Whether marginal stride variability explains the remaining structure without residual cross-joint timing |
| Independently phase-randomized channels | Each channel's power spectrum and variance | Whether joint timing adds information beyond marginal spectra |
| Matched colored noise and i.i.d. noise | Selected marginal statistics, sample count, and dimensions | Numerical floors for fitting and delay embedding |

Where data permit, add independently shuffled residual stride blocks while preserving phase-dependent residual variance. Keep the mean cycle's cross-joint structure intact in the mean-cycle-plus-noise control; destroy relationships among residuals, not the mean coordination the control is meant to preserve. Independent channel phase randomization is a different null and can also change waveform shape.

Match sampling, record length/stride count, preprocessing, delay, and admissible rank-selection rules. Do not force a rank-70 inverse onto an exactly repeated, numerically low-rank cycle. Report effective rank and use the same singular-value cutoff policy. Use independent surrogate seeds and enough realizations for stable uncertainty; a single surrogate curve is insufficient.

Compare real and surrogate response maps, singular spectra, input/output summaries, transient gain, non-normality, Q, and data participation ratio. More importantly, repeat the actual speed, knee-asymmetry, and identity analyses on the controls. High real–surrogate map correlation alone cannot decide whether a small but reliable biological difference remains.

For a positive normalized map `R`, compare in log-ratio space. One defensible starting point is:

```text
L_ij = log(R_ij + epsilon) - mean_over_all_entries[log(R + epsilon)]
D_ij = L_ij(real) - mean_over_surrogates[L_ij(null)]
```

Choose and disclose a numerical floor, assess sensitivity to it, and propagate surrogate uncertainty. `D` is a relative excess-response feature, not an absolute gain, a physical subtraction of forces, or proof that all waveform confounding is removed. For predictive testing, build waveform templates from training data; if a new trial's observed portion is used to construct its own surrogate feature, apply that rule identically to every method without consulting its future or identity label.

**Supportive outcome.** The mean cycle explains common hip/knee structure but not the reproducible effects of interest; residual structure predicts deviations or retains state, stroke, or individual information. A visibly different full map is not required if the meaningful residual is reliable.

**If it fails.** If surrogates reproduce the pathology and identity results and leave no reliable incremental information, describe the method as a compact organization of gait waveforms. Stop presenting those results as response structure beyond mean movement. If only some effects survive, retain only those claims; do not define a noisy residual merely to preserve the original narrative.

**Deliverable:** Proposed Fig. 2, plus a table showing which headline effects survive each null model.

### A5. Determine whether the map is pairwise or nearly rank one

**Why this is needed.** A 6×6 gain map suggests 36 separate pathways. If a dominant resolvent component factorizes into an output vector and an input vector, most entries may be explained by two joint-level profiles instead. That is an interpretive simplification, not necessarily a failure.

**How to do it.** For each trial, compute singular spectra of the positive gain map and, separately, the underlying complex transfer operator. Report `s1^2 / sum(sk^2)`, rank-one reconstruction error, and reproducibility of higher-rank structure. Global normalization leaves a matrix's relative singular spectrum unchanged; taking logs or subtracting a surrogate does not, so analyze those representations separately.

Compare the measured map with its best rank-one approximation and with an input/output marginal baseline. Test whether any claimed pathway-specific effect remains beyond the factors. If the response is dominated by one component, label the factors **model receptivity** (input direction weighting) and **response expression** (output weighting). Nonnegative gain matrices can lose signs and phases, so their factorization alone does not establish a directed anatomical network. In particular, independent channel phase randomization may leave some rank-one norm summaries similar; that result is informative rather than a reason to discard the null.

**Supportive outcome.** Either two stable, interpretable factors explain the biological results, or reproducible higher-rank structure justifies discussing selected pair-specific responses. The representation used in the paper follows the result.

**If it fails.** If higher-rank components are unstable, remove pair-specific pathway claims and use the stable factors. If those factors are completely explained by waveform controls, fall back to the waveform interpretation from A4.

**Deliverable:** Singular spectra and rank-one residuals; a documented decision about whether subsequent figures show factors, full maps, or both.

### A6. Refit healthy speed effects using equal numbers of strides

**Why this is needed.** Faster walking contributes more cycles to a fixed 15-second recording. That changes estimation precision, signal content, and effective sample size, and could explain apparent speed effects in current Fig. 5.

**How to do it.** Select the same number of complete strides from each eligible trial, with stride count chosen from the observed minimum that supports a meaningful fit. Record exclusions and whether slow participants are lost. Repeat admissible segment choices without treating them as independent trials. Recompute every Fig. 5 endpoint, including forecasting, response spectra, global summaries, and joint maps or factors.

Equal stride count and equal sample count are different controls. With native 100-Hz data, equal-stride windows are longer at slower speeds. Keep this as the primary cycle-count control and add a clearly labeled common-phase-grid sensitivity analysis if feasible; its operator advances gait phase rather than fixed physical time. Compare fixed-duration and fixed-stride estimates using paired participants. Check response horizons in both seconds and strides and delay windows in both physical duration and stride fraction. Otherwise a “speed effect” can be a change in how much of a stride the analysis includes.

Estimate within-person speed relationships and combine them across people, or use a parsimonious repeated-measures model. Keep speed continuous where possible and show participant trajectories. Re-select any highlighted pathways in development data only, or label selection and inference as exploratory with multiplicity correction.

**Supportive outcome.** The important speed-dependent reorganization persists with matched strides, waveform controls, and compatible horizons. The draft's anticipated global pattern is less transient gain, lower data participation ratio, and sharper tuning at faster speeds; these directions are hypotheses, not required outcomes.

**If it fails.** Attribute the disappearing component to sampling or precision and remove it from the state-dependence claim. If only joint organization survives, center the figure on that. If common-phase and physical-time fits differ, explain which aspect of timing each measures instead of choosing the more favorable result.

**Deliverable:** Proposed Fig. 3 with fixed-stride results, plus paired fixed-duration comparisons in SI.

### A7. Define a response to a physically meaningful current perturbation

**Why this is needed.** The present `B_in` and `C_out` select all delayed copies of a joint. Their resolvent gain describes an optimal mapping between histories. A physical perturbation now does not change the recorded past. The current quantity therefore cannot simply be called the response to kicking one joint.

**How to do it.** Retain the current measure as a clearly named **joint-history gain** if useful, and define a separate current-state response for validation. Write the input and output mappings explicitly. For an instantaneous observable displacement with unchanged past samples, a candidate is:

```text
delta_x_0 = [delta_y_0; 0; ...; 0]
b_j = U_r^T delta_x_0 / perturbation_amplitude
C_i = selector_for_current_output_joint_i times U_r
h_ij(k) = C_i A_reduced^k b_j
```

Here `delta_y_0` must be the observation change produced by the actual admissible perturbation. A constrained walker may require several coordinates to change together; an arbitrary isolated joint edit may violate foot contact or other constraints. Document the projection of the physical perturbation into reduced coordinates and how much of it that subspace retains. Specify units: equal angle changes, velocity changes, and torque impulses are not interchangeable input norms.

An angular-velocity impulse is a particular observability problem when the state contains only current and past angles: at the instant of the impulse, all measured angles can remain identical. Such a model cannot distinguish the kicked and unperturbed states at that instant. Use a represented observable reset for the primary test, an explicitly expanded angle/velocity observation model, or a declared delayed-start prediction after observing post-impulse samples. The last option tests prediction after detection, not prediction from the pre-impulse history alone. Never manufacture a changed past to make the input visible.

Define a finite-horizon response-strength matrix from the predicted physical outputs, for example the root-sum-square response over a prespecified one- or two-stride horizon per unit input. Preserve signed time courses as well as magnitude summaries. Distinguish an initial-state response `C A^k b` from a forced response `C(zI-A)^(-1)B`: the latter requires a defined input sequence and mapping. Observational DMD does not itself identify a physical actuator matrix; input-aware system identification requires additional input information. [Proctor et al., 2016](https://epubs.siam.org/doi/10.1137/15M1013857).

**Supportive outcome.** The input/output definition has a direct simulator counterpart and yields reproducible output predictions. Agreement with the original history gain can then be tested rather than assumed.

**If it fails.** Keep the original quantity as an observational history-space summary. If a new quantity is validated, rerun the human biological claims with that quantity; validating one metric does not automatically validate another.

**Deliverable:** A precise response definition, dimensional/unit checks, and a schematic of what is perturbed and what is measured.

### A8. Validate actual perturbation propagation in a multijoint walker

**Why this is needed.** SLIP has no hip/knee/ankle network with which to validate joint-resolved responses. The one-way oscillator is a useful mathematical control but does not establish performance in a mechanical, contact-switching walker. This simulation is the most important addition for the proposed response interpretation.

**How to do it.** Use an existing validated multijoint walker implementation where possible, such as a kneed compass walker or a planar multilink biped. Choose the simplest system with a sustained gait and multiple measurable joint outputs. Verify contact events, numerical integration, nominal gait, and basic perturbation behavior before fitting the data-driven model. Do not assume that a short custom simulation is automatically a trustworthy ground truth.

1. Generate training trajectories during steady walking, with no designed test kicks included. If small process fluctuations or initial-condition variability are needed to explore nearby states, state that explicitly. Add measurement noise separately so its effect can be distinguished from real dynamical excitation.
2. Include a perfectly repeated, noiseless limit-cycle training condition as a negative control. A trajectory confined to the cycle generally cannot uniquely determine transverse dynamics: the recovery claim is conditional on observable variability and an adequate model. This is an identifiability issue, not something delay embedding automatically solves.
3. Freeze fitting choices, input mappings, and output metrics before the final perturbation tests. Use separate development simulations for any tuning.
4. Apply admissible small positive and negative perturbations at several prespecified gait phases and joints. Test more than one amplitude to check the linear range. If stochastic simulations are used, compare paired runs with matched future noise or sufficiently many replicates.
5. Define the measured response as the perturbed trajectory minus its matched unperturbed continuation. Compare it with the difference between perturbed and unperturbed model forecasts. Score the next one or two strides in physical time, and report phase-aligned shape deviations separately; real phase shifts are part of the response, not an error to remove silently.
6. Compare signed joint time courses, response amplitude, time to peak, cross-joint ordering, and the full finite-horizon response-strength matrix. Use a mean-cycle/phase-shift baseline, a modest autoregressive baseline, and the mean-cycle-surrogate model from A4. Include a known-equations local response as an upper benchmark where feasible.
7. Repeat across a small, prespecified range of mechanical or controller settings, phases, noise levels, and training lengths. Report failures near contact transitions and responses outside the training-supported subspace. A phase-independent operator may capture cycle-averaged strengths without correctly predicting phase-specific responses.

An especially informative optional test is a pair of simulated settings with similar nominal gait but different measured responses. It directly demonstrates the motivating distinction, but select examples by a declared similarity rule and report the broader tested set.

**Supportive outcome.** Models trained only on the permitted trajectories predict meaningful signed and joint-resolved perturbation responses beyond simple waveform baselines, across a stated range of phases and settings. Success only in magnitude or cycle averages earns that narrower claim.

**If it fails.** Diagnose insufficient excitation, unobserved state changes, contact/phase dependence, model class, and incorrect input mappings. A phase-conditioned or residual-state model is a possible scientific revision, but requires fresh validation and corresponding human reanalysis. If success requires fitting the test perturbations, it no longer supports inference from unperturbed recordings. If prediction still fails, retain “model-implied response” language and reconsider the PNAS framing rather than presenting attractive simulated examples as validation.

**Deliverable:** Proposed Fig. 1, a reproducible simulation protocol, and full held-out performance across the tested conditions in SI.

### A9. Predict deviations in held-out human recordings

**Why this is needed.** Good prediction of a periodic trajectory can be dominated by the average gait. Joint-resolved prediction of deviations is a stronger test of whether the model captures the fluctuations relevant to the paper. It remains forecast validation, not a substitute for exogenous perturbation experiments.

**How to do it.** Estimate a phase-dependent mean cycle from training data, define observed deviations from it, and forecast their subsequent evolution from complete observed histories. Evaluate several initialization phases and horizons on independent samples using A2's split rules. Estimate phase from available observations; if retrospective phase alignment is used only for scoring, label it and provide a physical-time analysis as well.

Compare the full model with zero residual/mean-cycle prediction, single-joint persistence, and a regularized multivariate autoregressive predictor of residuals. Report error by output joint, signed trajectory agreement, amplitude calibration, and incremental improvement. If selecting episodes with a large initial knee deviation, define that selection on the initialization only, not on its future trajectory, and use the same rule for all models.

Do not infer a causal knee-to-hip effect merely because knee deviations precede hip deviations. Initial deviations are correlated across joints and may share unobserved causes. A full-state prediction and a hypothetical single-joint intervention answer different questions.

**Supportive outcome.** The fitted dynamics improve prediction of observed deviations beyond a phase template, especially for joint-resolved evolution, with consistent subject-level benefit. Agreement between physically interpretable model responses and forecast behavior is supportive but does not identify a human input matrix.

**If it fails.** State that accuracy is largely trajectory prediction. Reduce human response claims, even if simulation validation succeeds, and consider a more modest paper about reproducible model summaries.

**Deliverable:** A compact deviation-forecast panel in Fig. 3 and detailed phase/joint results in SI.

### A10. Make paretic/nonparetic asymmetry the primary stroke analysis

**Why this is needed.** The post-stroke groups differ strongly in speed, with little or no control/LF overlap. Global gain differences are therefore hard to interpret. A within-person, anatomically directed asymmetry is more informative, although it is not automatically independent of speed or waveform asymmetry.

**How to do it.** Verify side labels and the control-side convention from acquisition metadata. Never assign healthy “paretic” sides based on whichever side gives the desired result. For the original map, define output-joint mass and paired asymmetry explicitly:

```text
output_mass_i = sum_over_input_joints R_ij
asymmetry_knee = output_mass_paretic_knee - output_mass_nonparetic_knee
```

Keep current Fig. 6G's convention if it uses a mean rather than a sum, but state the scaling. Analyze a paired log-ratio as a complementary measure. If maps are effectively rank one, test output expression and input receptivity separately. Reduced paretic-knee output does not by itself imply reduced paretic-knee input.

Establish healthy left–right asymmetry versus speed in HYA, with subject-level intervals and an a priori meaningful equivalence range if equivalence is claimed. Then show stroke asymmetry by participant, joint, speed, and impairment. Use a parsimonious repeated-measures model that distinguishes within-person speed variation from between-person average speed; with about 20 stroke participants, avoid a large covariate model. Report the available overlap and how dependent conclusions are on model form.

Verify how HF/LF were defined. If group membership is based on walking speed, a group trend cannot be presented as an independent impairment effect after “adjusting for speed.” Use available clinical impairment measures as separate exploratory associations and disclose missingness and confounding.

Repeat the knee result on mean-cycle surrogates and excess-response features, controlling for simple knee ROM and waveform asymmetry where estimable. Prespecify the knee as the principal hypothesis from the current draft; use hip and ankle as secondary anatomical comparisons, not automatic negative controls. Distinguish relative redistribution from an absolute reduction: a normalized knee share can fall because another output grows.

**Supportive outcome.** Healthy asymmetry is small across the observed speed range; stroke has a consistent paretic-side output deficit concentrated at the knee; the deficit survives waveform controls and remains compatible with an impairment association after feasible speed adjustment. It need not be exactly absent at hip or ankle.

**If it fails.** If the mean cycle explains the effect, report kinematic asymmetry. If speed and impairment cannot be separated, state a stroke-associated pattern in the observed sample rather than an impairment-specific mechanism. If uncertainty is too large, retain an exploratory result. Do not claim that bilateral healthy speed effects rule out speed–pathology interactions.

**Deliverable:** Proposed Fig. 4, centered on within-person asymmetry rather than global TG/Henrici/Q group means.

### A11. Use a speed-conditioned healthy reference only as a secondary analysis

**Why this is needed.** A healthy reference varying with speed could improve the interpretation of patient deviations, but pooling studies introduces assumptions about age, acquisition, processing, and behavior. Five controls in the post-stroke study cannot provide a strong equivalence test. Nonsignificant cohort separation is not proof that normalized maps transfer safely.

**How to do it.** Retrieve existing demographic and acquisition metadata, including age if available. Model the healthy expectation of log-ratio response features as a modest function of speed, with subject-held-out validation in HYA. Fit only within its supported speed range and characterize predictive uncertainty. Before applying to stroke, test predictions in the separate five-person control cohort using calibration and effect-size intervals, not just a difference-test p-value.

If transfer is sufficiently credible, form speed-conditioned deviations and propagate reference uncertainty into abnormality scores and shared axes. Analyze raw-map deviations and surrogate-excess deviations distinctly. Do not extrapolate to unsupported patient speeds. Very slow treadmill walking by young adults is not automatically an appropriate normative model for older impaired adults walking at the same speed.

**Supportive outcome.** The separate controls are predicted to a useful accuracy within the shared speed range, and patient patterns agree with the within-person asymmetry results under reasonable reference choices.

**If it fails or remains inconclusive.** Do not pool the cohorts for a clinical “normality” scale. Use study-specific descriptive references, emphasize within-person asymmetry, and label any transferred score exploratory. This analysis is not a prerequisite for the asymmetry result and should not become the sole foundation of it.

**Deliverable:** A transfer-calibration and common-support figure, probably in SI, with any justified speed-adjusted patient summary in Fig. 4 or 5.

### A12. Test individual specificity beyond speed, waveform, and diagnosis

**Why this is needed.** A roughly 39–41% identification rate against 1/42 chance is interesting but can exploit preferred speed, group, mean kinematics, anatomy, marker placement, or a single session. The useful claim is residual within-person reproducibility under demanding comparisons, not a clinical “control fingerprint.”

**How to do it.** First reconcile whether the original query is one trial or an average of half the trials. Report these separately if both are feasible. Construct a sequence of feature sets and evaluate each on identical eligible participants and splits:

1. Speed, stride frequency, bilateral ROM/asymmetry, and mean-waveform features alone.
2. Raw response maps or the supported receptivity/expression factors.
3. Mean-cycle-surrogate response features alone.
4. Surrogate-excess response features from A4.
5. Excess features after removing population associations with speed, stride frequency, ROM, a parsimonious set of mean-waveform components, and relevant group/study or impairment variables.

Fit nuisance regressions, waveform components, scaling, reference distributions, and any feature selection only on the permitted training/gallery or independent development data. Never fit subject-specific intercepts that use probe trials and then call the remainder an identity test. With limited subjects, use few nuisance dimensions or regularization and report the uncertainty of residualization.

Run three complementary tests:

| Test | Implementation | Interpretation |
| --- | --- | --- |
| Same-person versus different-person similarity | Independent trial halves, balanced contributions per person; compare at matched or adjusted state | Reproducibility within available recordings |
| Cross-speed identification | Gallery from a low-speed range and probes from a nonoverlapping high-speed range, then reverse; use comparable speed ranges across people where data allow | Specificity that transfers across locomotor state |
| Within-group identification | Restrict candidates to the same clinical group and, where useful, the same source study | Information beyond diagnosis or study membership |

Use held-out-session or day splits only if those already exist. If all trials share a session and marker placement, explicitly limit the conclusion to within-session reproducibility. Cross-speed performance does not resolve anatomy or calibration effects.

Define candidate-set chance for each test, normally `1/N_candidates` with balanced identities, and obtain permutation nulls respecting group and split structure. Repeated splits are not independent samples. Report participant-bootstrap intervals, balanced accuracy or per-person success, eligible sample sizes, and nearest-neighbor similarity distributions. Compare response features with kinematic baselines and combined kinematic-plus-response features; above-chance response identification alone does not prove incremental information beyond kinematics.

Recompute the shared stroke axes on justified references and features. Distinguish uncentered deviation energy from centered variance explained and report cumulative fractions; do not call a remaining component “most” of the signal without the corresponding calculation. Orient arbitrary singular-vector signs consistently. If axes are used predictively, learn them within training folds; otherwise label the embedding descriptive. A small variance-components analysis can compare state, group, and subject contributions, with intervals rather than a blanket claim that identity always dominates.

Finally, select example patients by a declared rule. Similar operator abnormality scores are not the same as matched clinical impairment. If clinical severity is not matched, label examples “similar overall response deviation, different spatial patterns.”

**Supportive outcome.** Residual features remain reproducible and informative across speed and within group, and add information beyond simple kinematics. The rate may fall substantially under these controls and still be useful if the remaining improvement is precise and interpretable.

**If it fails.** If identity is explained by waveform or speed, report that. If only unadjusted maps identify people, describe a kinematic/model signature. If no cross-speed signal remains, remove state-invariant individuality from the title and abstract. Do not infer long-term stability without repeated-session data.

**Deliverable:** Proposed Fig. 5 with matched baseline comparisons, exact candidate counts, and a transparent evaluation diagram.

### A13. Establish how short a recording can be

**Why this is needed.** “Short recordings” is a central promise. A fixed 15-second application demonstrates feasibility at that duration, but does not establish the minimum data needed or whether one trial reliably estimates a person's signature. Identification from averaged half-sets cannot support a single-short-record claim by itself.

**How to do it.** Construct a feasible ladder of complete stride counts within the available recordings and report their durations at each speed. Refit at each length using a fixed or prespecified complexity-selection rule. In simulation, evaluate true perturbation-response error across length and process/measurement noise. In humans, assess map/factor reliability, knee-asymmetry stability, deviation-forecast improvement, and single-trial identification where possible.

Compare independently sampled recordings or nonoverlapping segments. A short segment compared with a longer fit containing the same data gives an optimistic reliability estimate; label such nested comparisons and supplement them with independent tests. Include length-matched surrogates. Do not treat a full-trial human estimate as ground truth or extrapolate beyond the durations actually observed.

**Supportive outcome.** Useful response prediction or reproducibility is achieved within the actual short recordings, with a stated stride requirement and uncertainty. Some metrics may require more data than others.

**If it fails.** Report the practical minimum supported by the data, aggregate trials if necessary, and revise the title and feasibility claims accordingly. Do not promise a ten-second personal signature if the evidence requires several trials.

**Deliverable:** Reliability-versus-stride-count curves; a main-text practical statement supported by detailed SI.

### A14. Rebuild the biological interpretation and presentation

**Why this is needed.** Even successful analyses would not justify several current statements about neural control, compensation, clinical action, or superiority to classical stability measures. The new paper's strength should come from explicit tests of a clear biological hypothesis.

**How to do it.** Make these changes throughout, including titles, abstract, captions, and discussion:

| Current formulation or emphasis | Replacement |
| --- | --- |
| “Which joint drives/is responsible for another” in human data | “Model-implied joint-resolved response,” with input/output definitions |
| “Control is routed away from the paretic knee” | “Reduced relative response expression at the paretic knee,” if that is the verified output result |
| “Classic stability fails” | Response organization complements scalar stability; distinguish DMD eigenvalues, Floquet multipliers, and local divergence estimates |
| “Control fingerprint” | “Individual-specific response signature,” qualified by the available sessions and controls |
| Circumduction or hip-hiking inferred from sagittal flexion | Remove the mechanism claim; those movements are not observed by these channels |
| Identification of the joint that needs strengthening or an assistive device setting | A hypothesis for future intervention studies, not an established use |
| Promised assistive-nudge results in simulations and humans, p. 5 | Delete unless completed results actually exist; the proposed validation simulation does not establish therapeutic benefit |
| Stable personal differences over time | Within-session or cross-session reproducibility, exactly as tested |

Keep the negative clinical comparison: the present analyses do not establish added predictive value beyond walking speed. If clinical prediction is revisited, compare speed-only, feature-only, and combined models with subject-level outer validation and feature screening inside training folds. Given the small and incomplete clinical sample, keep this exploratory and in SI unless a robust, prespecified result emerges. Failure of clinical prediction does not invalidate a descriptive dynamics result, but it does limit clinical claims.

Move most oscillator, SLIP, and robot material to SI. Preserve the one-way oscillator as a directional mathematical control. If keeping SLIP decoders, compare with simple Fourier/waveform features using identical validation and consider held-out regions of the parameter grid rather than only nearby-point interpolation. If waveform features perform equally well, describe parameter-associated kinematic decoding, not unique recovery of stability. Do not add a large survey of standard stability estimators merely to win a benchmark; if making claims about one, compute that estimator correctly or narrow the claim.

Reduce the Introduction to the biological question, the missing multivariate description, the inference hypothesis, and the study design. Put the essential equations and coordinate/input assumptions in the main text. Report participant demographics, group definitions, recording protocol, exclusions, ethics, data sources, and code availability. If those metadata cannot be recovered, disclose the gap and reduce corresponding claims.

**Supportive outcome.** Every biological statement can be traced to a validated result, a stated model assumption, or a clearly labeled future hypothesis. The five main figures carry one coherent argument.

**If it fails.** Remove claims whose evidence is missing. If the core response-specific evidence does not survive, write the strongest methods-and-kinematics paper supported by the data rather than retaining a PNAS-shaped narrative.

**Deliverable:** A complete rewritten manuscript and SI, with a claim-to-evidence table for internal review.

## 5. Decision criteria before choosing the final framing

The following are scientific decisions, not predictions of journal acceptance.

| Result pattern | Recommended framing |
| --- | --- |
| Mean-cycle controls leave reproducible biological information; equal-stride speed effects persist; multijoint perturbation predictions succeed; human asymmetry and residual individuality survive their controls | The full proposed PNAS response-organization framing is defensible to pursue |
| Response is reliably low rank, with surrogate-resistant receptivity/expression effects | Keep the broader framing but simplify it around those two factors; low rank alone is not a reason to abandon it |
| Simulation validation succeeds but one human headline fails | Remove that headline and assess the narrower paper on its actual biological contribution; do not retain “individualized” or impairment-independent claims by default |
| Human maps are reproducible but physical perturbation validation fails | Present an observational model-summary method; consider the specialist methods-plus-application route discussed for Journal of the Royal Society Interface |
| Mean-cycle surrogates reproduce the pathology/identity effects without useful residual prediction | Present waveform organization rather than a newly identified response property |
| Key results depend on clipping, high rank, or one processing convention | Resolve the numerical dependence before making strong biological claims |
| Estimates are too imprecise to distinguish these possibilities | Report the limitation; an inconclusive test is not a pass |

Before submission, complete a compact internal checklist:

- [ ] Source data, figures, numerical claims, and metric definitions agree.
- [ ] Forecast evaluation excludes fitting and preprocessing leakage.
- [ ] The selected response representation survives waveform and numerical controls.
- [ ] The physical response quantity is defined and tested in simulation.
- [ ] Speed results use matched strides and compatible horizons.
- [ ] Stroke conclusions match the asymmetry, speed-support, and waveform evidence.
- [ ] Identity comparisons include kinematic baselines, cross-speed/within-group tests where feasible, and accurate session limits.
- [ ] Recording-length claims match the number of trials and strides actually needed.
- [ ] Clinical findings and limitations are reported without treatment claims.
- [ ] The title, abstract, captions, and discussion have been rewritten to reflect all failed as well as successful tests.

## 6. Manuscript length and submission structure

Plan for five focused main figures, with numerical validation and broad parameter sweeps in SI. Do not squeeze seven existing multipanel figures into a nominal six-page target.

The official NAS 2025 PNAS research-article template, checked September 25, 2026, specifies a 50–120-word significance statement and a single-paragraph abstract of at most 250 words. It describes six pages as the preferred Direct Submission length and twelve as the maximum; approximately 4,000 words and four medium graphical elements are a six-page guide, not independent allowances. Use the current template to assess actual space and verify requirements again at submission. [Official PNAS template](https://www.overleaf.com/latex/templates/template-for-preparing-your-research-report-submission-to-pnas-using-overleaf-2025/whbdryzwztnd).

Five main figures are a narrative proposal, not a claim that they will fit on six pages. If needed, combine the compact forecast/sampling panels or move secondary summaries to SI while retaining the decisive controls in the main paper.

## 7. Proposed title, significance statement, abstract, and full paper outline

**Conditional manuscript language:** The following is a writing target for the outcome in which the relevant analyses succeed. It is not a statement that those results have already been obtained. Before use, replace unsupported clauses and add a small number of verified effect sizes or prediction improvements to the abstract. In particular, the title requires short-record reliability and residual individual specificity; the perturbation claims require A7–A8; the human claims require A4, A6, and A9–A12.

### Proposed title

**Short recordings of walking reveal individualized response organization in human locomotion**

### Proposed significance statement

Walking is commonly summarized by average movement or a single measure of stability. These summaries do not describe how deviations evolve across the joints of a moving body. We show that short recordings of walking contain reproducible information about this response organization. Models learned from unperturbed movement predict imposed perturbation responses in a simulated multijoint walker. In humans, model-implied response patterns change with walking speed, become asymmetric after stroke, and retain individual structure beyond average gait and simple movement descriptors. These findings provide a description of locomotor dynamics that complements conventional gait analysis and generates testable hypotheses about how people differ in their responses to disturbances.

### Proposed abstract

Walking involves small deviations whose evolution depends on the coordinated dynamics of a multijoint body. Average gait trajectories and scalar stability measures do not fully describe this response organization. Here we ask whether it can be inferred from short recordings of walking. We fit linear dynamical models to delay-embedded joint kinematics and evaluate their finite-time, joint-resolved responses using explicit input and output mappings. In a simulated multijoint walker, models learned from unperturbed trajectories predict the propagation of subsequently imposed perturbations across joints. Comparisons with repeated mean cycles and matched-noise surrogates identify reproducible response structure beyond periodic waveform content. In healthy humans, this structure varies systematically with walking speed after matching stride counts, and the models improve prediction of held-out deviations. In post-stroke walking, model-implied responses show a paretic-side asymmetry concentrated at the knee that persists beyond mean-waveform differences and remains evident under feasible speed adjustment. Response signatures also retain individual information across walking speeds and within clinical groups after accounting for simple kinematic descriptors. Together, these results show that walking kinematics contain information about locomotor response organization beyond average movement. Perturbation validation in simulation and observational evidence in humans establish complementary grounds for using these models to study state-dependent and individual variation in locomotor dynamics.

### Full paper outline

#### Opening/Introduction: four paragraphs

1. **The biological problem.** Small deviations accompany ordinary walking. Their evolution is multijoint and occurs over the next steps. Motivate differences in response organization between walkers without claiming to have identified neural compensation or fall risk.
2. **What existing descriptions capture.** Average trajectories, speed, and symmetry describe movement; orbital and local stability measures describe specific aspects of convergence/divergence. Explain the additional question of magnitude, timing, and joint distribution of deviations. Avoid treating all stability estimators as a single “classic number.”
3. **The inference hypothesis.** Temporal structure in kinematics may support an empirical response model. Explain delay coordinates in one sentence and state the key limitation: predictive information about observed movement does not automatically identify a physical input or a neural controller. This motivates a perturbable simulated walker and waveform nulls.
4. **The tests.** Preview the order: simulation validation, mean-cycle and numerical controls, healthy speed dependence, stroke asymmetry, and individual specificity. State that human perturbation responses are not directly measured in these datasets.

Keep the opening accessible but precise. Remove the extended car, domino, and wine-glass analogies; one concrete multijoint example is enough. No result-free promise of assistive control should remain.

#### Result 1. Models learned from locomotion predict perturbation propagation in a multijoint walker

**Argument:** Establish the meaning and limits of “response” before presenting human maps.

Start with a compact definition of the delay model, its observed outputs, and the physical perturbation mapping. Present the unperturbed training protocol and held-out imposed perturbations. Show both signed time-course prediction and a summary across all tested phases, inputs, and parameter settings. Compare with periodic and autoregressive baselines. Explain any condition under which only cycle-averaged response strengths are recovered.

**Figure 1. Inferring joint-resolved responses from locomotion.**

- **A:** Conceptual illustration of two possible responses to the same initial deviation. Mark it as a schematic unless it is an actual tested pair of simulations.
- **B:** Pipeline: observed histories → fitted operator → specified present-state input → predicted current joint outputs.
- **C:** Multijoint walker and training/testing design, with no test perturbations used to fit the final model.
- **D:** True versus predicted signed responses at several output joints after one representative perturbation, selected by a declared rule.
- **E:** Measured versus predicted finite-horizon response-strength matrices on matched scales, including a baseline.
- **F:** Summary of held-out prediction error across phase, joint, model setting, and perturbation amplitude; identify the linear range and uncertainty.

**Closing inference:** Observations can support perturbation-response prediction in this tested mechanical system when the relevant dynamics are observable and sampled. The result does not identify the human controller.

**Depends on:** A2, A3, A7, A8.

#### Result 2. Reproducible response organization extends beyond the mean gait cycle

**Argument:** Establish that the representation used in the human results contains more than periodic waveform features or poorly determined fitted directions.

Introduce the surrogate hierarchy and ask which parts of the response are explained by each level. Define the excess-response feature only after showing the raw and surrogate maps. Establish whether two input/output factors or a higher-rank representation is warranted. Include a concise numerical robustness result for the features that survive; move exhaustive grids to SI.

**Figure 2. Separating mean-cycle structure from additional response organization.**

- **A:** Example real recording and the repeated-cycle, matched-residual, and phase-randomized controls, with what each preserves.
- **B:** Real map, surrogate expectation, and log-ratio excess map on appropriate separate scales; alternatively show receptivity/expression factors if rank one dominates.
- **C:** Singular-value spectra and rank-one reconstruction errors for real and surrogate maps.
- **D:** Real versus surrogate values of the principal biological effects, with repeatability of the residual; avoid a panel limited to matrix correlation alone.
- **E:** Compact stability of the retained conclusions across parsimonious rank and fitting conventions.

**Closing inference:** The mean waveform may account for shared organization, while the measured excess contains reliable additional information. If there is no such excess, this section and the paper's central claim must change.

**Depends on:** A3–A5 and endpoint-specific analyses from A6/A10/A12.

#### Result 3. Human response organization changes with locomotor state

**Argument:** Establish systematic within-person state dependence with the sampling confound addressed.

Describe the 17-person healthy dataset and equal-stride design. Present actual held-out prediction, including deviations from the mean cycle. Then show the surviving speed relationships in output organization or input/output factors. Keep global gain, Q, and participation ratio only if their numerical and surrogate controls justify their interpretation. Report the data duration required for a useful estimate.

**Figure 3. Speed-dependent response organization in healthy walking.**

- **A:** Participant speed coverage, equal-stride windows, and actual time durations.
- **B:** Held-out deviation-forecast improvement over the mean-cycle and autoregressive baselines, by horizon with subject-level uncertainty.
- **C:** The small set of robust global response summaries versus speed, with within-person trajectories; omit this panel if those metrics fail the controls.
- **D:** The main speed-dependent changes in excess-response maps or receptivity/expression factors.
- **E:** Recording-length reliability or a concise fixed-stride versus fixed-duration comparison, whichever most directly supports the principal result; the other goes in SI.

**Closing inference:** Behavioral state changes the observed response organization. This is not a claim that faster walking is universally more stable or that all apparent speed effects are physiological.

**Depends on:** A2–A6, A9, A13.

#### Result 4. Stroke is associated with asymmetric response organization centered on the paretic knee

**Argument:** Make the within-person anatomical finding the central stroke result, and show exactly how much confounding has been addressed.

Show the cohort and speed overlap before comparing responses. Introduce healthy left–right asymmetry as an empirical control, then quantify paretic/nonparetic asymmetry and its anatomical distribution. Present the effect after mean-cycle controls and feasible within-stroke speed adjustment. If a normative transfer model is used, summarize its limits rather than treating it as established by nonsignificant cohort differences.

**Figure 4. Paretic-side asymmetry in model-implied joint responses.**

- **A:** Participants, clinical-group definitions, and trial speed distributions, emphasizing the independent sample sizes.
- **B:** Healthy left–right asymmetry versus speed, with uncertainty and the observed range.
- **C:** Stroke paretic/nonparetic output asymmetry at hip, knee, and ankle, with each participant shown; distinguish input effects if they are also analyzed.
- **D:** Estimated knee effect under feasible speed/impairment adjustment, including sensitivity to overlap and model choice.
- **E:** Knee asymmetry in real data, mean-cycle controls, and excess-response features; an anatomical map or factor display localizes the result.

**Closing inference:** The observed stroke-associated change is directionally and anatomically organized, with the specificity supported by these controls. Relative gain redistribution is not evidence that neural control or mechanical load was transferred to a particular joint.

**Depends on:** A4–A7, A10; A11 only for any transferred normative claim.

#### Result 5. Response organization retains reproducible individual structure

**Argument:** Separate a shared stroke pattern from individual variation, then test whether individual information remains beyond ordinary kinematics and diagnosis.

Define the justified reference and residual features. Show descriptive shared axes with correct variance accounting, followed by independent-split similarity and the harder cross-speed/within-group identification tests. Compare simple kinematics, surrogate-derived features, response residuals, and combined features under identical splits. Use matched-example maps to illustrate spatial differences, with the matching variable labeled precisely.

**Figure 5. Individual response signatures beyond locomotor state and group.**

- **A:** Feature construction and leakage-free gallery/probe split, including nuisance adjustment.
- **B:** Shared stroke deviation axes or factor variation, with explained variance/energy explicitly defined and reference uncertainty acknowledged.
- **C:** Same-person versus different-person residual similarity, with subject-level uncertainty.
- **D:** Cross-speed identification in both directions, alongside kinematic and surrogate baselines and the exact candidate-set chance rate.
- **E:** Within-group identification with group-specific candidate counts; include session-held-out performance only if existing data support it.
- **F:** Two or three examples with similar overall response deviation but different reproducible spatial patterns; label clinical severity matching only if it was actually performed.

**Closing inference:** The available recordings contain individual response information beyond the tested nuisance descriptions. State whether this is within-session reproducibility, cross-session reproducibility, or both. Do not imply longitudinal biomarker stability.

**Depends on:** A4, A5, A10–A13.

#### Discussion: five focused parts

1. **What was learned.** State the supported finding in ordinary language: kinematics can contain information about response organization beyond the average gait. Recap simulation truth, healthy state variation, stroke asymmetry, and individuality only to the extent each survived.
2. **Relationship to stability and coordination.** Explain how a joint-resolved empirical response complements trajectory, Floquet, local-divergence, and muscle-synergy descriptions. Acknowledge the coordinate dependence of operator norms and the possible low-dimensional factor structure.
3. **Biological interpretation.** Discuss why a paretic-knee output deficit and residual individual variation are interesting hypotheses about locomotor organization. Separate observed association from compensatory mechanism, and avoid attributing sagittal findings to unobserved circumduction or hip-hiking.
4. **Limits of inference.** Address modest participant numbers, speed/impairment confounding and support, age and study transfer, session effects, only sagittal kinematics, excitation, model/input assumptions, contact-phase dependence, and the absence of imposed human perturbations. Explain what a failed or inconclusive control changed in the interpretation.
5. **Practical significance and next experiments.** State the verified recording-length requirements and current clinical prediction result. Propose future human perturbation and longitudinal studies to test intervention relevance; do not claim they have been performed or that a treatment can already be selected.

#### Materials and Methods: complete logical specification

1. **Datasets and provenance:** acquisition sources, subject/trial counts, demographics, HF/LF definitions, missing clinical scores, side labels, protocols, ethics, and exclusions.
2. **Preprocessing and gait phase:** units, filtering, centering/scaling, sampling, event detection, mean-cycle estimation, and treatment of boundaries and missing data.
3. **Data partitioning:** subject/trial/time splits, available initialization data, nested tuning, and leakage safeguards.
4. **Operator fitting:** delay construction, admissible rank selection, regularization, raw spectra, stabilization, and numerical tolerances.
5. **Response definitions:** current-state versus history-space inputs, output selectors, norms/units, finite horizons, stride frequency, resolvent convention, Q handling, and normalization.
6. **Surrogates and decomposition:** repeated-cycle construction, residual matching, phase randomization, numerical rank, log-ratio excess, and input/output factor analysis.
7. **Multijoint simulation:** equations or implementation reference, controller/contact rules, variability, perturbation mappings, training/testing separation, amplitudes/phases, and ground-truth comparisons.
8. **Human analyses:** equal-stride speed models, deviation forecasts, asymmetry, optional normative transfer, individual signatures, and recording-length reliability.
9. **Statistics:** independent units, repeated-measures structure, uncertainty, permutation schemes, multiplicity, effect-size definitions, and exploratory versus validation analyses.
10. **Reproducibility:** data-access conditions, code/environment versions, seeds, results tables, and commands/workflows that regenerate figures and reported values.

Keep the essential inference and input/output definitions in the main article. Put extended derivations, simulation settings, and full robustness grids in SI so that the main text remains understandable on its own.

#### Supporting Information plan

- **S1. Definitions and numerical checks:** exact metric formulas, coordinate norms, dimensional checks, raw/stabilized spectra, resolvent conditioning, frequency-grid and Q sensitivity.
- **S2. Model selection and forecast validation:** rank/delay/regularization grid, partition diagrams, all baseline comparisons, and full joint/horizon errors.
- **S3. Surrogate construction and complete results:** marginal-statistic matching, realization counts, numerical-rank handling, all endpoint null comparisons, and residual uncertainty.
- **S4. Multijoint walker validation:** model specification, all phases and perturbation directions, amplitude/noise sweeps, negative controls, failures, and optional matched-gait examples.
- **S5. Existing mathematical validation:** reconciled Stuart–Landau analyses, one-way oscillator control, and SLIP results with the return-map estimator correctly identified and waveform decoder baselines where retained.
- **S6. Robot applications:** verified forecasts and joint organization, with causal/controller interpretations limited to what the known setup actually supports.
- **S7. Human robustness:** equal-stride versus equal-duration fits, time versus phase conventions, healthy asymmetry, study-transfer calibration, and sensitivity to speed support and missing covariates.
- **S8. Individual specificity and clinical benchmarks:** all candidate sets and split rules, waveform/ROM/speed baselines, group/session restrictions, shared-axis stability, exploratory clinical comparisons, and multiplicity accounting.
- **S9. Recording-length analysis:** independent estimates across feasible stride counts and durations, including uncertainty and any failures at the shortest lengths.
- **Tables:** participant/trial accounting and metadata; metric definitions; prespecified/exploratory endpoints; main-claim robustness summary; data/code availability.

#### Back matter

Provide verified author affiliations and contributions, acknowledgments and funding, competing interests, ethics and consent statements, data/code availability, and complete references for the source datasets and methods. Keep unresolved administrative or citation placeholders out of the submission version.

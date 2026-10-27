"""
# Paper-exact stability analysis (Rummel et al. 2010)

The generic hybrid-Floquet machinery in `stability.jl` uses the full
`(y, xdot, ydot)` reduced state at VLO and a full-stride (same-leg-to-
same-leg) return map. Comparing against Rummel, Blum, Maus, Rode &
Seyfarth, "Stable and Robust Walking with Compliant Legs" (ICRA 2010)
exposed two mismatches with how *that* paper (and the wider
Seyfarth/Geyer SLIP-walking literature) actually does this analysis:

1. Their model has **no hip spring** -- pass `params.kth = [0,0]` to match it.
2. Critically, they use a **single-step** (VLO of one leg to VLO of the
   *other* leg) Poincare map in **reduced (y, theta) coordinates**, where
   `theta` is the velocity angle from horizontal and the velocity
   *magnitude* is not a free state variable at all -- it is fixed by total
   mechanical energy `E` via energy conservation (their eq. 3, identical to
   [`biped_slip_energy_ic`](@ref)). Because `E` is a chosen parameter
   rather than part of the dynamical state, this map has no exactly (or
   near-)conserved direction, unlike the full 3D `(y, xdot, ydot)` map used
   in `stability.jl`, which -- for this conservative model -- has one
   Floquet multiplier pinned at (or very near) `+1` corresponding to the
   energy direction, making the raw spectral radius a poor/misleading
   proxy for the transverse (physically relevant) stability.

Reproducing their published example (pattern A: k~ = 20, dimensionless
energy Ebar = 1.05, angle of attack 69 deg from horizontal) with the
functions below gives a genuinely stable fixed point with Floquet
multipliers `0.628 +/- 0.713i` (`max|lambda| = 0.95`), matching the shape
of their Fig. 5(a) -- confirming that self-stable walking is real, and
that the earlier "razor-thin marginal stability" conclusion from
`stability.jl`'s full 3D analysis was a methodological artifact (the
unreduced energy direction, compounded by the extraneous hip spring), not
a property of the physical model.
"""

"""
    poincare_step(y, theta, Ebar, stance_leg, params; opts=...) -> (y1, theta1)

Single-step (VLO-to-opposite-leg-VLO) Poincare map, following Rummel et
al. (2010). `theta` is the velocity angle from horizontal at VLO (their
`theta0 = arctan(ydot0/xdot0)`); the velocity magnitude is fixed by the
dimensionless total energy `Ebar = E/(L0*m*g)` via
[`biped_slip_energy_ic`](@ref). Returns `(y1, theta1)` at VLO of the
*opposite* leg one step later.

Note `opts.max_steps` must be at least 3: [`simulate_biped_slip`](@ref)
stops exactly at the transition event that advances `n_steps` to
`max_steps`, so `max_steps=2` would stop right at the trailing leg's
liftoff and never actually integrate the following single-support phase
in which VLO typically occurs.
"""
function poincare_step(y::Real, theta::Real, Ebar::Real, stance_leg::Int,
                        params::BipedSLIPParams;
                        opts::SimOptions=SimOptions(t_max=5.0, max_steps=3, dt=0.0005))
    E = Ebar * params.L0[stance_leg] * params.m * params.g
    ic = biped_slip_energy_ic(params, stance_leg, y, E, theta)
    results = simulate_biped_slip(params, ic, opts)

    if results.fallen
        error("poincare_step: walker fell before completing a step")
    end
    if results.n_steps < 2
        error("poincare_step: only $(results.n_steps)/2 transitions completed; " *
              "increase opts.t_max/opts.max_steps")
    end

    opp = stance_leg == LEFT ? RIGHT : LEFT
    th = opp == LEFT ? results.thL : results.thR

    # the opposite leg is on the ground continuously from the moment it
    # touches down (first transition) through the end of the following
    # single-support phase -- VLO can occur anywhere in that combined
    # range, not just after the trailing leg lifts off
    initial_mode = stance_leg == LEFT ? SS_LEFT : SS_RIGHT
    idx_td = findfirst(m -> m != initial_mode, results.mode)
    idx_td === nothing && error("poincare_step: never left initial stance")
    idx_end = length(results.mode)

    cross = findfirst(i -> th[i] > 0 && th[i + 1] <= 0, idx_td:(idx_end - 1))
    cross === nothing && error("poincare_step: no VLO crossing found")
    i = idx_td - 1 + cross

    th0, th1_ = th[i], th[i + 1]
    frac = th0 / (th0 - th1_)
    X0 = @view results.X[i, :]
    X1 = @view results.X[i + 1, :]
    Xc = X0 .+ frac .* (X1 .- X0)

    y1 = Xc[2]
    theta1 = atan(Xc[4], Xc[3])
    return y1, theta1
end

"""
    find_periodic_gait_step(y0, theta0, Ebar, stance_leg, params; opts=..., h=1e-6,
                             xtol=1e-9, maxiter=50, min_step=1e-4) -> (z_star, n_iter)

Damped Newton on `poincare_step(y, theta, Ebar, ...) - (y, theta) = 0`, in
the reduced `(y, theta)` coordinates at fixed energy (see module
docstring). Same backtracking-on-failure strategy as
[`find_periodic_gait`](@ref).
"""
function find_periodic_gait_step(y0::Real, theta0::Real, Ebar::Real, stance_leg::Int,
                                  params::BipedSLIPParams;
                                  opts::SimOptions=SimOptions(t_max=5.0, max_steps=3, dt=0.0005),
                                  h::Real=1e-6, xtol::Real=1e-9, maxiter::Int=50,
                                  min_step::Real=1e-4)
    G(z) = collect(poincare_step(z[1], z[2], Ebar, stance_leg, params; opts=opts))
    F(z) = G(z) .- z
    safe_F(z) = try
        F(z)
    catch
        nothing
    end

    z = [float(y0), float(theta0)]
    for iter in 1:maxiter
        Fz = F(z)
        norm(Fz) < xtol && return z, iter
        J = finite_diff_jacobian(F, z; h=h)
        dz = J \ Fz

        step = 1.0
        z_new = z .- step .* dz
        Fz_new = safe_F(z_new)
        while (Fz_new === nothing || norm(Fz_new) >= norm(Fz)) && step > min_step
            step /= 2
            z_new = z .- step .* dz
            Fz_new = safe_F(z_new)
        end

        if Fz_new === nothing
            error("find_periodic_gait_step: could not find a survivable step " *
                  "from z=$z")
        end
        z = z_new
    end
    error("find_periodic_gait_step: did not converge in $maxiter iterations " *
          "(last residual norm = $(norm(F(z))))")
end

"""
    floquet_multipliers_step(z_star, Ebar, stance_leg, params; opts=..., h=1e-6)
        -> (multipliers, J)

2x2 finite-difference Jacobian of [`poincare_step`](@ref) at the periodic
fixed point `z_star = [y*, theta*]`, and its eigenvalues -- the Floquet
multipliers as computed in Rummel et al. (2010).
"""
function floquet_multipliers_step(z_star::AbstractVector, Ebar::Real, stance_leg::Int,
                                   params::BipedSLIPParams;
                                   opts::SimOptions=SimOptions(t_max=5.0, max_steps=3, dt=0.0005),
                                   h::Real=1e-6)
    G(z) = collect(poincare_step(z[1], z[2], Ebar, stance_leg, params; opts=opts))
    J = finite_diff_jacobian(G, z_star; h=h)
    return eigen(J).values, J
end

"""
    gait_stability_step(y0, theta0, Ebar, stance_leg, params; opts=..., h=1e-6,
                         xtol=1e-9, maxiter=50) -> GaitStability

Convenience wrapper mirroring [`gait_stability`](@ref) but using the
paper-exact reduced single-step map: locates the periodic gait with
[`find_periodic_gait_step`](@ref), then its Floquet multipliers with
[`floquet_multipliers_step`](@ref).
"""
function gait_stability_step(y0::Real, theta0::Real, Ebar::Real, stance_leg::Int,
                              params::BipedSLIPParams;
                              opts::SimOptions=SimOptions(t_max=5.0, max_steps=3, dt=0.0005),
                              h::Real=1e-6, xtol::Real=1e-9, maxiter::Int=50)
    z_star, _ = find_periodic_gait_step(y0, theta0, Ebar, stance_leg, params;
                                         opts=opts, h=h, xtol=xtol, maxiter=maxiter)
    multipliers, _ = floquet_multipliers_step(z_star, Ebar, stance_leg, params; opts=opts, h=h)
    rho = maximum(abs, multipliers)
    stable = rho < 1
    decay_steps = stable ? 1 / (-log(rho)) : Inf
    return GaitStability(z_star, multipliers, rho, stable, decay_steps)
end

"""
    VLOEvent

One Vertical Leg Orientation crossing extracted from a
[`BipedSLIPResults`](@ref) trajectory by [`extract_vlo_events`](@ref).

- `t`     : time of the crossing [s] (linearly interpolated between samples)
- `leg`   : which leg is vertical at this crossing (`LEFT`/`RIGHT`)
- `y`     : CoM height at the crossing [m]
- `theta` : velocity angle from horizontal, `atan(ydot, xdot)` [rad]
- `xdot`, `ydot` : CoM velocity components at the crossing [m/s]
"""
struct VLOEvent
    t::Float64
    leg::Int
    y::Float64
    theta::Float64
    xdot::Float64
    ydot::Float64
end

"""
    extract_vlo_events(results::BipedSLIPResults) -> Vector{VLOEvent}

Scan an entire (possibly many-stride) trajectory for every Vertical Leg
Orientation crossing of either leg, i.e. every point sampled by the
single-step Poincare map used in [`poincare_step`](@ref), but over the
whole trajectory at once rather than stopping after one step. Events are
sorted by time and alternate legs in a healthy periodic (or
near-periodic) gait. This is the natural way to turn one long simulated
trajectory into the discrete per-stride `(y, theta)` sequence for
comparing a data-driven (e.g. DMD) estimate of the stride-to-stride
return map against the analytical Floquet multipliers from
[`floquet_multipliers_step`](@ref).
"""
function extract_vlo_events(results::BipedSLIPResults)
    n = length(results.T)
    events = VLOEvent[]
    for (leg, th) in ((LEFT, results.thL), (RIGHT, results.thR))
        for i in 1:(n - 1)
            th_i, th_ip1 = th[i], th[i + 1]
            if !isnan(th_i) && !isnan(th_ip1) && th_i > 0 && th_ip1 <= 0
                frac = th_i / (th_i - th_ip1)
                t = results.T[i] + frac * (results.T[i + 1] - results.T[i])
                y = results.X[i, 2] + frac * (results.X[i + 1, 2] - results.X[i, 2])
                xdot = results.X[i, 3] + frac * (results.X[i + 1, 3] - results.X[i, 3])
                ydot = results.X[i, 4] + frac * (results.X[i + 1, 4] - results.X[i, 4])
                theta = atan(ydot, xdot)
                push!(events, VLOEvent(t, leg, y, theta, xdot, ydot))
            end
        end
    end
    sort!(events, by=e -> e.t)
    return events
end

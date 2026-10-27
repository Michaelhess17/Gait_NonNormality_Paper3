"""
# Floquet / Poincaré-map stability analysis for the hybrid gait

The bipedal SLIP walker is a piecewise-smooth hybrid dynamical system (4
discrete modes glued together by touchdown/liftoff resets), not a single
smooth periodic ODE, so the classical continuous-time Floquet theorem
(monodromy matrix of a linear time-periodic ODE) does not apply directly.
The standard substitute used throughout the legged-locomotion literature
(e.g. Rummel et al. 2010, Full & Koditschek 1999, Seyfarth/Geyer) is a
**discrete stroboscopic Poincaré return map**: fix a section transverse to
the flow, sample the state once per stride, and linearize that discrete
map. Its eigenvalues are the Floquet (or "characteristic") multipliers of
the hybrid limit cycle, and stability/decay-rate follows exactly as in the
continuous case, just one step at a time instead of one instant at a time.

## Section and reduced coordinates

The natural section here is Vertical Leg Orientation (VLO, `th = 0`) of a
given stance leg -- the same convention [`biped_slip_vlo_ic`](@ref) already
uses. Horizontal position `x` is cyclic (translation-invariant) and drops
out, leaving a 3-dimensional reduced state `z = [y, xdot, ydot]` at VLO.

## The return map

Starting a stride at VLO on `stance_leg`, a full period is: touchdown of
the opposite leg, double support, opposite single support, touchdown of
`stance_leg` again, double support, and finally single support on
`stance_leg` again, in which the CoM passes back through VLO. That is
exactly 4 hybrid transitions plus the crossing partway through the 5th
segment. [`poincare_return`](@ref) integrates this with
[`simulate_biped_slip`](@ref) and locates the VLO crossing in the final
segment by linear interpolation of the stance leg angle's sign change.

## Fixed point and multipliers

[`find_periodic_gait`](@ref) Newton-iterates on `P(z) - z = 0` (using a
finite-difference Jacobian) to locate the periodic orbit near a starting
guess. [`floquet_multipliers`](@ref) then finite-differences `P` itself at
the fixed point; the eigenvalues of that 3x3 Jacobian are the Floquet
multipliers. The orbit is asymptotically stable iff all `|lambda_i| < 1`;
the slowest-decaying mode's magnitude sets the number of strides needed to
recover from a perturbation, `n ~ 1/(1 - |lambda_max|)`.

[`gait_stability`](@ref) is a convenience wrapper that does both steps and
reports a summary.
"""

using LinearAlgebra: eigen, norm

"""
    poincare_return(z, stance_leg, params; opts=SimOptions(...)) -> Vector{Float64}

Integrate one full stride starting from Vertical Leg Orientation (VLO) on
`stance_leg` with reduced state `z = [y, xdot, ydot]`, and return the
reduced state `[y, xdot, ydot]` the next time the CoM passes back through
VLO on the *same* leg (i.e. after both legs have had their turn).

Throws an error if the walker falls, or does not complete the 4
transitions (2 touchdowns + 2 liftoffs) needed to return to the same
stance leg, within `opts`.
"""
function poincare_return(z::AbstractVector, stance_leg::Int, params::BipedSLIPParams;
                          opts::SimOptions=SimOptions(t_max=10.0, max_steps=5, dt=0.001))
    y0, xdot0, ydot0 = z
    ic = biped_slip_vlo_ic(params, stance_leg, y0, xdot0, ydot0)
    results = simulate_biped_slip(params, ic, opts)

    if results.fallen
        error("poincare_return: walker fell before completing a full stride " *
              "(z=$z, n_steps=$(results.n_steps), T_end=$(results.T[end]))")
    end
    if results.n_steps < 4
        error("poincare_return: only $(results.n_steps)/4 transitions completed; " *
              "increase opts.t_max/opts.max_steps")
    end

    target_mode = stance_leg == LEFT ? SS_LEFT : SS_RIGHT
    th = stance_leg == LEFT ? results.thL : results.thR

    # locate the final contiguous run of `target_mode` samples: the
    # single-support segment in which the CoM returns to VLO
    idx_end = length(results.mode)
    idx_start = idx_end
    while idx_start > 1 && results.mode[idx_start - 1] == target_mode
        idx_start -= 1
    end
    if idx_start == 1
        error("poincare_return: walker never left the initial stance phase")
    end

    cross = findfirst(i -> th[i] > 0 && th[i + 1] <= 0, idx_start:(idx_end - 1))
    cross === nothing && error("poincare_return: no VLO (th = 0) crossing found " *
                                "in the final stance segment")
    i = idx_start - 1 + cross

    # linear interpolation between the bracketing samples for the exact crossing
    th0, th1 = th[i], th[i + 1]
    frac = th0 / (th0 - th1)   # in (0, 1]

    X0 = @view results.X[i, :]
    X1 = @view results.X[i + 1, :]
    Xc = X0 .+ frac .* (X1 .- X0)

    return [Xc[2], Xc[3], Xc[4]]   # [y, xdot, ydot] at the return VLO
end

"""
    finite_diff_jacobian(F, z; h=1e-6) -> Matrix{Float64}

Central-difference Jacobian of `F: R^n -> R^m` at `z`.
"""
function finite_diff_jacobian(F, z::AbstractVector; h::Real=1e-6)
    n = length(z)
    F0 = F(z)
    m = length(F0)
    J = zeros(m, n)
    for j in 1:n
        zp = copy(z); zp[j] += h
        zm = copy(z); zm[j] -= h
        J[:, j] = (F(zp) .- F(zm)) ./ (2h)
    end
    return J
end

"""
    find_periodic_gait(z0, stance_leg, params; opts=..., h=1e-6, xtol=1e-10,
                        maxiter=50, min_step=1e-4)
        -> (z_star, n_iter)

Damped (backtracking) Newton iteration on
`poincare_return(z, stance_leg, params) - z = 0`, starting from initial
guess `z0 = [y0, xdot0, ydot0]`, to locate the periodic-gait fixed point
of the Poincaré return map. Returns the converged reduced state `z_star`
and the number of iterations used.

A bare (undamped) Newton step can overshoot into a state from which the
walker falls before completing a stride -- `poincare_return` throws in
that case. This is treated as "step too large": the step is halved
(down to `min_step` of its original length) until it either lands on a
state with a smaller residual or a state where the simulator survives at
all. This does not fix genuine non-existence of an isolated fixed point
(e.g. the conservative, no-hip-spring, no-control case -- see module
docstring), which will still fail to converge.
"""
function find_periodic_gait(z0::AbstractVector, stance_leg::Int, params::BipedSLIPParams;
                             opts::SimOptions=SimOptions(t_max=10.0, max_steps=5, dt=0.001),
                             h::Real=1e-6, xtol::Real=1e-10, maxiter::Int=50,
                             min_step::Real=1e-4)
    P(z) = poincare_return(z, stance_leg, params; opts=opts)
    F(z) = P(z) .- z

    safe_F(z) = try
        F(z)
    catch
        nothing
    end

    z = collect(float.(z0))
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
            error("find_periodic_gait: could not find a step (down to " *
                  "min_step=$min_step of the Newton step) from which the " *
                  "walker survives a full stride at z=$z")
        end

        z = z_new
    end
    error("find_periodic_gait: did not converge in $maxiter iterations " *
          "(last residual norm = $(norm(F(z))))")
end

"""
    floquet_multipliers(z_star, stance_leg, params; opts=..., h=1e-6)
        -> (multipliers, J)

Finite-difference Jacobian `J` of the Poincaré return map at the periodic
fixed point `z_star`, and its eigenvalues (the Floquet multipliers of the
hybrid limit cycle).
"""
function floquet_multipliers(z_star::AbstractVector, stance_leg::Int, params::BipedSLIPParams;
                              opts::SimOptions=SimOptions(t_max=10.0, max_steps=5, dt=0.001),
                              h::Real=1e-6)
    P(z) = poincare_return(z, stance_leg, params; opts=opts)
    J = finite_diff_jacobian(P, z_star; h=h)
    return eigen(J).values, J
end

"""
    GaitStability

Summary returned by [`gait_stability`](@ref).

- `z_star`       : converged VLO fixed point `[y, xdot, ydot]`
- `multipliers`  : Floquet multipliers (eigenvalues of the return-map Jacobian)
- `spectral_radius` : `maximum(abs, multipliers)`
- `stable`       : `spectral_radius < 1`
- `decay_steps`  : approximate strides to decay a perturbation by `1/e`,
                   `1 / -log(spectral_radius)`; `Inf` if not stable
"""
struct GaitStability
    z_star::Vector{Float64}
    multipliers::Vector{ComplexF64}
    spectral_radius::Float64
    stable::Bool
    decay_steps::Float64
end

"""
    gait_stability(z0, stance_leg, params; opts=..., h=1e-6, xtol=1e-10, maxiter=50)
        -> GaitStability

Convenience wrapper: locate the periodic gait near `z0` with
[`find_periodic_gait`](@ref), then compute its Floquet multipliers with
[`floquet_multipliers`](@ref), and summarize stability.
"""
function gait_stability(z0::AbstractVector, stance_leg::Int, params::BipedSLIPParams;
                         opts::SimOptions=SimOptions(t_max=10.0, max_steps=5, dt=0.001),
                         h::Real=1e-6, xtol::Real=1e-10, maxiter::Int=50)
    z_star, _ = find_periodic_gait(z0, stance_leg, params; opts=opts, h=h,
                                    xtol=xtol, maxiter=maxiter)
    multipliers, _ = floquet_multipliers(z_star, stance_leg, params; opts=opts, h=h)
    rho = maximum(abs, multipliers)
    stable = rho < 1
    decay_steps = stable ? 1 / (-log(rho)) : Inf
    return GaitStability(z_star, multipliers, rho, stable, decay_steps)
end

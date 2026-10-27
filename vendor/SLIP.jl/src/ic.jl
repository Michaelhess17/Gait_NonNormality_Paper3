"""
    BipedSLIPIC

Initial-condition struct for [`simulate_biped_slip`](@ref).

- `mode`  : initial discrete mode (see [`SS_LEFT`](@ref) etc.)
- `state` : `[x0, y0, xdot0, ydot0]` initial CoM state
- `xfL`   : initial LEFT foot x-position (`NaN` if the left leg is not on
            the ground in `mode`)
- `xfR`   : initial RIGHT foot x-position (`NaN` if the right leg is not
            on the ground in `mode`)
"""
mutable struct BipedSLIPIC
    mode::Int
    state::Vector{Float64}
    xfL::Float64
    xfR::Float64
end

"""
    biped_slip_vlo_ic(params, stance_leg, y0, xdot0, ydot0) -> BipedSLIPIC

Build an initial-condition struct at Vertical Leg Orientation (VLO) for
[`simulate_biped_slip`](@ref).

At VLO the stance leg is exactly vertical (`th = 0`): the CoM is directly
above its planted foot at height `y0`, with horizontal/vertical velocity
`xdot0`/`ydot0`. This is the initial-condition convention used by Rummel
et al. (2010): a walking gait is fully specified (up to translation) by
the height and velocity at VLO of one single-support phase.

`stance_leg` is `LEFT` or `RIGHT`; the stance foot is placed at `x = 0`.
"""
function biped_slip_vlo_ic(params::BipedSLIPParams, stance_leg::Int, y0, xdot0, ydot0)
    if stance_leg == LEFT
        mode = SS_LEFT
        xfL  = 0.0
        xfR  = NaN
    else
        mode = SS_RIGHT
        xfR  = 0.0
        xfL  = NaN
    end

    return BipedSLIPIC(mode, [0.0, y0, xdot0, ydot0], xfL, xfR)
end

"""
    biped_slip_energy_ic(params, stance_leg, y0, E, theta0) -> BipedSLIPIC

Build a VLO initial condition from a target total mechanical energy,
following Rummel et al. (2010), eq. (3), generalized to include the
hip-spring potential energy.

At VLO (stance leg vertical, `th = 0`) the leg length equals `y0`, so the
total mechanical energy is
```
E = m*g*y0 + 0.5*k*(L0 - y0)^2 + 0.5*kth*th0^2 + 0.5*m*|v0|^2
```
Solving for the velocity magnitude:
```
|v0|^2 = (2/m)*(E - m*g*y0 - 0.5*k*(L0-y0)^2 - 0.5*kth*th0^2)
```
and splitting into components using the velocity angle `theta0` (angle of
the velocity vector above horizontal, as in Rummel 2010):
```
xdot0 = |v0|*cos(theta0),  ydot0 = |v0|*sin(theta0)
```
"""
function biped_slip_energy_ic(params::BipedSLIPParams, stance_leg::Int, y0, E, theta0)
    idx = stance_leg

    v0sq = (2 / params.m) * (E - params.m * params.g * y0
             - 0.5 * params.k[idx] * (params.L0[idx] - y0)^2
             - 0.5 * params.kth[idx] * params.th0[idx]^2)

    if v0sq < 0
        error("Requested energy E is too low for height y0 (negative " *
              "velocity^2). Increase E or reduce y0.")
    end

    v0 = sqrt(v0sq)
    xdot0 = v0 * cos(theta0)
    ydot0 = v0 * sin(theta0)

    return biped_slip_vlo_ic(params, stance_leg, y0, xdot0, ydot0)
end

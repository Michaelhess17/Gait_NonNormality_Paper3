"""
    FootPositions

Fixed horizontal foot positions currently active during a stance segment.
A leg not currently touching the ground carries `NaN` for its entry (it is
never read while that leg is airborne).
"""
struct FootPositions
    L::Float64
    R::Float64
end

left_on_ground(mode::Int)  = mode != SS_RIGHT
right_on_ground(mode::Int) = mode != SS_LEFT

"""
    biped_slip_dynamics!(dz, z, p, t)

Continuous-time equations of motion of the point mass for a given hybrid
mode. Written in the in-place form expected by `OrdinaryDiffEq`.

`p = (mode, xf, params)` where:
- `mode`   : current discrete mode, one of `SS_LEFT`, `SS_RIGHT`,
             `DS_LEAD_LEFT`, `DS_LEAD_RIGHT`
- `xf`     : `FootPositions` giving the (fixed, horizontal) foot positions
             of the left/right feet currently in stance (the value for a
             leg not currently touching the ground is ignored)
- `params` : `BipedSLIPParams`

State `z = [x, y, xdot, ydot]`.
"""
function biped_slip_dynamics!(dz, z, p, t)
    mode, xf, params = p

    x, y, xd, yd = z

    Fx = 0.0
    Fy = 0.0

    if left_on_ground(mode)
        Fx_L, Fy_L, _, _ = leg_force_total(x, y, xf.L, params, LEFT)
        Fx += Fx_L
        Fy += Fy_L
    end

    if right_on_ground(mode)
        Fx_R, Fy_R, _, _ = leg_force_total(x, y, xf.R, params, RIGHT)
        Fx += Fx_R
        Fy += Fy_R
    end

    dz[1] = xd
    dz[2] = yd
    dz[3] = Fx / params.m
    dz[4] = Fy / params.m - params.g
    return nothing
end

"""
    transition_direction(mode)

Sign of the crossing that defines the "regular" hybrid transition event
(event 1) for the given mode: single-support modes transition on a
decreasing crossing (touchdown, `-1`); double-support modes transition on
an increasing crossing (liftoff, `+1`).
"""
function transition_direction(mode::Int)
    return (mode == SS_LEFT || mode == SS_RIGHT) ? -1 : +1
end

"""
    biped_slip_event_values!(out, z, mode, xf, params)

Hybrid transition and fall-detection event values, matching
`biped_slip_events.m`.

`out[1]` is always the "regular" hybrid transition (a touchdown in single
support, or a liftoff in double support). `out[2:5]` are fall/safety
detectors, expressed as "margins" that are positive while healthy and
cross zero (decreasing) at the point of failure:
- `out[2]`: CoM height margin above a minimum height
- `out[3]`: leg-length margin above a near-zero collapse length (checked
  over whichever leg(s) are on the ground)
- `out[4]`: leg-length margin below a generous overstretch length
  (numerical safety net)
- `out[5]`: leg-angle margin below a generous bound on |angle from
  vertical| (checked over whichever leg(s) are on the ground)

# Touchdown condition
The swing leg is idle (massless, no dynamics) until it reappears on the
ground. It is assumed to reappear at its own resting length `L0` and
landing angle `th_contact` (measured from vertical), i.e. its foot is a
point at distance `L0` from the CoM along the direction `th_contact`.
Purely from geometry, such a foot reaches ground level (height 0) exactly
when the CoM height falls to `y_td = L0*cos(th_contact)`, with the CoM
still descending (`dy/dt < 0`). This is the bipedal-SLIP touchdown rule of
Rummel et al. (2010), `y_TD = L0*sin(alpha0)` in their horizontal
-referenced angle `alpha0`, translated to our vertical-referenced angle.
Triggering on CoM height (rather than on the current stance leg's angle)
guarantees the new leg always touches down at exactly its resting length
-- i.e. with zero initial spring force -- so no spurious energy is added
to the conservative system at touchdown.
"""
function biped_slip_event_values!(out, z, mode::Int, xf::FootPositions, params::BipedSLIPParams)
    x, y, vx, vy = z

    min_height     = 0.15 * minimum(params.L0)
    max_leg_angle  = deg2rad(85)
    min_leg_length = 0.15 * minimum(params.L0)
    max_leg_length = 1.1 * maximum(params.L0)

    L_list  = Float64[]
    th_list = Float64[]

    if left_on_ground(mode)
        L_L, th_L = leg_kinematics(x, y, xf.L)
        push!(L_list, L_L)
        push!(th_list, th_L)
    end
    if right_on_ground(mode)
        L_R, th_R = leg_kinematics(x, y, xf.R)
        push!(L_list, L_R)
        push!(th_list, th_R)
    end

    val_transition = if mode == SS_LEFT
        # Active control of landing angle - take longer step if speed too high
        th_new = params.th_contact[RIGHT] + params.kappa * (vx - params.speed * 1.5)
        y_td = params.L0[RIGHT] * cos(th_new) + params.eta_contact * randn()
        y - y_td
    elseif mode == SS_RIGHT
        th_new = params.th_contact[LEFT] + params.kappa * (vx - params.speed * 1.5)
        y_td = params.L0[LEFT] * cos(th_new) + params.eta_contact * randn()
        y - y_td
    elseif mode == DS_LEAD_LEFT
        # right leg is trailing (last entry in L_list here); it lifts off
        # once its length exceeds its own resting length
        L_list[end] - params.L0[RIGHT]
    else # DS_LEAD_RIGHT
        # left leg is trailing (first entry in L_list here); it lifts off
        # once its length exceeds its own resting length
        L_list[1] - params.L0[LEFT]
    end

    out[1] = val_transition
    out[2] = y - min_height
    out[3] = minimum(L_list) - min_leg_length
    out[4] = max_leg_length - maximum(L_list)
    out[5] = max_leg_angle - maximum(abs, th_list)
    return out
end

"""
    FallReason

Which fall/safety condition (if any) terminated a stance segment.
"""
@enum FallReason NO_FALL FELL_HEIGHT FELL_COLLAPSE FELL_OVERSTRETCH FELL_ANGLE

const FALL_REASONS = (FELL_HEIGHT, FELL_COLLAPSE, FELL_OVERSTRETCH, FELL_ANGLE)

"""
    build_event_callback(mode, xf, params, event_idx)

Build a `VectorContinuousCallback` implementing `biped_slip_event_values!`
with per-event directions (mirroring the `direction` output of
`biped_slip_events.m`): event 1 uses `transition_direction(mode)`, events
2-5 always fire on a decreasing crossing.

`affect!` here is called as `affect!(integrator, event_idx)` where
`event_idx` is a length-5 vector whose nonzero entries mark which
event(s) crossed zero on this step, signed by crossing direction
(matching MATLAB's `direction` event field: `+1` for an increasing
crossing, `-1` for a decreasing one). Only the entry whose sign matches
the desired direction for that event is accepted as a genuine transition.

Terminates the integration and writes the index of the event that fired
into the mutable `Ref` `event_idx_out`.
"""
function build_event_callback(mode::Int, xf::FootPositions, params::BipedSLIPParams,
                               event_idx_out::Base.RefValue{Int})
    dir1 = transition_direction(mode)

    condition = function (out, z, t, integrator)
        biped_slip_event_values!(out, z, mode, xf, params)
    end

    affect! = function (integrator, ei)
        for idx in 1:5
            d = ei[idx]
            d == 0 && continue
            want_dir = idx == 1 ? dir1 : -1
            if d == want_dir
                event_idx_out[] = idx
                terminate!(integrator)
                return
            end
        end
    end

    return VectorContinuousCallback(condition, affect!, 5)
end

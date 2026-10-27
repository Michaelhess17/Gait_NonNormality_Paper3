"""
    compute_leg_history(Z, mode, xf, params)

Per-sample leg length/angle and foot position for both legs over a
trajectory segment simulated in a fixed hybrid mode.

# Arguments
- `Z`      : Nx4 state history `[x y xdot ydot]` for this segment
- `mode`   : the (constant) hybrid mode active during this segment
- `xf`     : `FootPositions`, the fixed foot positions active during this
             segment
- `params` : parameter struct (unused here but kept for API consistency)

# Returns
- `LL, thL` : Nx1 left leg length / angle-from-vertical; `NaN` when the
              left leg is not on the ground during this segment
- `LR, thR` : Nx1 right leg length / angle-from-vertical; `NaN` when the
              right leg is not on the ground during this segment
- `footL, footR` : Nx2 `[x y]` foot positions; `NaN` rows when not in
              contact
"""
function compute_leg_history(Z::AbstractMatrix, mode::Int, xf::FootPositions,
                              params::BipedSLIPParams)
    n = size(Z, 1)

    LL  = fill(NaN, n); thL = fill(NaN, n)
    LR  = fill(NaN, n); thR = fill(NaN, n)
    footL = fill(NaN, n, 2); footR = fill(NaN, n, 2)

    x = @view Z[:, 1]
    y = @view Z[:, 2]

    if left_on_ground(mode)
        for i in 1:n
            LL[i], thL[i] = leg_kinematics(x[i], y[i], xf.L)
        end
        footL[:, 1] .= xf.L
        footL[:, 2] .= 0.0
    end

    if right_on_ground(mode)
        for i in 1:n
            LR[i], thR[i] = leg_kinematics(x[i], y[i], xf.R)
        end
        footR[:, 1] .= xf.R
        footR[:, 2] .= 0.0
    end

    return LL, thL, LR, thR, footL, footR
end

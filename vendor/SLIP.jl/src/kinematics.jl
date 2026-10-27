"""
    leg_kinematics(x, y, xf) -> (L, th)

Leg length and angle for a leg with a fixed foot on flat ground.

# Arguments
- `x, y`: Cartesian position of the point mass
- `xf`  : horizontal foot position (foot height is assumed 0, i.e. flat
          ground)

# Returns
- `L`  : leg length (distance from foot to mass)
- `th` : leg angle measured from VERTICAL, positive when the foot is AHEAD
         of the mass in the direction of walking (+x). With this
         convention `th = +th_contact` at touchdown (foot planted ahead of
         the mass) and `th` decreases through zero (mass over the foot,
         mid-stance) to negative values as the mass passes over the
         planted foot (foot trailing behind the mass).
"""
function leg_kinematics(x, y, xf)
    dx = x - xf   # mass minus foot, horizontal
    dy = y        # foot height = 0

    L  = hypot(dx, dy)
    th = atan(-dx, dy)   # = atan2(xf - x, y)
    return L, th
end

"""
    leg_force(L, th, k, L0, kth, th0) -> (Fx, Fy)

Cartesian force on the point mass from one leg's axial spring and hip
(rotary) spring, given the leg is planted on the ground.

# Model
The leg is a massless, lossless linear spring of stiffness `k` and resting
length `L0` acting along the leg. In addition, a torsional (hip) spring of
stiffness `kth` and neutral angle `th0` acts about the planted foot,
resisting rotation of the leg away from `th0`.

Treating `(L, th)` as polar coordinates centred on the (fixed) foot, the
generalized forces conjugate to `L` and `th` are
```
Q_L  =  k*(L0 - L)          # axial spring force
Q_th =  kth*(th0 - th)      # hip spring torque
```
which map to Cartesian force through the radial/tangential unit vectors
```
e_r  = [-sin(th),  cos(th)]     # points from foot to mass
e_th = [-cos(th), -sin(th)]     # direction of increasing th
```
giving `F = Q_L*e_r + (Q_th/L)*e_th`. This is the standard
point-mass-in-polar-coordinates result (see e.g. the axial-torsional SLIP
model of Shahbazi et al. 2016) adapted to a leg angle measured from
vertical rather than horizontal.

Gravity is NOT included here; add it separately.
"""
function leg_force(L, th, k, L0, kth, th0)
    F_axial = k * (L0 - L)

    # Leg force cannot be < 0 (leg is pulling)
    F_axial = max(F_axial, 0.0)

    F_tang = L > 1e-9 ? kth * (th0 - th) / L : 0.0

    sin_th, cos_th = sincos(th)
    e_r  = (-sin_th,  cos_th)
    e_th = (-cos_th, -sin_th)

    Fx = F_axial * e_r[1] + F_tang * e_th[1]
    Fy = F_axial * e_r[2] + F_tang * e_th[2]
    return Fx, Fy
end

"""
    leg_force_total(x, y, xf, params, idx) -> (Fx, Fy, L, th)

Cartesian force on the mass from leg number `idx` (`LEFT`/`RIGHT`), given
the mass position and that leg's (fixed) foot position. `params` must
contain the length-2 vectors `k`, `L0`, `kth`, `th0` (see
[`biped_slip_default_params`](@ref)).
"""
function leg_force_total(x, y, xf, params::BipedSLIPParams, idx::Int)
    L, th = leg_kinematics(x, y, xf)
    Fx, Fy = leg_force(L, th, params.k[idx], params.L0[idx],
                        params.kth[idx], params.th0[idx])
    return Fx, Fy, L, th
end

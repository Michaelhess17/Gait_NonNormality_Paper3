"""
    BipedSLIPParams

Parameters for the bipedal SLIP-with-hip-spring walking model.

All leg-related quantities are stored as 2-element vectors, indexed as
`[LEFT]`/`[RIGHT]` (see [`LEFT`](@ref)/[`RIGHT`](@ref)), so the two legs
may be given different properties if desired.

Fields:
- `m`           mass [kg]
- `g`           gravitational acceleration [m/s^2]
- `L0`          leg resting length [m]                    (length 2)
- `k`           axial leg-spring stiffness [N/m]           (length 2)
- `th0`         hip-spring resting angle [rad]              (length 2)
- `kth`         hip (rotary) spring stiffness [N*m/rad]     (length 2)
- `th_contact`  landing (angle-of-attack) angle [rad]       (length 2)
- `eta_contact` noise level randomly shifting th_contact each stride
- `kappa`       feedback gain on forward velocity error
- `speed`       target/nominal forward speed [m/s]
"""
Base.@kwdef mutable struct BipedSLIPParams
    m::Float64
    g::Float64
    L0::Vector{Float64}
    k::Vector{Float64}
    th0::Vector{Float64}
    kth::Vector{Float64}
    th_contact::Vector{Float64}
    speed::Float64
    eta_contact::Float64
    kappa::Float64
end

"""
    biped_slip_default_params()

Literature-informed default parameters for the bipedal SLIP-with-hip-spring
walking model.

Parameter values are based on the human-scale bipedal spring-mass model
used in:
- Rummel, J. et al. (2010) "Stable and Robust Walking with Compliant Legs",
  ICRA. (m = 80 kg, L0 = 1 m, dimensionless leg stiffness ~ k*L0/(m*g) = 20,
  angle of attack in the range that gives the stable walking patterns of
  their Fig. 3).
- Geyer, H. et al. (2006) "Compliant leg behaviour explains basic dynamics
  of walking and running", Proc. R. Soc. B.
"""
function biped_slip_default_params()
    m  = 80.0                     # kg
    g  = 9.81                     # m/s^2
    L0 = [1.0, 1.0]                # m, leg resting length (left, right)

    # Forward velocity (m/s)
    speed = 1.10

    # Dimensionless leg stiffness ~20 (Rummel 2010) -> k = ktilde*m*g/L0
    ktilde = 20.0
    k = ktilde .* m .* g ./ L0    # N/m (left, right)

    # Angle of attack ~20 deg from vertical (~70 deg from horizontal),
    # within the stable walking range reported by Rummel (2010, Fig. 3).
    th_contact = deg2rad.([20.0, 20.0])   # rad (left, right)

    # Contact noise (rad)
    eta_contact = 0.0

    # Hip (rotary) spring. th0 = 0 means the spring's neutral angle is
    # the vertical leg orientation. A modest stiffness is used by default
    # so both the axial and torsional terms are active; set kth = [0, 0]
    # to recover the classical (Rummel/Geyer) point-mass SLIP walker.
    th0  = [0.0, 0.0]              # rad
    kth  = [15.0, 15.0]            # N*m/rad

    # Gain for feedback control (default = 0)
    kappa = 0.0

    return BipedSLIPParams(; m, g, L0, k, th0, kth, th_contact, speed,
                            eta_contact, kappa)
end

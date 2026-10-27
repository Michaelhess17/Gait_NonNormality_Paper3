module SLIP

using OrdinaryDiffEq
using DiffEqBase: terminate!
using Statistics: median

export LEFT, RIGHT, SS_LEFT, SS_RIGHT, DS_LEAD_LEFT, DS_LEAD_RIGHT
export BipedSLIPParams, biped_slip_default_params
export FootPositions
export leg_kinematics, leg_force, leg_force_total
export biped_slip_dynamics!
export biped_slip_event_values!, build_event_callback, transition_direction,
       FallReason, NO_FALL, FELL_HEIGHT, FELL_COLLAPSE, FELL_OVERSTRETCH, FELL_ANGLE
export BipedSLIPIC, biped_slip_vlo_ic, biped_slip_energy_ic
export compute_leg_history
export SimOptions, BipedSLIPResults, simulate_biped_slip
export poincare_return, finite_diff_jacobian, find_periodic_gait,
       floquet_multipliers, GaitStability, gait_stability
export poincare_step, find_periodic_gait_step, floquet_multipliers_step,
       gait_stability_step
export VLOEvent, extract_vlo_events

include("constants.jl")
include("params.jl")
include("kinematics.jl")
include("dynamics.jl")
include("events.jl")
include("ic.jl")
include("leg_history.jl")
include("simulate.jl")
include("stability.jl")
include("paper_stability.jl")

end # module SLIP

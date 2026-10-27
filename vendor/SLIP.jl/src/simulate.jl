"""
    SimOptions

Options controlling [`simulate_biped_slip`](@ref).

- `t_max`     : maximum simulation time [s] (default 20)
- `max_steps` : maximum number of footsteps/transitions to simulate before
                stopping (default 40)
- `dt`        : output sampling interval [s] (default 0.002)
- `reltol`    : ODE solver relative tolerance (default 1e-6)
- `abstol`    : ODE solver absolute tolerance (default 1e-9)
- `dtmax`     : ODE solver maximum step size [s] (default 0.001)
"""
Base.@kwdef struct SimOptions
    t_max::Float64     = 20.0
    max_steps::Int      = 40
    dt::Float64         = 0.002
    reltol::Float64     = 1e-6
    abstol::Float64     = 1e-9
    dtmax::Float64      = 0.001
end

"""
    BipedSLIPResults

Output of [`simulate_biped_slip`](@ref).

- `T`        : Nx1 time vector
- `X`        : Nx4 CoM state history `[x y xdot ydot]`
- `mode`     : Nx1 discrete hybrid mode at each time sample
- `LL, thL`  : Nx1 left leg length / angle-from-vertical (`NaN` when the
               left leg is not on the ground)
- `LR, thR`  : Nx1 right leg length / angle-from-vertical (`NaN` when the
               right leg is not on the ground)
- `footL, footR` : Nx2 `[x y]` foot positions (`NaN` when not in contact)
- `fallen`   : true if the simulation was terminated by a fall condition
- `n_steps`  : number of completed footstep transitions
- `params`   : the parameter struct used (for convenience, e.g. in
               animation)
"""
struct BipedSLIPResults
    T::Vector{Float64}
    X::Matrix{Float64}
    mode::Vector{Int}
    LL::Vector{Float64}
    thL::Vector{Float64}
    LR::Vector{Float64}
    thR::Vector{Float64}
    footL::Matrix{Float64}
    footR::Matrix{Float64}
    fallen::Bool
    n_steps::Int
    params::BipedSLIPParams
end

"""
    simulate_biped_slip(params, ic, opts=SimOptions()) -> BipedSLIPResults

Simulate a bipedal SLIP walker with axial leg springs and rotary hip
springs over multiple gait cycles.

`params` is a [`BipedSLIPParams`](@ref) (see
[`biped_slip_default_params`](@ref)), `ic` is a [`BipedSLIPIC`](@ref)
(see [`biped_slip_vlo_ic`](@ref)/[`biped_slip_energy_ic`](@ref)), and
`opts` is a [`SimOptions`](@ref).

Each stance segment is integrated with `OrdinaryDiffEq.Tsit5` (a
Runge-Kutta 4(5) method, playing the role of MATLAB's `ode45`) until a
terminal event fires: either the "regular" hybrid transition (touchdown in
single support, or liftoff in double support), which advances the
discrete mode and continues, or one of the fall/safety conditions, which
stops the simulation with `fallen = true`.
"""
function simulate_biped_slip(params::BipedSLIPParams, ic::BipedSLIPIC,
                              opts::SimOptions=SimOptions())
    mode = ic.mode
    z    = copy(ic.state)
    xfL  = ic.xfL
    xfR  = ic.xfR
    t0   = 0.0

    T_all     = Float64[]
    X_all     = Matrix{Float64}(undef, 0, 4)
    mode_all  = Int[]
    LL_all    = Float64[]; thL_all = Float64[]
    LR_all    = Float64[]; thR_all = Float64[]
    footL_all = Matrix{Float64}(undef, 0, 2)
    footR_all = Matrix{Float64}(undef, 0, 2)

    fallen  = false
    n_steps = 0

    while (t0 < opts.t_max) && (n_steps < opts.max_steps) && !fallen
        tspan = (t0, opts.t_max)
        saveat = t0:opts.dt:opts.t_max
        saveat = length(saveat) < 2 ? [t0, opts.t_max] : collect(saveat)

        xf = FootPositions(xfL, xfR)
        p  = (mode, xf, params)

        event_idx = Ref(0)
        cb = build_event_callback(mode, xf, params, event_idx)

        prob = ODEProblem(biped_slip_dynamics!, z, tspan, p)
        sol = solve(prob, Tsit5(); callback=cb, reltol=opts.reltol,
                    abstol=opts.abstol, dtmax=opts.dtmax, saveat=saveat)

        Tseg = collect(sol.t)
        Zseg = permutedims(reduce(hcat, sol.u))

        LLseg, thLseg, LRseg, thRseg, footLseg, footRseg =
            compute_leg_history(Zseg, mode, xf, params)

        # avoid duplicating the shared time point between segments
        if !isempty(T_all)
            Tseg     = Tseg[2:end]
            Zseg     = Zseg[2:end, :]
            LLseg    = LLseg[2:end];    thLseg = thLseg[2:end]
            LRseg    = LRseg[2:end];    thRseg = thRseg[2:end]
            footLseg = footLseg[2:end, :]
            footRseg = footRseg[2:end, :]
        end

        T_all     = vcat(T_all, Tseg)
        X_all     = vcat(X_all, Zseg)
        mode_all  = vcat(mode_all, fill(mode, length(Tseg)))
        LL_all    = vcat(LL_all, LLseg);   thL_all = vcat(thL_all, thLseg)
        LR_all    = vcat(LR_all, LRseg);   thR_all = vcat(thR_all, thRseg)
        footL_all = vcat(footL_all, footLseg)
        footR_all = vcat(footR_all, footRseg)

        if isempty(Tseg)
            break
        end

        t0 = Tseg[end]
        z  = Zseg[end, :]

        if event_idx[] == 0
            # reached t_max within this segment without any event: done
            break
        end

        ie = event_idx[]

        if ie != 1
            # any event other than the "regular" transition event (#1) is
            # one of the fall-detection conditions
            fallen = true
            continue
        end

        # ie == 1: advance the hybrid mode
        if mode == SS_LEFT
            # right leg touches down ahead of the CoM
            xfR  = z[1] + params.L0[RIGHT] * sin(params.th_contact[RIGHT])
            mode = DS_LEAD_RIGHT
        elseif mode == SS_RIGHT
            # left leg touches down ahead of the CoM
            xfL  = z[1] + params.L0[LEFT] * sin(params.th_contact[LEFT])
            mode = DS_LEAD_LEFT
        elseif mode == DS_LEAD_LEFT
            # trailing right leg lifts off
            xfR  = NaN
            mode = SS_LEFT
        elseif mode == DS_LEAD_RIGHT
            # trailing left leg lifts off
            xfL  = NaN
            mode = SS_RIGHT
        end

        n_steps += 1
    end

    return BipedSLIPResults(T_all, X_all, mode_all, LL_all, thL_all,
                             LR_all, thR_all, footL_all, footR_all,
                             fallen, n_steps, params)
end

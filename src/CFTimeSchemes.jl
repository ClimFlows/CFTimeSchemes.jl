module CFTimeSchemes

# Void type and instance to merge mutating and non-mutating implementations
using MutatingOrNot: void, Void

#============ ClimFlows time integration API ===============#

"""
    # This variant is called by explicit time schemes.
    dstate, scratch = tendencies!(dstate, scratch, model, state, time) # Mutating
    dstate, scratch = tendencies!(void, void, model, state, time)      # Non-mutating
Return tendencies `dstate` and scratch space `scratch` for a certain `model`, `state` and `time`.
Pass `void` as output arguments for non-mutating variant.

    # This variant is called by diagonally-implicit time schemes.
    dstate, scratch = tendencies!(dstate, scratch, model, state, time, tau) # Mutating
    dstate, scratch = tendencies!(void, void, model, state, time, tau)      # Non-mutating

Perform a backward Euler time step from time `time` to `time+tau` then return tendencies `dstate` evaluated at `time+tau`
and scratch space `scratch` for a certain `model` and `state`. `tau` must be non-negative and may be zero.
Pass `void` as output arguments for non-mutating variant.

    # This variant is called by IMEX time schemes.
    dstate_exp, dstate_imp, scratch = tendencies!(dstate_exp, dstate_imp, scratch, model, state, time, tau) # Mutating
    dstate_exp, dstate_imp, scratch = tendencies!(void, void, void, model, state, time, tau)                # Non-mutating

Perform a backward Euler time step of length tau for implicit tendencies and 
return explicit tendencies `dstate_exp` and implicit tendencies `dtstate_imp`,
all evaluated at time `time+tau`. Also return scratch space `scratch`.
Pass `void` as output arguments for non-mutating variant.

This function is not implemented. It is meant to be implemented by the user for user-defined type `Model`.
"""
function tendencies! end

"""
    k = model_dstate(model, state, time)
Returns an object `k` in which we can store tendencies. Argument `state` is the model state.
Its type may contain information needed to allocate `k`, e.g. arrays may be of eltype `ForwardDiff.Dual`.

The fallback implementation looks like:
    k, _ = tendencies(void, void, model, state, time)
which incurs the needless computation of tendencies.
If this behavior is undesirable, one may pass a special value for `time` such as `nothing`
and implement a specialized version of `tendencies!` which only allocates and skips computations.
"""
model_dstate(model, state, time) = tendencies!(void, void, model, state, time)[1]

"""
    scratch = scratch_space(model, state, time)
    scratch = scratch_space(scheme, state, time)
Return scratch space `scratch`, to be used later to compute tendencies for `model`
or to hold sub-stages of Runge-Kutta scheme `scheme`.

Returns an object `k` in which we can store tendencies. Argument `state` is the model state.
Its type may contain information needed to allocate `k`, e.g. arrays may be of eltype `ForwardDiff.Dual`.

For a model, the fallback implementation looks like:
    _, scratch = tendencies(void, void, model, state, time)
which incurs the needless computation of tendencies.
If this behavior is undesirable, one may pass a special value for `time` such as `nothing`
and implement a specialized version of `tendencies!` which only allocates and skips computations.
"""
scratch_space(model, state, time) = tendencies!(void, void, model, state, time)[2]

# Time scheme API
"""
    future, t = advance!(future, scratch, scheme, present, t, dt) # mutating
    future, t = advance!(void, void, scheme, present, t, dt)      # non-mutating
Integrate in time by one time step, respectively mutating (non-allocating) and non-mutating
"""
function advance! end

"""
    dt = max_time_step(scheme, dt_lim)
Return the max time step `dt` for time-stepping `scheme`,
assuming that stability is limited by imaginary eigenvalues.
`dt_lim` is the smallest time scale (inverse of largest pulsation for a linear system) in the model.
"""
function max_time_step end

"""
    new = update!(new, model, new, factor1, increment1, [factor2, increment2, ...])
    new = update!(new, model, old, factor1, increment1, [factor2, increment2, ...])
    new = update!(void, model, old, factor1, increment1, [factor2, increment2, ...])
Return `new`` model state, obtained by updating the `old` state by `factor1*increment1`.
Extra arguments `factor2, increment2, ...` can be appended.
Allocates `new` if `new::Void`. Acts in-place if `new==old`.

The provided implementation ignores argument `model` and calls
    `Update.update!(new, nothing, old, factor, increment)`
which operates recursively on nested tuples / named tuples and looks like
    new = @. new = old + factor*increment
for arrays.

If a different behavior is desired, one may specialize `update!` for a specific type of `model!`.

See also `Update.manage`
"""
@inline update!(new, _, old, args...) = Update.update!(new, nothing, old, args...)

include("julia/update.jl")
include("julia/explicit.jl")
include("julia/implicit.jl")
include("julia/imex.jl")

# Initial-value problem solver
"""
    solver = IVPSolver(scheme, dt)                  # non-mutating
    solver = IVPSolver(scheme, dt, state0, time0)   # mutating
Return initial-value problem solver `solver`, to be used with `advance!`.

The first syntax returns a non-mutating solver, known to work with Zygote but which allocates.
When `state0` and `time0` are provided, scratch spaces are allocated and a mutating solver is returned.
`advance` should then not allocate and have better performance due to memory reuse.

`state0` and `time0` are used only to allocate scratch spaces. A special value of `time0`
such as `nothing` may be used to avoid the needless computation of tendencies, see `scratch_space` and `model_dstate`.
"""
struct IVPSolver{F,Scheme,Scratch}
    dt::F # time step
    scheme::Scheme
    scratch::Scratch # scratch space, or void
end

IVPSolver(scheme, dt) = IVPSolver(dt, scheme, void)
IVPSolver(scheme, dt, state0, time0) = IVPSolver(dt, scheme, scratch_space(scheme, state0, time0))

"""
    future, t = advance!(future, solver, present, t, N)
    future, t = advance!(void, solver, present, t, N)
"""
function advance!(storage::Union{Void, State}, (; dt, scheme, scratch)::IVPSolver, state::State, t, N::Int) where State
    @assert N>0
    @assert typeof(t)==typeof(dt)
    state = advance!(storage, scheme, state, t, dt, scratch)::State
    for i=2:N
        state = advance!(storage, scheme, state, t+(i-1)*dt, dt, scratch)::State
    end
    return state, t+N*dt
end

#========== for Julia <1.9 ==========#

using PackageExtensionCompat
function __init__()
    @require_extensions
end

end

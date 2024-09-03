module CFTimeSchemes

# Void type and instance to merge mutating and non-mutating implementations
using MutatingOrNot: void, Void

#============ ClimFlows time integration API ===============#

# Model API

"""
    dstate, scratch = tendencies!(dstate, scratch, model::Model, state, time) # Mutating
    dstate, scratch = tendencies!(void, void, model::Model, state, time)      # Non-mutating
Return tendencies `dstate` and scratch space `scratch` for a certain `model`, `state` and `time`.
Pass `void` to arguments `dstate` and `scratch` for non-mutating variant.

This function is not implemented. It is meant to be implemented by the user for user-defined type `Model`.
"""
function tendencies! end

"""
    k = model_dstate(model, state, time)
Returns an object `k` in which we can store tendencies. Argument `state` is the model state. 
Its type may contain information needed to allocate `k`, e.g. arrays may be of eltype `ForwardDiff.Dual`.

The implementation looks like:
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

For a model, the implementation looks like:
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

"""
    scheme = RungeKutta4(model)
4th-order Runge Kutta scheme for `model`. Pass `scheme` to `IVPSolver`.
"""
struct RungeKutta4{Model}
    model::Model
end

function scratch_space((; model)::RungeKutta4, u0, t0)
    k() = model_dstate(model, u0, t0)
    return (scratch = scratch_space(model, u0, t0), k0 = k(), k1 = k(), k2 = k(), k3 = k())
end

function advance!(future, (; model)::RungeKutta4, u0, t0, dt, (; scratch, k0, k1, k2, k3))
    # we ignore the second value (scratch) returned by tendencies!
    k0, _ = tendencies!(k0, scratch, model, u0, t0)
    # u1 = u0 + dt/2*k0
    u1 = update!(future, model, u0, dt / 2, k0)
    k1, _ = tendencies!(k1, scratch, model, u1, t0 + dt / 2)
    # u2 = u1 - dt/2*k0 + dt/2*k1 == u0 + dt/2*k1
    u2 = update!(future, model, u1, -dt / 2, k0, dt / 2, k1)
    k2, _ = tendencies!(k2, scratch, model, u2, t0 + dt / 2)
    # u3 = u2 - dt/2*k1 + dt*k2 == u0 + dt*k2
    u3 = update!(future, model, u2, -dt / 2, k1, dt, k2)
    k3, _ = tendencies!(k3, scratch, model, u3, t0 + dt)
    # u4 = u0 + (k0+k3)*(dt/6) + (k1+k2)*(dt/3)
    future = update!(future, model, u3, dt / 6, k0, dt / 3, k1, -2dt / 3, k2, dt / 6, k3)
    return future
end

#========== for Julia <1.9 ==========#

using PackageExtensionCompat
function __init__()
    @require_extensions
end

end

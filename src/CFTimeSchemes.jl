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
    dt = max_time_step(scheme, dt_lim)
Return the max time step `dt` for time-stepping `scheme`,
assuming that stability is limited by imaginary eigenvalues.
`dt_lim` is the smallest time scale (inverse of largest pulsation for a linear system) in the model.
"""

"""
    scheme = RungeKutta4(model)
4th-order Runge Kutta scheme for `model`. Pass `scheme` to `IVPSolver`.
"""
struct RungeKutta4{Model}
    model::Model
end
max_time_step(::RungeKutta4, dt::F) where F = F(dt*sqrt(8))

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

"""
    scheme = KinnmarkGray{2, 5}(model)
second-order, 5-stage Runge Kutta scheme for `model`. Pass `scheme` to `IVPSolver`. This scheme
has a maximum Courant number of 4 for imaginary eigenvalues, which is the best that can be
achieved with a 5-stage RK scheme
(Kinnmark & Gray, 1984a, https://doi.org/10.1016/0378-4754(84)90039-9).

scheme = KinnmarkGray{3, 5}(model)
third-order, 5-stage Runge Kutta scheme for `model`. Pass `scheme` to `IVPSolver`. This scheme
has a maximum Courant number of √15 for imaginary eigenvalues.
(Kinnmark & Gray, 1984b, https://doi.org/10.1016/0378-4754(84)90056-9).

See also Guba et al. 2020 https://doi.org/10.5194/gmd-13-6467-2020 .
"""
struct KinnmarkGray{order, stages, Model}
    model::Model
    KinnmarkGray{order,stages}(model) where {order, stages} =
        new{order, stages,typeof(model)}(model)
end
max_time_step(::KinnmarkGray{2,5}, dt::F) where F = 4dt
max_time_step(::KinnmarkGray{3,5}, dt::F) where F = F(dt*sqrt(15))

function scratch_space((; model)::KinnmarkGray{order, 5}, u0, t0) where order
    k() = model_dstate(model, u0, t0)
    k0, k1 = k(), k()
    return (scratch = scratch_space(model, u0, t0), k=(k0, k1, k0, k1, k0))
end

advance!(future, scheme::KinnmarkGray{2,5}, u0, t0, dt, space) =
    LSRK5(future, scheme, u0, t0, dt, space, (dt/4, dt/6, 3dt/8, dt/2))

advance!(future, scheme::KinnmarkGray{3,5}, u0, t0, dt, space) =
    LSRK5(future, scheme, u0, t0, dt, space, (dt/5, dt/5, dt/3, dt/2))

function LSRK5(future, (; model), u0, t0, dt, (; scratch, k), butcher)
    (k0, k1, k2, k3, k4) = k
    a, b, c, d = butcher
    # we ignore the second value (scratch) returned by tendencies!
    k0, _ = tendencies!(k0, scratch, model, u0, t0)
    # u1 = u0 + a*k0
    u1 = update!(future, model, u0, a, k0)
    k1, _ = tendencies!(k1, scratch, model, u1, t0 + a)
    # u2 = u1 - a*k0 + b*k1 == u0 + b*k1
    u2 = update!(future, model, u1, -a, k0, b, k1)
    k2, _ = tendencies!(k2, scratch, model, u2, t0 + b)
    # u3 = u2 - b*k1 + c*k2 == u0 + c*k2
    u3 = update!(future, model, u2, -b, k1, c, k2)
    k3, _ = tendencies!(k3, scratch, model, u3, t0 + c)
    # u4 = u0 -c*k2 + d*k3 = u0 + d*k3
    u4 = update!(future, model, u3, -c, k2, d, k3)
    k4, _ = tendencies!(k4, scratch, model, u4, t0 + d)
    # u4 = u0 -d*k3 + k4 = u0 + k4
    future = update!(future, model, u4, -d, k3, dt, k4)
    return future
end

"""
    scheme = BackwardEuler(model)
First-order backward Euler scheme for `model`. 
Pass `scheme` to `IVPSolver`. 
"""
struct BackwardEuler{Model}
    model::Model
end
function scratch_space((; model)::BackwardEuler, u0, t0)
    k = model_dstate(model, u0, t0)
    return (scratch = scratch_space(model, u0, t0), k)    
end
function advance!(future, (; model)::BackwardEuler, u0, t0, dt, space)
    (; scratch, k) = space
    k, _ = tendencies!(k, scratch, model, u0, t0, dt)
    future = update!(future, model, u0, dt, k)
end

"""
    scheme = Midpoint(model)
Midpoint rule, second-order: forward Euler scheme for 1/2 time step
followed by backward Euler scheme for 1/2 time step.
Pass `scheme` to `IVPSolver`. 
"""
struct Midpoint{Model}
    model::Model
end
function scratch_space((; model)::Midpoint, u0, t0)
    k = model_dstate(model, u0, t0)
    return (scratch = scratch_space(model, u0, t0), k0=k, k1=k)    
end
function advance!(future, (; model)::Midpoint, u0, t0, dt, space)
    (; scratch, k0, k1) = space
    k0, _ = tendencies!(k0, scratch, model, u0, t0, zero(dt))
    u1 = update!(future, model, u0, dt/2, k0)
    k1, _ = tendencies!(k1, scratch, model, u1, t0+dt/2, dt/2)
    future = update!(future, model, u1, dt/2, k1)
end

"""
    scheme = TRBDF2(model)
Three-stage, second-order, L-stable scheme with two implicit stages.
Pass `scheme` to `IVPSolver`. 
"""
struct TRBDF2{Model}
    model::Model
end
function scratch_space((; model)::TRBDF2, u0, t0)
    k() = model_dstate(model, u0, t0)
    return (scratch = scratch_space(model, u0, t0), k=(k(), k(), k()))
end
function advance!(future, (; model)::TRBDF2, u0, t0::F, dt::F, space) where F
    (; scratch, k) = space
    (k0, k1, k2) = k
    beta = F(1/sqrt(8))
    alpha = 1-2*beta
    k0, _ = tendencies!(k0, scratch, model, u0, t0, zero(dt))
    u1 = update!(future, model, u0, alpha*dt, k0)
    # u1 = u0 + alpha*dt*k0
    k1, _ = tendencies!(k1, scratch, model, u1, t0+alpha*dt, alpha*dt)
    u2 = update!(future, model, u1, (beta-alpha)*dt, k0, beta*dt, k1)
    # u2 = u1 + (beta-alpha)*dt*k0 + beta*dt*k1
    #    = u0 + beta*dt*k0 + beta*dt*k1
    k2, _ = tendencies!(k2, scratch, model, u2, t0+2beta*dt, alpha*dt)
    future = update!(future, model, u2, alpha*dt, k2)
    # u3 = u2 + alpha*dt*k2
    #    = u0 + beta*dt*k0 + beta*dt*k1 + alpha*k2
end

#========== for Julia <1.9 ==========#

using PackageExtensionCompat
function __init__()
    @require_extensions
end

end

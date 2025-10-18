abstract type IMEXScheme end

"""
    scheme = ForwardBackwardEuler(model)
First-order IMEX scheme for `model`: backward Euler for implicit terms 
followed by forward Euler for explicit terms.
Pass `scheme` to `IVPSolver`. 
"""
struct ForwardBackwardEuler{Model} <: IMEXScheme
    model::Model
end
max_time_step(::ForwardBackwardEuler, tau) = tau

function scratch_space((; model)::ForwardBackwardEuler, u0, t0)
    k() = model_dstate(model, u0, t0)
    return (scratch = scratch_space(model, u0, t0), k=k(), l=k())    
end
function advance!(future, (; model)::ForwardBackwardEuler, u0, t0, dt, space)
    (; scratch, k, l) = space
    k, l, _ = tendencies!(k, l, scratch, model, u0, t0, dt)
    future = update!(future, model, u0, dt, k, dt, l)
end

"""
    scheme = ARK_TRBDF2(model)
Second-order IMEX scheme from Giraldo et al. 2013. Implicit terms of `model` are integrated
with a TRBDF2 scheme and explicit terms with a 3-stage second-order RK scheme.
Pass `scheme` to `IVPSolver`. 
"""
struct ARK_TRBDF2{Model} <: IMEXScheme
    model::Model
end
max_time_step(::ARK_TRBDF2, tau) = tau

function scratch_space((; model)::ARK_TRBDF2, u0, t0::Number)
    k0, l0, scratch = tendencies!(void, void, void, model, u0, t0, zero(t0)) 
    k1, l1, _ = tendencies!(void, void, scratch, model, u0, t0, zero(t0)) 
    k2, l2, _ = tendencies!(void, void, scratch, model, u0, t0, zero(t0)) 
    return (; scratch, k0, k1, k2, l0, l1, l2)
end

function advance!(future, (; model)::ARK_TRBDF2, u0, t0::F, dt::F, space) where F
    (; scratch, k0, k1, k2, l0, l1, l2) = space
    beta = dt/sqrt(F(8))
    alpha, a32 = dt-2beta, (3dt + 8beta)/6

    k0, l0, _ = tendencies!(k0, l0, scratch, model, u0, t0, zero(dt))
    u1 = update!(future, model, u0, 
                2alpha, k0,
                alpha, l0)

    k1, l1, _ = tendencies!(k1, l1, scratch, model, u1, t0+2alpha, alpha)
    u2 = update!(future, model, u1, 
                dt-a32-2alpha, k0, 
                a32, k1,
                beta-alpha, l0, 
                beta, l1)

    k2, l2, _ = tendencies!(k2, l2, scratch, model, u2, t0+dt, alpha)
    future = update!(future, model, u2, 
                beta-(dt-a32), k0, 
                beta-a32, k1,
                alpha, k2,
                alpha, l2)
end

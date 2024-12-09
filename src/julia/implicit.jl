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

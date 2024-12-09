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

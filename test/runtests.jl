using CFTimeSchemes: RungeKutta4, KinnmarkGray, BackwardEuler, 
        Midpoint, TRBDF2, ForwardBackwardEuler, ARK_TRBDF2
using CFTimeSchemes: void, max_time_step, IVPSolver, advance!
import CFTimeSchemes: tendencies!, scratch_space, model_dstate

using ForwardDiff: ForwardDiff
using Zygote: Zygote
using Enzyme: Enzyme

using LinearAlgebra: dot
using UnicodePlots
using Test

import Zygote.ChainRulesCore: rrule

# custom rule for muladd, currently flawed in Zygote
# @inline rrule(::typeof(muladd), a,b,c) = muladd(a,b,c), (x)->(b*x, a*x, x)

#========= simple model dx/dt = x*z =========#

struct Model{A}
    z::A
end
tendencies!(dstate, _, m::Model, state, _) = (@. dstate = state*m.z), nothing
tendencies!(dstate, _, m::Model, state, _, tau) = (@. dstate = state*m.z/(1-tau*m.z)), nothing

function tendencies!(k, l, _, m::Model, state, _, tau)
    # tendencies are 1/2 explicit, 1/2 implicit
    k = @. k = m.z*state / (2-m.z*tau)
    l = @. l = m.z*state / (2-m.z*tau)
    return k, l, nothing
end

function make_model(Scheme)
    x, y = range(-3, 0.5, 101), range(-4.5, 4.5, 201)
    model = Model([xx+yy*1im for xx in x, yy in y])
    z0 = one.(model.z) # complex 1
    scheme = Scheme(model)
    return model, scheme, z0
end

#============ same but with a named tuple and scratch space to challenge AD ===========#

struct Model2{A}
    z::A
end
function tendencies!(dstate, scratch, m::Model2, state, _) # fully explicit
    (; a, b) = state
    da, db, z = dstate.a, dstate.b, m.z
    scratch = @. scratch = z
    da = @. da = a * scratch
    db = @. db = b * scratch
    return (a=da, b=db), scratch
end

function tendencies!(dstate, scratch, m::Model2, state, _, tau)  # fully implicit
    (; a, b) = state
    da, db, z = dstate.a, dstate.b, m.z
    scratch = @. scratch = z/(1-tau*z)
    da = @. da = a * scratch
    db = @. db = b * scratch
    return (a=da, b=db), scratch
end

function tendencies!(k, l, scratch, m::Model2, state, _, tau) # IMEX
    # tendencies are 1/2 explicit, 1/2 implicit
    (; a, b) = state
    ka, kb, la, lb, z = k.a, k.b, l.a, l.b, m.z
    scratch = @. scratch = z/(2-tau*z)
    ka = @. ka = a * scratch
    kb = @. kb = b * scratch
    la = @. la = a * scratch
    lb = @. lb = b * scratch
    return (a=ka, b=kb), (a=la, b=lb), scratch
end

function make_model2(Scheme)
    x, y = range(-3, 0.5, 101), range(-4.5, 4.5, 201)
    model = Model2([xx+yy*1im for xx in x, yy in y])
    z0 = one.(model.z) # complex 1
    scheme = Scheme(model)
    return model, scheme, (a = z0, b=copy(z0))
end

#==========================================================================================#

accuracy(::BackwardEuler) = 1e-5 # first-order scheme
accuracy(::Union{KinnmarkGray{2,5}, Midpoint, TRBDF2, ARK_TRBDF2}) = 1e-9 # second-order scheme
accuracy(::KinnmarkGray{3,5}) = 1e-13 # third-order
accuracy(::RungeKutta4) = 1e-17 # fourth-order

function stability_region(Scheme)
    model, scheme, z0 = make_model(Scheme)
    solver = IVPSolver(scheme, 1.0)
    z1, t = advance!(void, solver, z0, 0.0, 100)
    max_dt = max_time_step(scheme, 1.0)

    min_err = let N=20
        solver = IVPSolver(scheme, 1/N)
        z2, t = advance!(void, solver, z0, 0.0, N)
        minimum(abs, @. z2*exp(-model.z)-1)
    end
    @info Scheme max_dt min_err
    display(heatmap(@. min(1.0, abs(z1))))
    @test min_err<accuracy(scheme)
end

function autodiff(Scheme)
    model, scheme, z0 = make_model2(Scheme)
    solver = IVPSolver(scheme, 1.0)

    # first, check that non-mutating variant supports ForwardDiff
    function loss(x)
        zx = (a=x.*z0.a, b=x.*z0.b)
        z1, t = advance!(void, solver, zx, 0.0, 10)
        sum(abs2, z1.a)
    end
    g_ssa = ForwardDiff.derivative(loss, 1.0)
    @test true

    # now check mutating variant
    function loss_mutating(x)
        zx = (a=x.*z0.a, b=x.*z0.b)
        z1 = map(similar, zx)
        solver! = IVPSolver(scheme, 1.0, zx, 0.0)
        advance!(z1, solver!, zx, 0.0, 10)
        sum(abs2, z1.a)
    end
    g_mut = ForwardDiff.derivative(loss_mutating, 1.0)

    # and Zygote
    function loss_zyg(z)
        z1, t = advance!(void, solver, z, 0.0, 10)
        sum(abs2, z1.a)
    end
    _, pb = Zygote.pullback(loss_zyg, z0)
    g_zyg = dotprod(z0, pb(1.0)[1])

    # and Enzyme
    Dup(x) = Enzyme.Duplicated(x, Enzyme.make_zero(x))
    g_enz = Base.VERSION<v"1.11" ? Enzyme.gradient(Enzyme.Reverse, Dup(loss_mutating), 1.0)[1] :  g_zyg # Julia 1.11 fails for the moment

    # check mutual consistency
    @info Scheme g_ssa g_mut g_zyg g_enz
    @test (g_ssa ≈ g_mut) & (g_ssa ≈ g_zyg) & (g_ssa ≈ g_enz)
end

dotprod(a,b) = dot(a,b)
dotprod(a::Array, ::Nothing) = zero(eltype(a))
dotprod(a::NamedTuple, b::NamedTuple) = mapreduce(dotprod, +, a, b)

Schemes = [RungeKutta4, KinnmarkGray{2,5}, KinnmarkGray{3,5}, 
            BackwardEuler, Midpoint, TRBDF2,
            ARK_TRBDF2]
@testset "Accuracy" begin
    foreach(stability_region, Schemes)
    println()
end

@testset "Auto-diff" begin
    foreach(autodiff, Schemes)
    println()
end

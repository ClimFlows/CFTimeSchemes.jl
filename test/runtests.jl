using CFTimeSchemes: RungeKutta4, KinnmarkGray, BackwardEuler, Midpoint, TRBDF2
using CFTimeSchemes: void, max_time_step, IVPSolver, advance!
import CFTimeSchemes: tendencies!, scratch_space, model_dstate

using ForwardDiff: ForwardDiff
using Zygote: Zygote
using LinearAlgebra: dot
using UnicodePlots
using Test

import Zygote.ChainRulesCore: rrule

# custom rule for muladd, currently flawed in Zygote
@inline rrule(::typeof(muladd), a,b,c) = muladd(a,b,c), (x)->(b*x, a*x, x)

struct Model{A}
    z::A
end
tendencies!(dstate, _, m::Model, state, _) = (@. dstate = state*m.z), nothing
tendencies!(dstate, _, m::Model, state, _, tau) = (@. dstate = state*m.z/(1-tau*m.z)), nothing

function make_model(Scheme)
    x, y = range(-3, 0.5, 101), range(-4.5, 4.5, 201)
    model = Model([xx+yy*1im for xx in x, yy in y])
    z0 = one.(model.z) # complex 1
    scheme = Scheme(model)
    return model, scheme, z0
end

function stability_region(Scheme)
    model, scheme, z0 = make_model(Scheme)
    solver = IVPSolver(scheme, 1.0)
    z1, t = advance!(void, solver, z0, 0.0, 100)
    max_dt = max_time_step(scheme, 1.0)
    @info Scheme max_dt
    display(heatmap(@. min(1.0, abs(z1))))
    @test true

    solver = IVPSolver(scheme, 0.01)
    z1, t = advance!(void, solver, z0, 0.0, 100)
    display(heatmap(@. log10(abs(z1*exp(-model.z)-1))))
end

function autodiff(Scheme)
    model, scheme, z0 = make_model(Scheme)
    solver = IVPSolver(scheme, 1.0)

    # first, check that non-mutating variant supports ForwardDiff
    function loss(x)
        zx = x*z0
        z1, t = advance!(void, solver, zx, 0.0, 10)
        sum(abs2, z1)
    end
    g_ssa = ForwardDiff.derivative(loss, 1.0)
    @test true

    # now check mutating variant
    function loss_mutating(x)
        zx = x*z0
        z1 = similar(zx)
        solver! = IVPSolver(scheme, 1.0, zx, nothing)
        advance!(z1, solver!, zx, 0.0, 10)
        sum(abs2, z1)
    end
    g_mut = ForwardDiff.derivative(loss_mutating, 1.0)

    # and Zygote
    function loss_zyg(z)
        z1, t = advance!(void, solver, z, 0.0, 10)
        sum(abs2, z1)
    end
    _, pb = Zygote.pullback(loss_zyg, z0)
    g_zyg = dot(z0, pb(1)[1])

    # check mutual consistency
    @info Scheme g_ssa g_mut g_zyg
    @test (g_ssa ≈ g_mut) & (g_ssa ≈ g_zyg)
end

Schemes = [RungeKutta4, KinnmarkGray{2,5}, KinnmarkGray{3,5}, 
            BackwardEuler, Midpoint, TRBDF2]

@testset "Stability" begin
    foreach(stability_region, Schemes)
    println()
end

@testset "Auto-diff" begin
    foreach(autodiff, Schemes)
    println()
end

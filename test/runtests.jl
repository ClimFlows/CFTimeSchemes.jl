using CFTimeSchemes: void, RungeKutta4, IVPSolver, advance!
import CFTimeSchemes: tendencies!, scratch_space, model_dstate

using ForwardDiff: ForwardDiff
using Zygote: Zygote
using LinearAlgebra: dot
using UnicodePlots
using Test

import Zygote.ChainRulesCore: rrule

# custome rule for muladd, currently flawed in Zygote
@inline rrule(::typeof(muladd), a,b,c) = muladd(a,b,c), (x)->(b*x, a*x, x)

struct Model{A}
    z::A
end
tendencies!(dstate, (;z)::Model, state, _, _) = @. dstate = state*z
scratch_space(::Model, state) = void
model_dstate(::Model, z) = similar(z)


function make_model(Scheme)
    x, y = range(-3, 1, 101), range(-4, 4, 201)
    model = Model([xx+yy*1im for xx in x, yy in y])
    z0 = one.(model.z) # complex 1
    scheme = Scheme(model)
    return model, scheme, z0
end

function stability_region(Scheme)
    model, scheme, z0 = make_model(Scheme)
    solver = IVPSolver(scheme, 1.0)
    z1, t = advance!(void, solver, z0, 0.0, 100)
    @info Scheme
    display(heatmap(@. min(1.0, abs(z1))))
    @test true
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
        solver! = IVPSolver(scheme, 1.0, zx, true)
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

@testset "Stability" begin
    stability_region(RungeKutta4)
    println()
end

@testset "Auto-diff" begin
    autodiff(RungeKutta4)
    println()
end

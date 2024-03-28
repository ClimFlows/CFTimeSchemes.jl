using CFTimeSchemes: void, RungeKutta4, IVPSolver, advance!
using UnicodePlots
import CFTimeSchemes: tendencies!
using Test

struct Model{A}
    z::A
end
tendencies!(dstate, (;z)::Model, state, _, _) = @. dstate = state*z

function stability_region(Scheme)
    x, y = range(-3, 1, 101), range(-4, 4, 201)
    model = Model([xx+yy*1im for xx in x, yy in y])
    z0 = one.(model.z) # complex 1
    scheme = Scheme(model)
    solver = IVPSolver(scheme, 1.0)
    z1, t = advance!(void, solver, z0, 0.0, 100)
    @info Scheme
    display(heatmap(@. min(1.0, abs(z1))))
    @test true
end

@testset "Stability" begin
    stability_region(RungeKutta4)
end

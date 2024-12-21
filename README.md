# CFTimeSchemes

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ClimFlows.github.io/CFTimeSchemes.jl/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ClimFlows.github.io/CFTimeSchemes.jl/dev/)
[![Build Status](https://github.com/ClimFlows/CFTimeSchemes.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ClimFlows/CFTimeSchemes.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ClimFlows/CFTimeSchemes.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ClimFlows/CFTimeSchemes.jl)

AD-friendly, fixed time-step RK and ARK schemes.

## Installation

`CFTimesSchemes` is registered in the ClimFlows registry. [Follow instructions there](https://github.com/ClimFlows/JuliaRegistry), then:
```julia
] add CFTimeSchemes
```

## Example

```julia
using CFTimeSchemes: void, RungeKutta4, IVPSolver, advance!
import CFTimeSchemes: tendencies!
using UnicodePlots

struct Model{A}
    z::A
end
tendencies!(dstate, _, m::Model, state, _) = (@. dstate = state*m.z), nothing

function make_model(Scheme)
    x, y = range(-3, 1, 101), range(-4, 4, 201)
    model = Model([xx+yy*1im for xx in x, yy in y])
    z0 = one.(model.z) # complex 1
    scheme = Scheme(model)
    return model, scheme, z0
end

function stability_region(Scheme)
    model, scheme, z0 = make_model(Scheme)
    # non-mutating solver
    solver = IVPSolver(scheme, 1.0)
    z1, t = advance!(void, solver, z0, 0.0, 100)
    @info Scheme
    display(heatmap(@. min(1.0, abs(z1))))

    # mutating solver
    solver = IVPSolver(scheme, 1.0, z0, nothing)
    z1, t = advance!(z1, solver, z0, 0.0, 100) # compile
    @time z1, t = advance!(z1, solver, z0, 0.0, 100) # should not allocate
    return
end

stability_region(RungeKutta4)

```
## Change Log

### Version 0.3

* new API:
    * `tendencies!` now has variants for implicit and
    implicit-explicit (IMEX) schemes
    * `tendencies!` now returns `scratch`. This enables default implementations for `model_dstate` and `scratch_space`.
    * `update!` now takes `model` as an argument and can be customized for user model types.


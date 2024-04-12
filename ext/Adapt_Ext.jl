module Adapt_Ext

using Adapt
using CFTimeSchemes: IVPSolver, RungeKutta4

Adapt.adapt_structure(to, solver::IVPSolver) =
    IVPSolver(solver.dt, adapt(to, solver.scheme), adapt(to, solver.scratch))

Adapt.adapt_structure(to, scheme::RungeKutta4) =
    RungeKutta4(adapt(to, scheme.model))

end

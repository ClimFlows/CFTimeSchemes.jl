var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = CFTimeSchemes","category":"page"},{"location":"#CFTimeSchemes","page":"Home","title":"CFTimeSchemes","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for CFTimeSchemes.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [CFTimeSchemes]","category":"page"},{"location":"#CFTimeSchemes.IVPSolver","page":"Home","title":"CFTimeSchemes.IVPSolver","text":"solver = IVPSolver(scheme, dt ; u0, mutating=false)\nsolver = IVPSolver(scheme, dt)\n\nUse solver with advance.\n\nBy default, solver is non-mutating, which is known to work with Zygote but allocates. If u0 is provided and mutating=true, advance should not allocate and have better performance. The type and shape of u0 (but not the values) are used to allocate scratch spaces, see scratch_space. This allows the non-mutating mode to work with ForwardDiff.\n\n\n\n\n\n","category":"type"},{"location":"#CFTimeSchemes.RungeKutta4","page":"Home","title":"CFTimeSchemes.RungeKutta4","text":"scheme = RungeKutta4(model)\n\n4th-order Runge Kutta scheme for model. Pass scheme to IVPSolver.\n\n\n\n\n\n","category":"type"},{"location":"#CFTimeSchemes.advance!","page":"Home","title":"CFTimeSchemes.advance!","text":"future, t = advance!(future, scheme, present, t, dt, scratch)\nfuture, t = advance!(void, scheme, present, t, dt, void)\n\nIntegrate in time by one time step, respectively mutating (non-allocating) and non-mutating\n\n\n\n\n\n","category":"function"},{"location":"#CFTimeSchemes.advance!-Union{Tuple{State}, Tuple{Union{MutatingOrNot.Void, State}, CFTimeSchemes.IVPSolver, State, Any, Int64}} where State","page":"Home","title":"CFTimeSchemes.advance!","text":"future, t = advance!(future, solver, present, t, N)\nfuture, t = advance!(void, solver, present, t, N)\n\n\n\n\n\n","category":"method"},{"location":"#CFTimeSchemes.model_dstate","page":"Home","title":"CFTimeSchemes.model_dstate","text":"k = model_dstate(model, u)\n\nReturns an object in which we can store tendencies ; u0 is passed to handle ForwardDiff.Dual\n\n\n\n\n\n","category":"function"},{"location":"#CFTimeSchemes.scratch_space","page":"Home","title":"CFTimeSchemes.scratch_space","text":"space = scratch_space(model, u0)\nspace = scratch_space(scheme, u0)\n\nscratch space used to compute tendencies or keep sub-stages (RK); u0 is passed to handle ForwardDiff.Dual\n\n\n\n\n\n","category":"function"},{"location":"#CFTimeSchemes.tendencies!","page":"Home","title":"CFTimeSchemes.tendencies!","text":"dstate = tendencies!(dstate, model, u, scratch, t) # Mutating\ndstate = tendencies!(void, model, u, void, t) # Non-mutatingœ\n\nComputes tendencies. Pass void to arguments dstate and scratch for non-mutating variant.\n\n\n\n\n\n","category":"function"},{"location":"#CFTimeSchemes.time_stages","page":"Home","title":"CFTimeSchemes.time_stages","text":"stages = time_stages(scheme, model, u0)\n\nreturns an object storing tendencies at sub-stages (RK). u0 is passed to handle ForwardDiff.Dual\n\n\n\n\n\n","category":"function"},{"location":"#CFTimeSchemes.update!-Union{Tuple{S}, Tuple{F}, Tuple{Any, S, F, S}} where {F, S}","page":"Home","title":"CFTimeSchemes.update!","text":"new = update!(new, new, increment, factor)\nnew = update!(new, old, increment, factor)\nnew = update!(void, old, increment, factor)\n\nRespectively equivalent to:     @. new += factorincrement     @. new = old + factorincrement     new = @. old + factor*increment Operates recursively on nested tuples / named tuples\n\n\n\n\n\n","category":"method"}]
}

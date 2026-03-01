abstract type MODEL{S} end
abstract type TIMESCHEME{M,S} end

struct RK3{M<:MODEL,S}<:TIMESCHEME{M,S}
    dq1::S
    dq2::S
    dq3::S
    model::M
end

struct RK4{M<:MODEL,S}<:TIMESCHEME{M,S}
    dq1::S
    dq2::S
    dq3::S
    dq4::S
    model::M
end

" RK3{M,S} is a struct of the SSP RK3 time stepping

  RK3 depends on the model (type `M<:MODEL`) and its state (type `S`)
  Once create, it can be used to perform the update the state.

Usage:

julia> scheme = RK3(model, state)

julia> advance!(state, dt, scheme)
"
function RK3(model::M, state::S) where {M<:MODEL,S}
    RK3{M,S}([similar(state) for _ in 1:3]..., model)
end

" RK4{M,S} is a struct of the classical RK4 time stepping

  its API is like RK3"
function RK4(model::M, state::S) where {M<:MODEL,S}
    RK4{M,S}([similar(state) for _ in 1:4]..., model)
end

" `advance!(state,h,scheme)`

update the model `state` by one time step `h` using the time `scheme`"
function advance!(state::S, h, scheme::T) where {M<:MODEL,S,T<:TIMESCHEME{M,S}} end


Base.show(io::IO, scheme::T) where {S,M<:MODEL{S},T<:TIMESCHEME{M,S}} = begin
    print(io, "Time scheme $(nameof(T)) of model $(nameof(M))")
end


" `rhs!(dq,q,model::M,stage) where {M<:MODEL}`

computes the right-hand side `dq` of `model` using the state `q`

`stage` indicates the call rank during the time iteration, `stage` = 1,2,...,0
the first stage is 1, the last stage is 0. This allows `rhs!` to be aware of the last call.

`rhs!` must be defined for each model `M`."
function rhs!(dq::S,q::S,model::M,stage) where {M<:MODEL,S} end

" `projection!(q,model::M,stage)`

correct the `model` state `q`, once the tendency has been added

This is where implicit terms, e.g. vertical mixing, or constraints, e.g. pressure projection, are computed.

For a model `M`, `projection!` can be ignored if no such computation is necessary.
"
function projection!(q::S,model::M,stage) where {M<:MODEL,S} end


# The general idea is to store the model state in the form of NamedTuple of arrays
STATE = NamedTuple{names, <:Tuple{Vararg{Array}}} where names
# from https://discourse.julialang.org/t/tuple-and-namedtuple-types/73701
# Question 2: for named tuples it’s trickier because they are not
# covariant, but since the types are encoded with a tuple (which is
# covariant) it’s still easy to do:

# This requires to extend Base.similar
Base.similar(state::NamedTuple) = begin
    names = typeof(state).parameters[1]
    NamedTuple{names}((similar(state[name]) for name in names))
end

# and to have a way add linear combinations of tendencies and coefficients to the model state
update!(q::S,(dq1,c1)::R) where {T,S<:STATE,R<:Tuple{S,T}} = begin
    for k in keys(q)
        @. q[k] += c1*dq1[k]
    end
end
update!(q::S,(dq1,c1)::R,(dq2,c2)::R) where {T,S<:STATE,R<:Tuple{S,T}} = begin
    for k in keys(q)
        @. q[k] += c1*dq1[k]+c2*dq2[k]
    end
end
update!(q::S,(dq1,c1)::R,(dq2,c2)::R,(dq3,c3)::R) where {T,S<:STATE,R<:Tuple{S,T}} = begin
    for k in keys(q)
        @. q[k] += c1*dq1[k]+c2*dq2[k]+c3*dq3[k]
    end
end
update!(q::S,(dq1,c1)::R,(dq2,c2)::R,(dq3,c3)::R,(dq4,c4)::R) where {T,S<:STATE,R<:Tuple{S,T}} = begin
    for k in keys(q)
        @. q[k] += c1*dq1[k]+c2*dq2[k]+c3*dq3[k]+c4*dq4[k]
    end
end

# The basic case is when state is simply an array
update!(q::S,(dq1,c1)::R) where {T,S<:Array,R<:Tuple{S,T}} = begin
    @. q += c1*dq1
end
update!(q::S,(dq1,c1)::R,(dq2,c2)::R) where {T,S<:Array,R<:Tuple{S,T}} = begin
    @. q += c1*dq1+c2*dq2
end
update!(q::S,(dq1,c1)::R,(dq2,c2)::R,(dq3,c3)::R) where {T,S<:Array,R<:Tuple{S,T}} = begin
    @. q += c1*dq1+c2*dq2+c3*dq3
end
update!(q::S,(dq1,c1)::R,(dq2,c2)::R,(dq3,c3)::R,(dq4,c4)::R) where {T,S<:Array,R<:Tuple{S,T}} = begin
    @. q += c1*dq1+c2*dq2+c3*dq3+c4*dq4
end


function advance!(q::S, h, rk3::RK3{M,S}) where {M<:MODEL,S}
    (;dq1,dq2,dq3,model) = rk3

    stage = 1
    rhs!(dq1,q,model,stage)
    update!(q,(dq1,h))
    projection!(q,model,stage)

    stage = 2
    rhs!(dq2,q,model,stage)
    update!(q,(dq1,-3h/4),(dq2,h/4))
    projection!(q,model,stage)

    stage = 0
    rhs!(dq3,q,model,stage)
    update!(q,(dq1,-h/12),(dq2,-h/12),(dq3,2h/3))
    projection!(q,model,stage)
    return q
end


function advance!(q::S, h, rk4::RK4{M,S}) where {M<:MODEL,S}
    (;dq1,dq2,dq3,dq4,model) = rk4

    stage = 1
    rhs!(dq1,q,model,stage)
    update!(q,(dq1,h/2))
    projection!(q,model,stage)

    stage = 2
    rhs!(dq2,q,model,stage)
    update!(q,(dq1,-h/2),(dq2,h/2))
    projection!(q,model,stage)

    stage = 3
    rhs!(dq3,q,model,stage)
    update!(q,(dq2,-h/2),(dq3,h/1))
    projection!(q,model,stage)

    stage = 0
    rhs!(dq4,q,model,stage)
    update!(q,(dq1,h/6),(dq2,h/3),(dq3,-2h/3),(dq4,h/6))
    projection!(q,model,stage)

    return q
end

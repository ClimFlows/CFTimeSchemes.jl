module Update

using MutatingOrNot: Void, void

style(x) = x
style(::NamedTuple{names}) where names = NamedStyle{names}()
style(::AbstractArray{T,N}) where {T,N} = ArrayStyle{N}()

update!(x, mgr, u::S, a::F, ka::S) where {F, S} = 
    LinUp(mgr, (a,))(x,u,(ka,))::S
update!(x, mgr, u::S, a::F, ka, b::F, kb) where {F, S} =
    LinUp(mgr, (a,b))(x,u,(ka,kb))::S
update!(x, mgr, u::S, a::F, ka, b::F, kb, c::F, kc) where {F, S} = 
    LinUp(mgr, (a,b,c))(x,u,(ka,kb,kc))::S
update!(x, mgr, u::S, a::F, ka, b::F, kb, c::F, kc, d::F, kd) where {F, S} = 
    LinUp(mgr, (a,b,c,d))(x,u,(ka,kb,kc,kd))::S
# Currently limited to 4-stage schemes

struct LinUp{F,N,Manager} # linear update with coefs=(a,b, ...)
    mgr::Manager # nothing for plain broadcast, see `manage`
    coefs::NTuple{N,F}
end

# we define these "styles" in order to be able
# to dispatch on a tuple of named tuples with the same names but different types
# or tuple of arrays with same rank but different types
struct NamedStyle{names} end
struct ArrayStyle{N} end

# generic LinUp: add style info to facilitate dispatch
(up::LinUp)(x, u, ks::Tuple) = up(x, style(u), map(style, ks), u, ks)

# LinUp on arrays
(up::LinUp{F,N})(x, ::S, ::NTuple{N, S}, u, ks) where {F, N, S<:ArrayStyle} = 
    update_array(Val(N), manage(x, up.mgr), u, up.coefs, ks)

update_array(::Val{1}, x, u, (a,),      (ka,))         = @. x = muladd(a, ka, u)
# we sacrifice one muladd to group the increments (see complex arrays), potentially more accurate
update_array(::Val{2}, x, u, (a,b),     (ka,kb))       = @. x = muladd(b, kb, a*ka) + u
update_array(::Val{3}, x, u, (a,b,c,),  (ka,kb,kc))    = @. x = muladd(c, kc, muladd(b, kb, a*ka)) + u
update_array(::Val{4}, x, u, (a,b,c,d), (ka,kb,kc,kd)) = @. x = muladd(d, kd, muladd(c, kc, muladd(b, kb, a*ka))) + u

# we avoid muladd for complex arrays due to a Zygote/ForwardDiff bug
(up::LinUp{F,N})(x, ::S, ::NTuple{N,S}, u::A, ks) where {N, F, A<:Array{<:Complex}, S<:ArrayStyle} = 
    update_carray(Val(N), manage(x, up.mgr), u, up.coefs, ks)
update_carray(::Val{1}, x, u, (a,),      (ka,))         = @. x = a*ka + u
update_carray(::Val{2}, x, u, (a,b),     (ka,kb))       = @. x = (b*kb + a*ka) + u
update_carray(::Val{3}, x, u, (a,b,c,),  (ka,kb,kc))    = @. x = (c*kc + b*kb + a*ka) + u
update_carray(::Val{4}, x, u, (a,b,c,d), (ka,kb,kc,kd)) = @. x = (d*kd + c*kc + b*kb + a*ka) + u

# LinUp on named tuples
function (up::LinUp{F,N})(x, s::S, ::NTuple{N,S}, u, ka) where {F, N, S<:NamedStyle}
    return map(up, svoid(x,u), u, transp(s, ka))
end

# LinUp on tuples
function (up::LinUp{F,N})(x, u::T, ka::NTuple{N,T}) where {F, N, T<:Tuple}
    return map(up, svoid(x,u), u, transpose_tuple(ka))
end

svoid(::Void, u) = map(uu->void, u)
svoid(x, u) = x

@inline function transp(::NamedStyle{names}, ntup) where {names}
    M = length(names) # compile-time constant
    getindexer(i) = coll->coll[i]
    t = ntuple(
        let nt=ntup
            i->map(getindexer(i), nt)
        end,
        Val{M}())
    return NamedTuple{names}(t)
end

@inline function transpose_tuple(ntup)
    M = length(ntup[1]) # compile-time constant
    getindexer(i) = coll->coll[i]
    return ntuple(
        let nt=ntup
            i->map(getindexer(i), nt)
        end,
        Val{M}())
end

"""
    managed_x = manage(x, mgr)
Return a decorated object `managed_x` to be used as the l.h.s of a broadcasted assignment in place of `x`.
This function is called by update!(mgr, ...) and implemented for `mgr==nothing` as:
    managed(x, ::Nothing) = x

More interesting behavior can be obtained by (i) passing a user-defined `mgr`
to `Update.update!`, (ii) implementing a specialized method for `Update.manage` and (iii) implement special
broadcasting behavior for the object `managed_x`.

For instance:
    CFTimesSchemes.Update.manage(x, mgr::LoopManager) = mgr[x]
where `LoopManager` is a type provided by package `LoopManagers`, which also implements the broadcasting machinery.
"""
manage(x, ::Nothing) = x

end # module

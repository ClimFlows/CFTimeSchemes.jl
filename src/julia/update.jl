module Update

using MutatingOrNot: Void, void

update!(x, mgr, u::S, a::F, ka::S) where {F, S} = 
    LinUp(mgr, (a,))(x,u,(ka,))::S
update!(x, mgr, u::S, a::F, ka::S, b::F, kb::S) where {F, S} = 
    LinUp(mgr, (a,b))(x,u,(ka,kb))::S
update!(x, mgr, u::S, a::F, ka::S, b::F, kb::S, c::F, kc::S) where {F, S} = 
    LinUp(mgr, (a,b,c))(x,u,(ka,kb,kc))::S
update!(x, mgr, u::S, a::F, ka::S, b::F, kb::S, c::F, kc::S, d::F, kd::S) where {F, S} = 
    LinUp(mgr, (a,b,c,d))(x,u,(ka,kb,kc,kd))::S
# Currently limited to 4-stage schemes

struct LinUp{F,N,Manager} # linear update with coefs=(a,b, ...)
    mgr::Manager # nothing for plain broadcast, see `manage`
    coefs::NTuple{N,F}
end

# LinUp on arrays
(up::LinUp{F,N})(x, u::A, ks::NTuple{N,A}) where {N, F, A<:AbstractArray} = 
    update_array(Val(N), manage(x, up.mgr), u, up.coefs, ks)

update_array(::Val{1}, x, u, (a,),      (ka,))         = @. x = muladd(a, ka, u)
# we sacrifice one muladd to group the increments (see complex arrays), potentially more accurate
update_array(::Val{2}, x, u, (a,b),     (ka,kb))       = @. x = muladd(b, kb, a*ka) + u
update_array(::Val{3}, x, u, (a,b,c,),  (ka,kb,kc))    = @. x = muladd(c, kc, muladd(b, kb, a*ka)) + u
update_array(::Val{4}, x, u, (a,b,c,d), (ka,kb,kc,kd)) = @. x = muladd(d, kd, muladd(c, kc, muladd(b, kb, a*ka))) + u

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

# we avoid muladd for complex arrays due to a Zygote/ForwardDiff bug
(up::LinUp{F,N})(x, u::A, ks::NTuple{N,A}) where {N, F, A<:Array{<:Complex}} = 
    update_carray(Val(N), manage(x, up.mgr), u, up.coefs, ks)
update_carray(::Val{1}, x, u, (a,),      (ka,))         = @. x = a*ka + u
update_carray(::Val{2}, x, u, (a,b),     (ka,kb))       = @. x = (b*kb + a*ka) + u
update_carray(::Val{3}, x, u, (a,b,c,),  (ka,kb,kc))    = @. x = (c*kc + b*kb + a*ka) + u
update_carray(::Val{4}, x, u, (a,b,c,d), (ka,kb,kc,kd)) = @. x = (d*kd + c*kc + b*kb + a*ka) + u

# LinUp on named tuples
function (up::LinUp{F,N})(x, u::NT, ka::NTuple{N,NT}) where {F, N, names, NT<:NamedTuple{names}}
    return map(up, svoid(x,u), u, transp(ka))
end
svoid(::Void, u) = map(uu->void, u)
svoid(x, u) = x

@inline function transp(ntup::NTuple{N,NT}) where {N, names, NT<:NamedTuple{names}}
    M = length(names) # compile-time constant
    getindexer(i) = coll->coll[i]
    t = ntuple(
        let nt=ntup
            i->map(getindexer(i), nt)
        end,
        Val{M}())
    return NamedTuple{names}(t)
end

end # module

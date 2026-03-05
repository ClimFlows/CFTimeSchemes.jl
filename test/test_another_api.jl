using CFTimeSchemes: MODEL, projection!, advance!, RK3, RK4, STATE
using CFTimeSchemes

# First tests on a FOO model

struct FOO{S}<:MODEL{S}
    state::S
end

Foo(n) = FOO((;u=zeros(n)))


function CFTimeSchemes.rhs!(dq::S,q::S,model::FOO{S},stage) where {S}
    @. dq.u = stage
end

function CFTimeSchemes.projection!(q::S,model::FOO{S},stage) where {S}
    @. q.u = stage
end

function test_foo_scheme(model, timescheme)
    (;state) = model
    scheme = timescheme(model,state)
    dt = 1.0
    advance!(state,dt,scheme)

    @test all(@. scheme.dq1.u == 1)
    @test all(@. scheme.dq2.u == 2)
    @test all(@. state.u == 0)
    if timescheme == RK3
        @test all(@. scheme.dq3.u == 0)
    elseif timescheme == RK4
        @test all(@. scheme.dq3.u == 3)
        @test all(@. scheme.dq4.u == 0)
    end

end

function test_foo()
    n=5
    model = Foo(n)
    state = model.state
    @test state isa STATE
    @test keys(similar(state)) == (:u,)
    @test length(similar(state).u) == n

    for timescheme in [RK3, RK4]
        test_foo_scheme(model,timescheme)
    end
end

# Second tests on the Lorenz model

struct Lorenz{S,T}<:MODEL{S} where T
    state:: S
    sigma::T
    rho::T
    beta::T
end

function CFTimeSchemes.rhs!(dq::S,q::S,(;sigma,rho,beta)::Lorenz{S},stage) where {S}
    x,y,z = q
    dx = sigma*(y-x)
    dy = x*(rho-z)-y
    dz = x*y-beta*z
    @. dq = dx,dy,dz
end


function test_lorenz_scheme(TS)
    (;state) = model = Lorenz(zeros(3), 10.0,28.0,8/3)
    scheme = TS(model,state)
    run(n) = begin
        for _ in 1:n
            advance!(state,dt,scheme)
        end
    end

    state[:] = [0,3,0]
    dt = 0.02
    run(10)
    if TS == RK3
        @test all(state .== [8.804272893448221, 18.251042040845917, 6.4657089552738105])
    elseif TS == RK4
        @test all(state .== [8.810586052614886, 18.258247900759972, 6.4820051132423810])
    end
end

function test_lorenz()
    test_lorenz_scheme(RK3)
    test_lorenz_scheme(RK4)
end

@testset "Another API" begin
    test_foo()
    test_lorenz()
end

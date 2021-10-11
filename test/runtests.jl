module TestingLinearInterpolatorsInterpolations

using LazyAlgebra, TwoDimensional

using LinearInterpolators

using Base: OneTo
using LinearAlgebra, SparseArrays, Printf
using Test
using Statistics: mean

distance(a::Real, b::Real) = abs(a - b)

distance(a::NTuple{2,Real}, b::NTuple{2,Real}) =
    hypot(a[1] - b[1], a[2] - b[2])

distance(A::AffineTransform2D, B::AffineTransform2D) =
    max(abs(A.xx - B.xx), abs(A.xy - B.xy), abs(A.x - B.x),
        abs(A.yx - B.yx), abs(A.yy - B.yy), abs(A.y - B.y))

distance(A::AbstractArray{Ta,N}, B::AbstractArray{Tb,N}) where {Ta,Tb,N} =
    mean(abs.(A - B))

shortname(::Nothing) = ""
shortname(m::RegexMatch) = m.captures[1]
shortname(::Type{T}) where {T} = shortname(string(T))
shortname(str::AbstractString) =
    shortname(match(r"([_A-Za-z][_A-Za-z0-9]*)([({]|$)", str))

kernels = ([BSpline{i}() for i in 1:4]...,
           CatmullRomSpline(), CardinalCubicSpline(-0.4),
           MitchellNetravaliSpline(), LanczosKernel{4}(), LanczosKernel{6}())

conditions = (Flat, SafeFlat)

sub = 3
x = range(-1, stop=1, length=100*sub + 1);
xsub = x[1:sub:end];
y = cos.(2 .* x .* (x .+ 2));
ysub = y[1:sub:end];
t = range(1, stop=length(xsub), length=length(x));

@testset "Utilities" begin
    let to_axis = LinearInterpolators.to_axis
        @test to_axis(3) === OneTo{Int}(3)
        @test to_axis(UInt(3)) === OneTo{Int}(3)
        @test to_axis(OneTo{Int}(3)) === OneTo{Int}(3)
        @test to_axis(OneTo{UInt}(3)) === OneTo{Int}(3)
        @test to_axis(2:8) === 2:8
        @test to_axis(UInt(2):UInt(8)) === 2:8
    end

    let check_size = LinearInterpolators.check_size
        @test check_size(()) == 1
        @test check_size((2,)) == 2
        @test check_size((UInt16(2),Int16(3))) === Int(6)
        @test check_size((2,3,0)) == 0
        @test_throws ErrorException check_size((2,3,-12))
    end

    let with_eltype = LinearInterpolators.with_eltype
        ker = BSpline{3,Float32}()
        @test with_eltype(eltype(ker), ker) === ker
        @test with_eltype(Float64, ker) === BSpline{3,Float64}()
        A = rand(Float32, (2, 3))
        @test with_eltype(eltype(A), A) === A
        @test with_eltype(Float64, A) == convert(Array{Float64}, A)
        @test_throws ArgumentError with_eltype(AbstractFloat, A)
        r = axes(A,1)
        @test with_eltype(eltype(r), r) === r
        @test with_eltype(Int16, r) === Base.OneTo{Int16}(length(r))
        r = 2:7
        @test with_eltype(eltype(r), r) === r
        @test with_eltype(Float32, r) === Float32(first(r)):Float32(last(r))
        r = -3:2:11
        @test with_eltype(eltype(r), r) === r
        @test with_eltype(Float32, r) === Float32(first(r)):Float32(step(r)):Float32(last(r))
    end

    let promote_kernel = LinearInterpolators.promote_kernel
        ker = BSpline{3,Float32}()
        siz = (2, 3)
        @test promote_kernel(ker) === ker
        @test promote_kernel(ker, eltype(ker)) === ker
        @test promote_kernel(ker, Float64) === Kernel{Float64}(ker)
        @test promote_kernel(ker, rand(eltype(ker), siz)) === ker
        @test promote_kernel(ker, rand(Float64, siz)) === Kernel{Float64}(ker)
        @test promote_kernel(ker, rand(eltype(ker), 2), rand(eltype(ker), 3)) === ker
        @test promote_kernel(ker, rand(Float32, 2), rand(Float64, 3)) === Kernel{Float64}(ker)
    end

    for T in (Float32, Float64), S in (2, 3)
        ker = BSpline{S,T}()
        for C in (Flat, SafeFlat)
            @test isa(C(ker, 5), C{T,S})
            @test isa(C{T}(ker, 5), C{T,S})
            @test isa(C{T,S}(ker, 5), C{T,S})
            @test isa(C{T,S}(5), C{T,S})
            @test C{T,S}(5) === C{T,S}(OneTo(5))
            @test isa(C{T,S}(2:5), C{T,S,typeof(2:5)})
        end
    end
end

#const VERBOSE = true
#@testset "TabulatedInterpolators" begin
#    tol = 1e-14
#
#    for K in kernels,
#        C in conditions
#        tol = isa(K, BSpline{1}) ? 0.02 : 0.006
#        ker = C(K)
#        T = TabulatedInterpolator(ker, t, length(xsub))
#        @test distance(T(ysub), T*ysub) ≤ 0
#        err = distance(T*ysub, y)
#        if VERBOSE
#            print(shortname(typeof(K)),"/",shortname(C)," max. err = ")
#            @printf("%.3g\n", err)
#        end
#        @test err ≤ tol
#        @test distance(T*ysub, apply(ker,t,ysub)) ≤ 1e-15
#
#    end
#end

@testset "SparseInterpolators" begin
    for ker in kernels, C in conditions
        tol = isa(ker, BSpline{1}) ? 0.02 : 0.006
        A = SparseInterpolator(ker, t, length(xsub), C)
        S = sparse(A)
        @test distance(A(ysub), S*ysub) ≤ 1e-15
        @test distance(A(ysub), y) ≤ tol
        #@test distance(S(ysub), apply(ker,t,ysub)) ≤ 1e-15
        b = rand(Float64, size(y))
        w = similar(b)
        for α in (0, -1, 1, 2), β in (0, -1, 1, -3)
            apply!(α,LazyAlgebra.Direct,A,ysub,false,β,copyto!(w,b))
            @test distance(w, α*(S*ysub) + β*b) ≤ 1e-15
        end
    end
    #let ker = kernels[1], T = Float32
    #    @test_deprecated SparseInterpolator(T, ker, t, length(xsub))
    #end
end

#@testset "TwoDimensionalTransformInterpolator" begin
#    kerlist = (BSpline{1}(), BSpline{2}(), CatmullRomSpline())
#    rows = (11,16)
#    cols = (10,15)
#
#    # Define a rotation around c.
#    c = (5.6, 6.4)
#    R = c + rotate(AffineTransform2D{Float64}() - c, 0.2)
#
#    for ker1 in kerlist, ker2 in kerlist
#        A = TwoDimensionalTransformInterpolator(rows, cols, ker1, ker2, R)
#        x = randn(cols)
#        y = randn(rows)
#        #x0 = randn(cols)
#        #y0 = Array{Float64}(undef, rows)
#        #x1 = Array{Float64}(undef, cols)
#        @test vdot(A*x, y) ≈ vdot(x, A'*y)
#    end
#
#end

end # module

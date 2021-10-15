module TestingLinearInterpolatorsInterpolations

using LazyAlgebra, TwoDimensional

using LinearInterpolators
using LinearInterpolators: getcoefs, limits

using Printf
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

struct CustomCoordinate{T<:Real}
    val::T
end

LinearInterpolators.convert_coordinate(T::Type, c::CustomCoordinate) =
    convert(T, c.val)

kernels = (RectangularSpline(), LinearSpline(), QuadraticSpline(),
           CubicSpline(), CatmullRomSpline(), KeysSpline(-0.4),
           MitchellNetravaliSpline(), LanczosKernel(4), LanczosKernel(6))

conditions = (Flat, SafeFlat)

const VERBOSE = true
sub = 3
x = range(-1, stop=1, length=100*sub + 1);
xsub = x[1:sub:end];
y = cos.(2 .* x .* (x .+ 2));
ysub = y[1:sub:end];
t = range(1, stop=length(xsub), length=length(x));

@testset "`getcoefs` method" begin
    T = Float64
    ker = CatmullRomSpline(T)
    len = 4
    lim = limits(ker, len)
    for x in (0, .1, -1.2)
        @test getcoefs(ker, lim, x) == getcoefs(ker, lim, CustomCoordinate(x))
    end
end

@testset "TabulatedInterpolators" begin
    tol = 1e-14

    for K in kernels,
        C in conditions
        tol = isa(K, RectangularSpline) ? 0.02 : 0.006
        ker = C(K)
        T = TabulatedInterpolator(ker, t, length(xsub))
        @test distance(T(ysub), T*ysub) ≤ 0
        err = distance(T*ysub, y)
        if VERBOSE
            print(shortname(typeof(K)),"/",shortname(C)," max. err = ")
            @printf("%.3g\n", err)
        end
        @test err ≤ tol
        @test distance(T*ysub, apply(ker,t,ysub)) ≤ 1e-15

    end
end

@testset "SparseInterpolators" begin
    for K in kernels, C in conditions
        tol = isa(K, RectangularSpline) ? 0.02 : 0.006
        ker = C(K)
        S = SparseInterpolator(ker, t, length(xsub))
        T = TabulatedInterpolator(ker, t, length(xsub))
        @test distance(S(ysub), S*ysub) ≤ 1e-15
        @test distance(S(ysub), T(ysub)) ≤ 1e-15
        @test distance(S(ysub), y) ≤ tol
        @test distance(S(ysub), apply(ker,t,ysub)) ≤ 1e-15
    end
    let ker = kernels[1], T = Float32
        @test_deprecated SparseInterpolator(T, ker, t, length(xsub))
    end
end

@testset "TwoDimensionalTransformInterpolator" begin
    kerlist = (RectangularSpline(), LinearSpline(), CatmullRomSpline())
    rows = (11,16)
    cols = (10,15)

    # Define a rotation around c.
    c = (5.6, 6.4)
    R = c + rotate(AffineTransform2D{Float64}() - c, 0.2)

    for ker1 in kerlist, ker2 in kerlist
        A = TwoDimensionalTransformInterpolator(rows, cols, ker1, ker2, R)
        x = randn(cols)
        y = randn(rows)
        #x0 = randn(cols)
        #y0 = Array{Float64}(undef, rows)
        #x1 = Array{Float64}(undef, cols)
        @test vdot(A*x, y) ≈ vdot(x, A'*y)
    end

end

end # module

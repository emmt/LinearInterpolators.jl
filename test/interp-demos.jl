isdefined(:LinearInterpolators) || include("../src/LinearInterpolators.jl")

module LinearInterpolatorsInterpolationsDemos

using Compat

using LazyAlgebra
using LinearInterpolators.Kernels
using LinearInterpolators.Interpolations

using PyCall
# pygui(:gtk); # can be: :wx, :gtk, :qt
using LaTeXStrings
import PyPlot
const plt = PyPlot


function rundemos(::Type{T} = Float64) where {T<:AbstractFloat}

    z = T[0.5, 0.3, 0.1, 0.0, -0.2, -0.7, -0.7, 0.0, 1.7, 1.9, 2.1]

    # 1-D example
    t = linspace(-3,14,2000);
    I0 = SparseInterpolator(RectangularSpline(T, Flat), t, length(z))
    I1 = SparseInterpolator(LinearSpline(T, Flat), t, length(z))
    I2 = SparseInterpolator(CatmullRomSpline(T, SafeFlat), t, length(z))
    I3 = SparseInterpolator(LanczosKernel(T, 8), t, length(z))
    plt.figure(1)
    plt.clf()
    plt.plot(t, I0(z), color="darkgreen",
             linewidth=1.5, linestyle="-", label="Rectangular spline");
    plt.plot(t, I1(z), color="darkred",
                 linewidth=1.5, linestyle="-", label="Linear spline");
    plt.plot(t, I2(z), color="orange",
             linewidth=1.5, linestyle="-", label="Catmull-Rom");
    plt.plot(t, I3(z), color="violet",
             linewidth=1.5, linestyle="-", label="Lanczos 8");
    plt.plot(1:length(z), z, "o", color="darkblue", label="Points")
    plt.legend()
    plt.title("Interpolations with Cardinal Kernels")

    # Interpolation + smoothing
    t = linspace(-3,14,2000);
    I5 = SparseInterpolator(QuadraticSpline(T), t, length(z))
    I6 = SparseInterpolator(CubicSpline(T, Flat), t, length(z))
    I7 = SparseInterpolator(MitchellNetravaliSpline(T, Flat), t, length(z))
    plt.figure(2)
    plt.clf()
    plt.plot(t, I6(z), color="darkred",
                 linewidth=1.5, linestyle="-", label="Cubic B-spline");
    plt.plot(t, I5(z), color="darkgreen",
             linewidth=1.5, linestyle="-", label="Quadratic B-spline");
    #plt.plot(t, I7(z), color="orange",
    #         linewidth=1.5, linestyle="-", label="Mitchell & Netravali");
    plt.plot(1:length(z), z, "o", color="darkblue", label="Points")
    plt.legend()
    plt.title("Interpolations with Smoothing Kernels")

    # Test conversion to a sparse matrix.
    S1 = sparse(I1)
    S2 = sparse(I2)
    println("max. error 1: ", maximum(abs.(I1(z) - S1*z)))
    println("max. error 2: ", maximum(abs.(I2(z) - S2*z)))

    # 2-D example
    t = reshape(linspace(-3,14,20*21), (20,21));
    I = SparseInterpolator(CatmullRomSpline(T, Flat), t, length(z))
    plt.figure(3)
    plt.clf()
    plt.imshow(I(z));
end
#rundemos()
#------------------------------------------------------------------------------

const CPU = 2.94e9

function printtest(prefix::String, count::Integer, tup)
    elt, mem, gct = tup[2:4]
    @printf("%s   %6.0f cycles (Mem. = %8d bytes   GC = %3.0f%%)\n",
            prefix, CPU*elt/count, mem, 1e2*gct/elt)
end

function testdirect!{T,S,B}(dst::Vector{T}, ker::Kernel{T,S,B},
                            x::AbstractVector{T}, src::Vector{T},
                            cnt::Integer)
    for k in 1:cnt
        apply!(dst, Direct, ker, x, src)
    end
    return dst
end

function testadjoint!{T,S,B}(dst::Vector{T}, ker::Kernel{T,S,B},
                             x::AbstractVector{T}, src::Vector{T},
                             cnt::Integer)
    for k in 1:cnt
        apply!(dst, Adjoint, ker, x, src)
    end
    return dst
end

function runtests{T<:AbstractFloat}(::Type{T} = Float64,
                                    len::Integer = 1000, cnt::Integer = 100)
    dim = 500
    w = 10
    K01 = RectangularSpline(T, Flat)
    K02 = RectangularSpline(T, SafeFlat)
    K03 = LinearSpline(T, Flat)
    K04 = LinearSpline(T, SafeFlat)
    K05 = QuadraticSpline(T, Flat)
    K06 = QuadraticSpline(T, SafeFlat)
    K07 = CardinalCubicSpline(T, -1, Flat)
    K08 = CardinalCubicSpline(T, -1, SafeFlat)
    K09 = CatmullRomSpline(T, Flat)
    K10 = CatmullRomSpline(T, SafeFlat)
    x = rand(T(1 - w):T(dim + w), len)
    y = randn(T, dim)
    z = Array{T}(undef, len)
    # compile methods
    testdirect!(z,  K01, x, y, 3)
    testadjoint!(y, K01, x, z, 3)
    testdirect!(z,  K02, x, y, 3)
    testadjoint!(y, K02, x, z, 3)
    testdirect!(z,  K03, x, y, 3)
    testadjoint!(y, K03, x, z, 3)
    testdirect!(z,  K04, x, y, 3)
    testadjoint!(y, K04, x, z, 3)
    testdirect!(z,  K05, x, y, 3)
    testadjoint!(y, K05, x, z, 3)
    testdirect!(z,  K06, x, y, 3)
    testadjoint!(y, K08, x, z, 3)
    testdirect!(z,  K07, x, y, 3)
    testadjoint!(y, K07, x, z, 3)
    testdirect!(z,  K08, x, y, 3)
    testadjoint!(y, K08, x, z, 3)
    testdirect!(z,  K09, x, y, 3)
    testadjoint!(y, K09, x, z, 3)
    testdirect!(z,  K10, x, y, 3)
    testadjoint!(y, K10, x, z, 3)
    # warm-up
    tmp = randn(T, 1_000_000)
    # run tests
    n = cnt*len
    printtest("RectangularSpline   Flat     direct ", n, @timed(testdirect!(z,  K01, x, y, cnt)))
    printtest("RectangularSpline   Flat     adjoint", n, @timed(testadjoint!(y, K01, x, z, cnt)))
    printtest("RectangularSpline   SafeFlat direct ", n, @timed(testdirect!(z,  K02, x, y, cnt)))
    printtest("RectangularSpline   SafeFlat adjoint", n, @timed(testadjoint!(y, K02, x, z, cnt)))
    println()
    printtest("LinearSpline        Flat     direct ", n, @timed(testdirect!(z,  K03, x, y, cnt)))
    printtest("LinearSpline        Flat     adjoint", n, @timed(testadjoint!(y, K03, x, z, cnt)))
    printtest("LinearSpline        SafeFlat direct ", n, @timed(testdirect!(z,  K04, x, y, cnt)))
    printtest("LinearSpline        SafeFlat adjoint", n, @timed(testadjoint!(y, K04, x, z, cnt)))
    println()
    printtest("QuadraticSpline     Flat     direct ", n, @timed(testdirect!(z,  K05, x, y, cnt)))
    printtest("QuadraticSpline     Flat     adjoint", n, @timed(testadjoint!(y, K05, x, z, cnt)))
    printtest("QuadraticSpline     SafeFlat direct ", n, @timed(testdirect!(z,  K06, x, y, cnt)))
    printtest("QuadraticSpline     SafeFlat adjoint", n, @timed(testadjoint!(y, K06, x, z, cnt)))
    println()
    printtest("CardinalCubicSpline Flat     direct ", n, @timed(testdirect!(z,  K07, x, y, cnt)))
    printtest("CardinalCubicSpline Flat     adjoint", n, @timed(testadjoint!(y, K07, x, z, cnt)))
    printtest("CardinalCubicSpline SafeFlat direct ", n, @timed(testdirect!(z,  K08, x, y, cnt)))
    printtest("CardinalCubicSpline SafeFlat adjoint", n, @timed(testadjoint!(y, K08, x, z, cnt)))
    println()
    printtest("CatmullRomSpline    Flat     direct ", n, @timed(testdirect!(z,  K09, x, y, cnt)))
    printtest("CatmullRomSpline    Flat     adjoint", n, @timed(testadjoint!(y, K09, x, z, cnt)))
    printtest("CatmullRomSpline    SafeFlat direct ", n, @timed(testdirect!(z,  K10, x, y, cnt)))
    printtest("CatmullRomSpline    SafeFlat adjoint", n, @timed(testadjoint!(y, K10, x, z, cnt)))

end

end # module

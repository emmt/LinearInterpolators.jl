#
# kernels.jl --
#
# Kernel functions used for linear filtering, windowing or linear
# interpolation.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2017, Éric Thiébaut.
# This file is part of TiPi.  All rights reserved.
#

module Kernels

import Base: eltype, length, size, convert

export
    boundaries,
    getweights,
    isnormalized,
    iscardinal,
    Boundaries,
    Flat,
    SafeFlat,
    Kernel,
    RectangularSpline,
    LinearSpline,
    QuadraticSpline,
    CardinalCubicSpline,
    CatmullRomSpline

#------------------------------------------------------------------------------
# EXTRAPOLATION METHODS

"""
All extrapolation methods (a.k.a. boundary conditions) are singletons and
inherit from the abstract type `Boundaries`.
"""
abstract Boundaries

immutable Flat     <: Boundaries; end
immutable SafeFlat <: Boundaries; end
#immutable Periodic <: Boundaries; end
#immutable Reflect  <: Boundaries; end

#------------------------------------------------------------------------------
# INTERPOLATION KERNELS

# This function is needed for rational constants.
@inline _{T<:AbstractFloat}(::Type{T}, num::Real, den::Real) = T(num)/T(den)

two{T<:Number}(::Type{T}) = convert(T,2)
three{T<:Number}(::Type{T}) = convert(T,3)
half{T<:AbstractFloat}(::Type{T}, n::Integer) = convert(T,n)/two(T)
for T in subtypes(AbstractFloat)
    @eval half(::Type{$T}) = $(half(T,1))
end

"""
# Interpolation Kernels

An interpolation kernel `Interpolations.Kernel{T,S,B}` is parametrized by the
floating-point type `T` of its coefficients, by the size `S` of its support and
by the boundary conditions to apply for extrapolation.  For efficiency reasons,
only kernels with (small) finite size supports are implemented.

A kernel may be used as a function wit a real argument:

    ker(x::Real)

yields kernel value at offset `x`.  All kernel supports are symmetric; that is
`ker(x)` is zero if `abs(x) > S/2`.  The argument can also be a floating-point
type and/or a boundary conditions type:

    ker(::Type{T}, ::Type{B}) where {T<:AbstractFloat,B<:Boundaries}

to convert the kernel to operate with given floating-point type `T` and use
boundary conditions `B` (any of which can be omitted and their order is
irrelevant). Beware that changing the floating-point type may lead to a loss of
precision if the new floating-point type has more digits).  The following
methods are available for any interpolation kernel `ker`:

    eltype(ker) -> T

yields the floating-point type for calculations,

    length(ker) -> S
    size(ker)   -> S

yield the size the support of `ker` which is also the number of neighbors
involved in an interpolation by this kernel,

    boundaries(ker) -> B

yields the type of the boundary conditions applied for extrapolation; finally:


    getweights(ker, t) -> w1, w2, ..., wS

yields the `S` interpolation weights for offset `t ∈ [0,1]` if `S` is even or
or for `t ∈ [-1/2,+1/2]` is `S` is odd.
"""
abstract Kernel{T<:AbstractFloat,S,B<:Boundaries}

eltype{T,S,B}(::Kernel{T,S,B}) = T
eltype{T,S,B}(::Type{Kernel{T,S,B}}) = T
length{T,S,B}(::Kernel{T,S,B}) = S
size{T,S,B}(::Kernel{T,S,B}) = S
#length{T,S}(::Type{Kernel{T,S,B}}) = S  # FIXME: does not work with Julia 0.5

"""
`boundaries(ker)` yields the type of the boundary conditions applied for
extrapolation with kernel `ker`.
"""
boundaries{T,S,B}(::Kernel{T,S,B}) = B

"""
`isnormalized(ker)` returns a boolean indicating whether the kernel `ker` has
the partition of unity property.  That is, the sum of the values computed by
the kernel `ker` on a unit spaced grid is equal to one.
"""
isnormalized{K<:Kernel}(::K) = isnormalized(K)

"""
`iscardinal(ker)` returns a boolean indicating whether the kernel `ker`
is zero for non-zero integer arguments.
"""
iscardinal{K<:Kernel}(::K) = iscardinal(K)

#------------------------------------------------------------------------------
"""
# Rectangular Spline

The rectangular spline (also known as box kernel or Fourier window or Dirichlet
window) is a 1st order (constant) B-spline equals to `1` on `[-1/2,+1/2)`,
and `0` elsewhere.

"""
immutable RectangularSpline{T,B} <: Kernel{T,1,B}; end

iscardinal{K<:RectangularSpline}(::Type{K}) = true

isnormalized{K<:RectangularSpline}(::Type{K}) = true

(ker::RectangularSpline{T,B}){T<:AbstractFloat,B}(x::T) =
    -half(T) ≤ x < half(T) ? one(T) : zero(T)

@inline getweights{T<:AbstractFloat,B}(::RectangularSpline{T,B}, t::T) = one(T)

#------------------------------------------------------------------------------
"""
# Linear Spline

The linear spline (also known as triangle kernel or Bartlett window or Fejér
window) is a 2nd order (linear) B-spline.

"""
immutable LinearSpline{T,B} <: Kernel{T,2,B}; end

iscardinal{K<:LinearSpline}(::Type{K}) = true

isnormalized{K<:LinearSpline}(::Type{K}) = true

(ker::LinearSpline{T,B}){T<:AbstractFloat,B}(x::T) =
    (a = abs(x); a < one(T) ? one(T) - a : zero(T))

@inline getweights{T<:AbstractFloat,B}(::LinearSpline{T,B}, t::T) =
    one(T) - t, t

#------------------------------------------------------------------------------
"""
# Quadratic Spline

The quadratic spline is 3rd order (quadratic) B-spline.
"""
immutable QuadraticSpline{T,B} <: Kernel{T,3,B}; end

iscardinal{K<:QuadraticSpline}(::Type{K}) = false

isnormalized{K<:QuadraticSpline}(::Type{K}) = true

function (ker::QuadraticSpline{T,B}){T<:AbstractFloat,B<:Boundaries}(x::T)
    t = abs(x)
    if t ≥ _(T,3,2)
        return zero(T)
    elseif t ≤ _(T,1,2)
        return _(T,3,4) - t*t
    else
        t -= _(T,3,2)
        return _(T,1,2)*t*t
    end
end

@inline function getweights{T<:AbstractFloat,B}(::QuadraticSpline{T,B}, t::T)
    #return (T(1/8)*(one(T) - two(T)*t)^2,
    #        T(3/4) - t^2,
    #        T(1/8)*(one(T) + two(T)*t)^2)
    const c1 = T(0.35355339059327376220042218105242451964241796884424)
    const c2 = T(0.70710678118654752440084436210484903928483593768847)
    const c3 = T(3)/T(4)
    c2t = c2*t
    q1 = c1 - c2t
    q3 = c1 + c2t
    return (q1*q1, c3 - t*t, q3*q3)
end

#------------------------------------------------------------------------------
"""
# Cubic Spline

    CubicSpline([::Type{T} = Float64,] [::Type{B} = Flat])

where `T <: AbstractFloat` and `B <: Boundaries` yields a cubic spline kernel
which operates with floating-point type `T` and use boundary conditions `B`
(any of which can be omitted and their order is irrelevant).

The 4th order (cubic) B-spline kernel is also known as Parzen window or de la
Vallée Poussin window.
"""
immutable CubicSpline{T,B} <: Kernel{T,4,B}; end

iscardinal{K<:CubicSpline}(::Type{K}) = false

isnormalized{K<:CubicSpline}(::Type{K}) = true

function (::Type{CubicSpline{T,B}}){T<:AbstractFloat,B}(x::T)
    t = abs(x);
    if t ≥ T(2)
        return zero(T)
    elseif t ≤ one(T)
        return (_(T,1,2)*t - one(T))*t*t + _(T,2,3)
    else
        t = T(2) - t
        return _(T,1,6)*t*t*t
    end
end

@inline function getweights{T<:AbstractFloat,B}(ker::CubicSpline{T,B}, t::T)
    error("FIXME: not yet implemented")
end

#------------------------------------------------------------------------------
# Catmull-Rom kernel is a special case of Mitchell & Netravali kernel.

immutable CatmullRomSpline{T,B} <: Kernel{T,4,B}; end

iscardinal{K<:CatmullRomSpline}(::Type{K}) = true

isnormalized{K<:CatmullRomSpline}(::Type{K}) = true

function (ker::CatmullRomSpline{T,B}){T<:AbstractFloat,B}(x::T)
    t = abs(x)
    t ≥ two(T) ? zero(T) :
    t ≤ one(T) ? (_(T,3,2)*t - _(T,5,2))*t*t + one(T) :
    ((_(T,5,2) - _(T,1,2)*t)*t - T(4))*t + T(2)
end

@inline function getweights{T<:AbstractFloat,B}(::CatmullRomSpline{T,B}, t::T)
    # 10 operations
    s = one(T) - t
    q = (T(-1)/T(2))*t*s
    w1 = q*s
    w4 = q*t
    r = w4 - w1
    w2 = s - w1 + r
    w3 = t - w4 - r
    return (w1, w2, w3, w4)
end

#------------------------------------------------------------------------------
"""
    CardinalCubicSpline(T,c) -> ker

yields a cardinal cubic spline interpolation kernel for floating-point type `T`
and tension parameter `c`.  The slope at `x = ±1` is `∓(1 - c)/2`.  Usually
`c ≤ 1`, choosing `c = 0` yields a Catmull-Rom spline, `c = 1` yields all zero
tangents, `c = -1` yields a truncated approximation of a cardinal sine.

"""
immutable CardinalCubicSpline{T,B} <: Kernel{T,4,B}
    α::T
    β::T
    function CardinalCubicSpline(c::Real)
        #@assert c ≤ 1
        new((c - 1)/2, (c + 1)/2)
    end
end

iscardinal{K<:CardinalCubicSpline}(::Type{K}) = true

isnormalized{K<:CardinalCubicSpline}(::Type{K}) = true

function CardinalCubicSpline{T<:AbstractFloat,B<:Boundaries}(::Type{T}, c::Real,
                                                             ::Type{B} = Flat)
    CardinalCubicSpline{T,B}(c)
end

function CardinalCubicSpline{T<:AbstractFloat,B<:Boundaries}(c::T,
                                                             ::Type{B} = Flat)
    CardinalCubicSpline{T,B}(c)
end

function convert{T<:AbstractFloat,B<:Boundaries}(::Type{CardinalCubicSpline{T,B}},
                                                 ker::CardinalCubicSpline)
    CardinalCubicSpline{T,B}(ker.α + ker.β)
end

function (ker::CardinalCubicSpline{T,B}){T<:AbstractFloat,B}(x::T)
    t = abs(x)
    if t < one(T)
        const l = one(T)
        return ((ker.β*t + t)*t - t - l)*(t - l)
    elseif t < two(T)
        r = two(T) - t
        s = t - one(T)
        return ker.α*s*r*r
    else
        return zero(T)
    end
end

@inline function getweights{T<:AbstractFloat,B}(ker::CardinalCubicSpline{T,B},
                                                t::T)
    α = ker.α
    β = ker.β
    # Computation of:
    #     w1 = α s² t
    #     w2 = s + t s² - β s t²
    #     w3 = t + t² s - β s² t
    #     w4 = α s t²
    # with s = 1 - t in 13 operations.
    s = one(T) - t
    st = s*t
    ast = α*st
    return (ast*s,
            (s - β*t)*st + s,
            (t - β*s)*st + t,
            ast*t)
end

#------------------------------------------------------------------------------
"""
# Mitchell & Netravali Kernels

These kernels are cubic splines which depends on 2 parameters `b` and `c`.
whatever the values of `(b,c)`, all these kernels are "normalized", symmetric
and their value and first derivative are continuous.

Taking `b = 0` is a sufficient and necessary condition to have cardinal
kernels.  This correspond to Keys's family of kernels.

Using the constraint: `b + 2c = 1` yields a cubic filter with, at least,
quadratic order approximation.

Some specific values of `(b,c)` yield other well known kernels:

    (b,c) = (1,0)     ==> cubic B-spline
    (b,c) = (0,-a)    ==> Keys's cardinal cubics
    (b,c) = (0,1/2)   ==> Catmull-Rom cubics
    (b,c) = (b,0)     ==> Duff's tensioned B-spline
    (b,c) = (1/3,1/3) ==> recommended by Mitchell-Netravali

Reference:

* Mitchell & Netravali ("Reconstruction Filters in Computer Graphics",
  Computer Graphics, Vol. 22, Number. 4, August 1988).
  http://www.cs.utexas.edu/users/fussell/courses/cs384g/lectures/mitchell/Mitchell.pdf.

"""
immutable MitchellNetraviliSpline{T,B} <: Kernel{T,4,B}
    b ::T
    c ::T
    p0::T
    p2::T
    p3::T
    q0::T
    q1::T
    q2::T
    q3::T
    function MitchellNetraviliSpline(b::Real, c::Real)
        new(b, c,
            (   6 -  2*b       )/6,
            ( -18 + 12*b +  6*c)/6,
            (  12 -  9*b -  6*c)/6,
            (        8*b + 24*c)/6,
            (     - 12*b - 48*c)/6,
            (        6*b + 30*c)/6,
            (     -    b -  6*c)/6)
    end
end

function MitchellNetraviliSpline{T<:AbstractFloat,B<:Boundaries}(
    ::Type{T}, b::Real, c::Real, ::Type{B} = Flat)
    MitchellNetraviliSpline{T,B}(T(b), T(c))
end

# Create Mitchell-Netravali kernel with default parameters.
function MitchellNetraviliSpline{T<:AbstractFloat,B<:Boundaries}(
    ::Type{T} = Float64, ::Type{B} = Flat)
    MitchellNetraviliSpline{T,B}(_(T,1,3), _(T,1,3))
end

iscardinal{T<:AbstractFloat,B}(ker::MitchellNetraviliSpline{T,B}) =
    (ker.b == zero(T))

isnormalized{T<:MitchellNetraviliSpline}(::Type{T}) = true

function (ker::MitchellNetraviliSpline{T,B}){T<:AbstractFloat,B}(x::T)
    t = abs(x)
    t ≥ T(2) ? zero(T) :
    t ≤ one(T) ? (ker.p3*t + ker.p2)*t*t + ker.p0 :
    ((ker.q3*t + ker.q2)*t + ker.q1)*t + ker.q0
end

@inline function getweights{T<:AbstractFloat,B}(
    ker::MitchellNetraviliSpline{T,B}, t::T)
    error("FIXME: not yet implemented")
end

#------------------------------------------------------------------------------
"""
# Keys cardinal kernels

These kernels are piecewise normalized cardinal cubic spline which depend on
one parameter `a`.

Reference:

* Keys, Robert, G., "Cubic Convolution Interpolation for Digital Image
  Processing", IEEE Trans. Acoustics, Speech, and Signal Processing,
  Vol. ASSP-29, No. 6, December 1981, pp. 1153-1160.

"""
immutable KeysSpline{T,B} <: Kernel{T,4,B}
    a ::T
    p0::T
    p2::T
    p3::T
    q0::T
    q1::T
    q2::T
    q3::T
    function KeysSpline(a::Real)
        new(a, 1, -a - 3, a + 2, -4*a, 8*a, -5*a, a)
    end
end

function KeysSpline{T<:AbstractFloat,B<:Boundaries}(
    ::Type{T}, a::Real, ::Type{B} = Flat)
    KeysSpline{T,B}(a)
end

function KeysSpline{B<:Boundaries}(
    a::Real, ::Type{B} = Flat)
    KeysSpline{Float64,B}(a)
end

iscardinal{K<:KeysSpline}(::Type{K}) = true

isnormalized{K<:KeysSpline}(::Type{K}) = true

function (ker::KeysSpline{T,B}){T<:AbstractFloat,B}(x::T)
    t = abs(x)
    t ≥ T(2) ? zero(T) :
    t ≤ one(T) ? (ker.p3*t + ker.p2)*t*t + ker.p0 :
    ((ker.q3*t + ker.q2)*t + ker.q1)*t + ker.q0
end

@inline function getweights{T<:AbstractFloat,B}(ker::KeysSpline{T,B}, t::T)
    error("FIXME: not yet implemented")
end

#------------------------------------------------------------------------------
"""
# Lanczos Resampling Kernel

`LanczosKernel(T, S)` creates a Lanczos kernel of support size `S` which must
be even.

The Lanczos kernel does not have the partition of unity property.  However,
Lanczos kernel tends to be normalized for large support size.
[link](https://en.wikipedia.org/wiki/Lanczos_resampling)
"""
immutable LanczosKernel{T,S,B} <: Kernel{T,S,B}
    a::T   # 1/2 support
    b::T   # a/pi^2
    c::T   # pi/a
    function LanczosKernel()
        @assert S > 0
        @assert iseven(S)
        a = T(S)/2
        new(a, a/pi^2, pi/a)
    end
end

function LanczosKernel{T<:AbstractFloat,B<:Boundaries}(
    ::Type{T}, s::Integer, ::Type{B} = Flat)
    LanczosKernel{T,Int(s),B}()
end

function LanczosKernel{B<:Boundaries}(
    s::Integer, ::Type{B} = Flat)
    LanczosKernel{Float64,Int(s),B}()
end

iscardinal{K<:LanczosKernel}(::Type{K}) = true

isnormalized{K<:LanczosKernel}(::Type{K}) = false

function (ker::LanczosKernel{T,S,B}){T<:AbstractFloat,S,B}(x::T)
    abs(x) ≥ ker.a ? zero(T) :
    x == zero(T) ? one(T) :
    ker.b*sin(pi*x)*sin(ker.c*x)/(x*x)
end

@inline function getweights{T<:AbstractFloat,S,B}(
    ker::LanczosKernel{T,S,B}, t::T)
    error("FIXME: not yet implemented")
end

#------------------------------------------------------------------------------

# Provide methods for parameter-less kernels.
for K in (:RectangularSpline, :LinearSpline, :QuadraticSpline,
          :CubicSpline, :CatmullRomSpline)
    @eval begin

        # Constructors.
        function $K{T<:AbstractFloat,B<:Boundaries}(::Type{T} = Float64,
                                                    ::Type{B} = Flat)
            $K{T,B}()
        end

        function $K{T<:AbstractFloat,B<:Boundaries}(::Type{B},
                                                    ::Type{T} = Float64)
            $K{T,B}()
        end

        # Conversion to different types.
        function convert{T<:AbstractFloat,B<:Boundaries}(::Type{$K{T,B}}, ::$K)
            $K{T,B}()
        end

    end
end

# Provide methods for all kernels.
for K in subtypes(Kernel)

    # We want that calling the kernel on a different type of real argument than
    # the floting-point type of the kernel convert the argument.
    # Unfortunately, defining:
    #
    #     (ker::$K{T,B}){T<:AbstractFloat,B<:Boundaries}(x::Real) = ker(T(x))
    #
    # leads to ambiguities, the following is ugly but works...
    for T in subtypes(AbstractFloat), R in (subtypes(AbstractFloat)..., Integer)
        if R != T
            @eval @inline (ker::$K{$T,B}){B<:Boundaries}(x::$R) = ker($T(x))
        end
    end

    @eval begin
        # Calling the kernel on an array.
        (ker::$K{T,B}){T<:AbstractFloat,B<:Boundaries}(A::AbstractArray) =
            map((x) -> ker(x), A)

        # Calling the kernel as a function to convert to another floating-point
        # type and/or other boundary conditions.
        (ker::$K{oldT,oldB}){
            oldT<:AbstractFloat, oldB<:Boundaries,
            newT<:AbstractFloat, newB<:Boundaries
        }(::Type{newT}, ::Type{newB}) = convert($K{newT,newB}, ker)

        (ker::$K{oldT,oldB}){
            oldT<:AbstractFloat, oldB<:Boundaries,
            newT<:AbstractFloat, newB<:Boundaries
        }(::Type{newB}, ::Type{newT}) = convert($K{newT,newB}, ker)

        (ker::$K{oldT,oldB}){
            oldT<:AbstractFloat, oldB<:Boundaries,
            newT<:AbstractFloat
        }(::Type{newT}) = convert($K{newT,oldB}, ker)

        (ker::$K{oldT,oldB}){
            oldT<:AbstractFloat, oldB<:Boundaries,
            newB<:Boundaries
        }(::Type{newB}) = convert($K{oldT,newB}, ker)

        # Conversion to the same type.
        function convert{T<:AbstractFloat,B<:Boundaries}(::Type{$K{T,B}},
                                                         ker::$K{T,B})
            ker
        end

    end
end

end # module

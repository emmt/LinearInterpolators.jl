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
    getweights,
    isnormalized,
    iscardinal,
    Kernel,
    RectangularSpline,
    LinearSpline,
    QuadraticSpline,
    CardinalCubicSpline,
    CatmullRomSpline

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

An interpolation kernel `Interpolations.Kernel{T,S}` is parametrized by the
floating-point type `T` of its coefficients and by the size `S` of its support.
For efficiency reasons, only kernels with (small) finite size supports are
implemented.

A kernel may be used as a function wit a real argument:

    ker(x::Real)

yields kernel value at offset `x`.  All kernel supports are symmetric; that is
`ker(x)` is zero if `abs(x) > S/2`.  The argument can also be a type:

    ker(T::DataType)

to convert the kernel to operate with given floating-point type `T`.  The
following methods are available for any interpolation kernel `ker`:

    eltype(ker) -> T

yields the floating-point type for calculations,

    length(ker) -> S
    size(ker)   -> S

yield the size the support of `ker` which is also the number of neighbors
involved in an interpolation by this kernel; finally:


    getweights(ker, t) -> w1, w2, ..., wS

yields the `S` interpolation weights for offset `t ∈ [0,1]` if `S` is even or
or for `t ∈ [-1/2,+1/2]` is `S` is odd.
"""
abstract Kernel{T<:AbstractFloat,S}

eltype{T,S}(::Kernel{T,S}) = T
eltype{T,S}(::Type{Kernel{T,S}}) = T
length{T,S}(::Kernel{T,S}) = S
size{T,S}(::Kernel{T,S}) = S
#length{T,S}(::Type{Kernel{T,S}}) = S  # FIXME: does not work with Julia 0.5

"""
`isnormalized(ker)` returns a boolean indicating whether the kernel `ker` has
the partition of unity property.  That is, the sum of the values computed by
the kernel `ker` on a unit spaced grid is equal to one.
"""
isnormalized{T<:Kernel}(::T) = isnormalized(T)

"""
`iscardinal(ker)` returns a boolean indicating whether the kernel `ker`
is zero for non-zero integer arguments.
"""
iscardinal{T<:Kernel}(::T) = iscardinal(T)

#------------------------------------------------------------------------------
"""
# Rectangular Spline

The rectangular spline (also known as box kernel or Fourier window or Dirichlet
window) is a 1st order (constant) B-spline equals to `1` on `[-1/2,+1/2)`,
and `0` elsewhere.

"""
immutable RectangularSpline{T} <: Kernel{T,1}; end

RectangularSpline{T}(::Type{T}) = RectangularSpline{T}()

iscardinal{T<:RectangularSpline}(::Type{T}) = true

isnormalized{T<:RectangularSpline}(::Type{T}) = true

convert{T}(::Type{RectangularSpline{T}}, ker::RectangularSpline{T}) = ker

convert{T}(::Type{RectangularSpline{T}}, ::RectangularSpline) =
    RectangularSpline(T)

(ker::RectangularSpline){T<:AbstractFloat}(::Type{T}) = RectangularSpline{T}()

(ker::RectangularSpline{T}){T<:AbstractFloat}(x::T) =
    -half(T) ≤ x < half(T) ? one(T) : zero(T)

@inline getweights{T<:AbstractFloat}(::RectangularSpline{T}, t::T) = one(T)

#------------------------------------------------------------------------------
"""
# Linear Spline

The linear spline (also known as triangle kernel or Bartlett window or Fejér
window) is a 2nd order (linear) B-spline.

"""
immutable LinearSpline{T} <: Kernel{T,2}; end

LinearSpline{T}(::Type{T}) = LinearSpline{T}()

iscardinal{T<:LinearSpline}(::Type{T}) = true

isnormalized{T<:LinearSpline}(::Type{T}) = true

convert{T}(::Type{LinearSpline{T}}, ker::LinearSpline{T}) = ker

convert{T}(::Type{LinearSpline{T}}, ::LinearSpline) =
    LinearSpline(T)

(ker::LinearSpline){T<:AbstractFloat}(::Type{T}) = LinearSpline{T}()

(ker::LinearSpline{T}){T<:AbstractFloat}(x::T) =
    (a = abs(x); a < one(T) ? one(T) - a : zero(T))

@inline getweights{T<:AbstractFloat}(::LinearSpline{T}, t::T) = one(T) - t, t

#------------------------------------------------------------------------------
"""
# Quadratic Spline

The quadratic spline is 3rd order (quadratic) B-spline.
"""
immutable QuadraticSpline{T} <: Kernel{T,3}; end

QuadraticSpline{T}(::Type{T}) = QuadraticSpline{T}()

iscardinal{T<:QuadraticSpline}(::Type{T}) = false

isnormalized{T<:QuadraticSpline}(::Type{T}) = true

convert{T}(::Type{QuadraticSpline{T}}, ker::QuadraticSpline{T}) = ker

convert{T}(::Type{QuadraticSpline{T}}, ::QuadraticSpline) =
    QuadraticSpline(T)

(ker::QuadraticSpline){T<:AbstractFloat}(::Type{T}) = QuadraticSpline{T}()

function (ker::QuadraticSpline{T}){T<:AbstractFloat}(x::T)
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

@inline function getweights{T<:AbstractFloat}(::QuadraticSpline{T}, t::T)
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

The 4th order (cubic) B-spline kernel is also known as Parzen window or de la
Vallée Poussin window.
"""
immutable CubicSpline{T} <: Kernel{T,4}; end

CubicSpline{T<:AbstractFloat}(::Type{T}) = CubicSpline{T}()

iscardinal{T<:CubicSpline}(::Type{T}) = false

isnormalized{T<:CubicSpline}(::Type{T}) = true

(ker::CubicSpline){T<:AbstractFloat}(::Type{T}) = CubicSpline{T}()

function (::Type{CubicSpline{T}}){T<:AbstractFloat}(x::T)
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

@inline function getweights{T<:AbstractFloat}(ker::CubicSpline{T},
                                              t::T)
    error("FIXME: not yet implemented")
end

#------------------------------------------------------------------------------
# Catmull-Rom kernel is a special case of Mitchell & Netravali kernel.

immutable CatmullRomSpline{T} <: Kernel{T,4}; end

CatmullRomSpline{T}(::Type{T}) = CatmullRomSpline{T}()

iscardinal{T<:CatmullRomSpline}(::Type{T}) = true

isnormalized{T<:CatmullRomSpline}(::Type{T}) = true

convert{T}(::Type{CatmullRomSpline{T}}, ker::CatmullRomSpline{T}) = ker

convert{T}(::Type{CatmullRomSpline{T}}, ::CatmullRomSpline) =
    CatmullRomSpline(T)

(ker::CatmullRomSpline){T<:AbstractFloat}(::Type{T}) = CatmullRomSpline{T}()

function (ker::CatmullRomSpline{T}){T<:AbstractFloat}(x::T)
    t = abs(x)
    t ≥ two(2) ? zero(T) :
    t ≤ one(T) ? (_(T,3,2)*t - _(T,5,2))*t*t + one(T) :
    ((_(T,5,2) - _(T,1,2)*t)*t - T(4))*t + T(2)
end

@inline function getweights{T<:AbstractFloat}(::CatmullRomSpline{T}, t::T)
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
immutable CardinalCubicSpline{T} <: Kernel{T,4}
    α::T
    β::T
    function CardinalCubicSpline(c::Real)
        #@assert c ≤ 1
        new((c - 1)/2, (c + 1)/2)
    end
end

CardinalCubicSpline{T}(::Type{T}, c::Real) = CardinalCubicSpline{T}(c)

convert{T}(::Type{CardinalCubicSpline{T}}, ker::CardinalCubicSpline{T}) = ker

convert{T}(::Type{CardinalCubicSpline{T}}, ker::CardinalCubicSpline) =
    CardinalCubicSpline(T, ker.α + ker.β)

(ker::CardinalCubicSpline){T<:AbstractFloat}(::Type{T}) =
    convert(CardinalCubicSpline{T}, ker)

function (ker::CardinalCubicSpline{T}){T<:AbstractFloat}(x::T)
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

@inline function getweights{T<:AbstractFloat}(ker::CardinalCubicSpline{T}, t::T)
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
immutable MitchellNetraviliSpline{T} <: Kernel{T,4}
    b ::T
    c ::T
    p0::T
    p2::T
    p3::T
    q0::T
    q1::T
    q2::T
    q3::T
    function MitchellNetraviliSpline(b::T, c::T)
        new(b, c,
            T(   6 -  2*b       )/T(6),
            T( -18 + 12*b +  6*c)/T(6),
            T(  12 -  9*b -  6*c)/T(6),
            T(        8*b + 24*c)/T(6),
            T(     - 12*b - 48*c)/T(6),
            T(        6*b + 30*c)/T(6),
            T(   -      b -  6*c)/T(6))
    end
end

MitchellNetraviliSpline{T<:AbstractFloat}(::Type{T}, b::Real, c::Real) =
    MitchellNetraviliSpline{T}(T(b), T(c))

# Create Mitchell-Netravali kernel with default parameters.
MitchellNetraviliSpline{T<:AbstractFloat}(::Type{T}) =
    MitchellNetraviliSpline(T, _(T,1,3), _(T,1,3))

iscardinal{T<:AbstractFloat}(ker::MitchellNetraviliSpline{T}) =
    (ker.b == zero(T))

isnormalized{T<:MitchellNetraviliSpline}(::Type{T}) = true

function (ker::MitchellNetraviliSpline{T}){T<:AbstractFloat}(x::T)
    t = abs(x)
    t ≥ T(2) ? zero(T) :
    t ≤ one(T) ? (ker.p3*t + ker.p2)*t*t + ker.p0 :
    ((ker.q3*t + ker.q2)*t + ker.q1)*t + ker.q0
end

@inline function getweights{T<:AbstractFloat}(ker::MitchellNetraviliSpline{T},
                                              t::T)
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
immutable KeysSpline{T} <: Kernel{T,4}
    a ::T
    p0::T
    p2::T
    p3::T
    q0::T
    q1::T
    q2::T
    q3::T
    function KeysSpline(a::T)
        new(a, 1, -a - 3, a + 2, -4*a, 8*a, -5*a, a)
    end
end

KeysSpline{T<:AbstractFloat}(::Type{T}, a::T) = KeysSpline{T}(T(a))

iscardinal{T<:KeysSpline}(::Type{T}) = true

isnormalized{T<:KeysSpline}(::Type{T}) = true

function (ker::KeysSpline{T}){T<:AbstractFloat}(x::T)
    t = abs(x)
    t ≥ T(2) ? zero(T) :
    t ≤ one(T) ? (ker.p3*t + ker.p2)*t*t + ker.p0 :
    ((ker.q3*t + ker.q2)*t + ker.q1)*t + ker.q0
end

@inline function getweights{T<:AbstractFloat}(ker::KeysSpline{T}, t::T)
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
immutable LanczosKernel{T,S} <: Kernel{T,S}
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

LanczosKernel{T<:AbstractFloat}(::Type{T}, s::Integer) =
    LanczosKernel{T,Int(s)}()

iscardinal{T<:LanczosKernel}(::Type{T}) = true

isnormalized{T<:LanczosKernel}(::Type{T}) = false

function (ker::LanczosKernel{T,S}){T<:AbstractFloat,S}(x::T)
    abs(x) ≥ ker.a ? zero(T) :
    x == zero(T) ? one(T) :
    ker.b*sin(pi*x)*sin(ker.c*x)/(x*x)
end

@inline function getweights{T<:AbstractFloat}(ker::LanczosKernel{T}, t::T)
    error("FIXME: not yet implemented")
end

#------------------------------------------------------------------------------

# Provide less specialized methods:
for K in subtypes(Kernel)
    @eval begin
        (ker::$K{T}){T<:AbstractFloat}(x::Real) = ker(T(x))
        (ker::$K{T}){T<:AbstractFloat}(A::AbstractArray) =
            map((x) -> ker(x), A)
    end
end

end # module

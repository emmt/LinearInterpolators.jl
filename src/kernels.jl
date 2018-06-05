#
# kernels.jl --
#
# Kernel functions used for linear filtering, windowing or linear
# interpolation.
#
#------------------------------------------------------------------------------
#
# This file is part of the LinearInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2016-2018, Éric Thiébaut.
#

module Kernels

import Base: convert

export
    Boundaries,
    CardinalCubicSpline,
    CardinalCubicSpline′,
    CatmullRomSpline,
    CubicSpline,
    CubicSpline′,
    Flat,
    Kernel,
    KeysSpline,
    LanczosKernel,
    LinearSpline,
    LinearSpline′,
    MitchellNetravaliSpline,
    QuadraticSpline,
    QuadraticSpline′,
    RectangularSpline,
    RectangularSpline′,
    SafeFlat,
    boundaries,
    getweights,
    iscardinal,
    isnormalized

#------------------------------------------------------------------------------
# EXTRAPOLATION METHODS

"""
All extrapolation methods (a.k.a. boundary conditions) are singletons and
inherit from the abstract type `Boundaries`.
"""
abstract type Boundaries end

struct Flat     <: Boundaries; end
struct SafeFlat <: Boundaries; end
#struct Periodic <: Boundaries; end
#struct Reflect  <: Boundaries; end

#------------------------------------------------------------------------------
# INTERPOLATION KERNELS

@inline frac(::Type{T}, num::Integer, den::Integer) where {T<:AbstractFloat} =
    (convert(T, num)/convert(T, den))

@inline square(x) = x*x
@inline cube(x) = x*x*x

"""
# Interpolation Kernels

An interpolation kernel `Interpolations.Kernel{T,S,B}` is parametrized by the
floating-point type `T` of its coefficients, by the size `S` of its support and
by the boundary conditions `B` to apply for extrapolation.  For efficiency
reasons, only kernels with (small) finite size supports are implemented.

A kernel may be used as a function wit a real argument:

    ker(x::Real)

yields kernel value at offset `x`.  All kernel supports are symmetric; that is
`ker(x)` is zero if `abs(x) > S/2`.


## Kernel conversion

The argument of a kernel be a floating-point type and/or a boundary conditions
type:

    ker(::Type{T}, ::Type{B}) where {T<:AbstractFloat, B<:Boundaries}

to convert the kernel to operate with given floating-point type `T` and use
boundary conditions `B` (any of which can be omitted and their order is
irrelevant). Beware that changing the floating-point type may lead to a loss of
precision if the new floating-point type has more digits).

It is possible to change the floating-point type of a kernel or its boundary
conditions by something like:

```julia
Float32(ker)    # change floating-point type of kernel `ker`
SafeFlat(ker)   # change boundary conditions of kernel `ker`
```

## Available methods

The following methods are available for any interpolation kernel `ker`:

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
or for `t ∈ [-1/2,+1/2]` if `S` is odd.

"""
abstract type Kernel{T<:AbstractFloat,S,B<:Boundaries} <: Function end

Base.eltype(::Kernel{T,S,B})         where {T,S,B} = T
Base.eltype(::Type{<:Kernel{T,S,B}}) where {T,S,B} = T
Base.length(::Kernel{T,S,B})         where {T,S,B} = S
Base.length(::Type{<:Kernel{T,S,B}}) where {T,S,B} = S
Base.size(::Kernel{T,S,B})           where {T,S,B} = S
Base.size(::Type{<:Kernel{T,S,B}})   where {T,S,B} = S

"""
`boundaries(ker)` yields the type of the boundary conditions applied for
extrapolation with kernel `ker`.
"""
boundaries(::Kernel{T,S,B})         where {T,S,B} = B
boundaries(::Type{<:Kernel{T,S,B}}) where {T,S,B} = B

"""
`isnormalized(ker)` returns a boolean indicating whether the kernel `ker` has
the partition of unity property.  That is, the sum of the values computed by
the kernel `ker` on a unit spaced grid is equal to one.
"""
function isnormalized end

"""
`iscardinal(ker)` returns a boolean indicating whether the kernel `ker`
is zero for non-zero integer arguments.
"""
function iscardinal end

"""
```julia
getweights(ker, t) -> w1, w2, ..., wS
```

yields the interpolation weights for the `S` neighbors of a position `x`.
Offset `t` between `x` and the nearest neighbor is in the range `[0,1]` for `S`
even and `[-1/2,+1/2]` for `S` odd.

"""
function getweights end

#------------------------------------------------------------------------------
"""
# Rectangular Spline

The rectangular spline (also known as box kernel or Fourier window or Dirichlet
window) is the 1st order (constant) B-spline equals to `1` on `[-1/2,+1/2)`,
and `0` elsewhere.

"""
struct RectangularSpline{T,B} <: Kernel{T,1,B}; end
struct RectangularSpline′{T,B} <: Kernel{T,1,B}; end

iscardinal(::Union{K,Type{K}}) where {K<:RectangularSpline} = true
iscardinal(::Union{K,Type{K}}) where {K<:RectangularSpline′} = false

isnormalized(::Union{K,Type{K}}) where {K<:RectangularSpline} = true
isnormalized(::Union{K,Type{K}}) where {K<:RectangularSpline′} = false

Base.show(io::IO, ::RectangularSpline) = print(io, "RectangularSpline()")
Base.show(io::IO, ::RectangularSpline′) = print(io, "RectangularSpline′()")

Base.ctranspose(::RectangularSpline{T,B}) where {T,B} = RectangularSpline′()

(::RectangularSpline{T,B})(x::T) where {T,B} =
    frac(T,-1,2) ≤ x < frac(T,1,2) ? one(T) : zero(T)

@inline getweights(::RectangularSpline{T,B}, t::T) where {T,B} = one(T)

(::RectangularSpline′{T,B})(x::T) where {T,B} = zero(T)

@inline getweights(::RectangularSpline′{T,B}, t::T) where {T,B} = zero(T)

#------------------------------------------------------------------------------
"""
# Linear Spline

The linear spline (also known as triangle kernel or Bartlett window or Fejér
window) is the 2nd order (linear) B-spline.

"""
struct LinearSpline{T,B} <: Kernel{T,2,B}; end
struct LinearSpline′{T,B} <: Kernel{T,2,B}; end

iscardinal(::Union{K,Type{K}}) where {K<:LinearSpline} = true
iscardinal(::Union{K,Type{K}}) where {K<:LinearSpline′} = false

isnormalized(::Union{K,Type{K}}) where {K<:LinearSpline} = true
isnormalized(::Union{K,Type{K}}) where {K<:LinearSpline′} = false

Base.show(io::IO, ::LinearSpline) = print(io, "LinearSpline()")
Base.show(io::IO, ::LinearSpline′) = print(io, "LinearSpline′()")

Base.ctranspose(::LinearSpline{T,B}) where {T,B} = LinearSpline′()

(::LinearSpline{T,B})(x::T) where {T<:AbstractFloat,B} =
    (a = abs(x); a < 1 ? 1 - a : zero(T))

@inline getweights(::LinearSpline{T,B}, t::T) where {T<:AbstractFloat,B} =
    (1 - t, t)

# The derivative of the linear B-spline must be non-symmetric for tests to
# succeed.  In particular we want that interpolating with the derivative of the
# linear B-spline amounts to taking the finite difference when 0 ≤ t < 1.
# This implies that f'(x) = 1 for x ∈ [-1,0), f'(x) = -1 for x ∈ [0,1), and
# f'(x) = 0 elsewhere.
(::LinearSpline′{T,B})(x::T) where {T<:AbstractFloat,B} =
    -1 ≤ x < 1 ? (x < 0 ? one(T) : -one(T)) : zero(T)

@inline getweights(::LinearSpline′{T,B}, t::T) where {T<:AbstractFloat,B} =
    (-one(T), one(T))

#------------------------------------------------------------------------------
"""
# Quadratic Spline

The quadratic spline is the 3rd order (quadratic) B-spline.
"""
struct QuadraticSpline{T,B} <: Kernel{T,3,B}; end
struct QuadraticSpline′{T,B} <: Kernel{T,3,B}; end

iscardinal(::Union{K,Type{K}}) where {K<:QuadraticSpline} = false
iscardinal(::Union{K,Type{K}}) where {K<:QuadraticSpline′} = false

isnormalized(::Union{K,Type{K}}) where {K<:QuadraticSpline} = true
isnormalized(::Union{K,Type{K}}) where {K<:QuadraticSpline′} = false

Base.show(io::IO, ::QuadraticSpline) = print(io, "QuadraticSpline()")
Base.show(io::IO, ::QuadraticSpline′) = print(io, "QuadraticSpline′()")

Base.ctranspose(::QuadraticSpline{T,B}) where {T,B} = QuadraticSpline′()

function (::QuadraticSpline{T,B})(x::T) where {T<:AbstractFloat,B<:Boundaries}
    a = abs(x)
    return (a ≥ frac(T,3,2) ? zero(T) :
            a ≤ frac(T,1,2) ? frac(T,3,4) - a*a :
            square(a - frac(T,3,2))*frac(T,1,2))
end

@static if false
    # Compute quadratic B-spline weights in 8 operations.
    @inline function getweights(::QuadraticSpline{T,B},
                                t::T) where {T<:AbstractFloat,B}
        # w1 = (1/8)*(1 - 2*t)^2 = (1/2)*((1/2) - t)^2
        # w2 = (3/4) - t^2
        # w3 = (1/8)*(1 + 2*t)^2 = (1/2)*((1/2) + t)^2
        const h = frac(T,1,2)
        return (h*square(h - t), frac(T,3,4) - t*t, h*square(h + t))
    end
else
    # Same result, but with 7 operations.
    @inline function getweights(::QuadraticSpline{T,B},
                                t::T) where {T<:AbstractFloat,B}
        # c1 = 1/sqrt(8)
        const c1 = T(0.35355339059327376220042218105242451964241796884424)
        # c2 = 2/sqrt(8)
        const c2 = T(0.70710678118654752440084436210484903928483593768847)
        # c2 = 3/4
        const c3 = frac(T,3,4)
        c2t = c2*t
        q1 = c1 - c2t
        q3 = c1 + c2t
        return (q1*q1, c3 - t*t, q3*q3)
    end
end

(::QuadraticSpline′{T,B})(x::T) where {T,B} =
    frac(T,-3,2) < x < frac(T,3,2) ? (
        x < frac(T,-1,2) ? x + frac(T,3,2) :
        x ≤ frac(T,1,2) ? -2x : x - frac(T,3,2)
    ) : zero(T)

@inline function getweights(::QuadraticSpline′{T,B},
                            t::T) where {T<:AbstractFloat,B}
    const h = frac(T,1,2)
    return (t - h, -2t, t + h)
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
struct CubicSpline{T,B} <: Kernel{T,4,B}; end
struct CubicSpline′{T,B} <: Kernel{T,4,B}; end

iscardinal(::Union{K,Type{K}}) where {K<:CubicSpline} = false
iscardinal(::Union{K,Type{K}}) where {K<:CubicSpline′} = false

isnormalized(::Union{K,Type{K}}) where {K<:CubicSpline} = true
isnormalized(::Union{K,Type{K}}) where {K<:CubicSpline′} = false

Base.show(io::IO, ::CubicSpline) = print(io, "CubicSpline()")
Base.show(io::IO, ::CubicSpline′) = print(io, "CubicSpline′()")

Base.ctranspose(::CubicSpline{T,B}) where {T,B} = CubicSpline′()

function (::CubicSpline{T,B})(x::T) where {T,B}
    a = abs(x)
    return (a ≥ 2 ? zero(T) :
            a ≥ 1 ? cube(2 - a)*frac(T,1,6) :
            (frac(T,1,2)*a - 1)*a*a + frac(T,2,3))
end

@inline function getweights(ker::CubicSpline{T,B},
                            t::T) where {T,B}
    # The weights are:
    #     w1 = 1/6 - t/2 + t^2/2 - t^3/6
    #        = 1/6 + (t^2 - t)/2 - t^3/6
    #        = (1 - t)^3/6
    #     w2 = 2/3 - t^2 + t^3/2
    #        = 2/3 + (t/2 - 1)*t^2
    #     w3 = 1/6 + t/2 + t^2/2 - t^3/2
    #        = 1/6 + (t + t^2 - t^3)/2
    #        = 1/6 - ((t - 1)*t - 1)*t/2
    #        = 4/6 - (1 - t)^2*(t + 1)/2
    #     w4 = t^3/6
    #
    # Horner's scheme takes 6 operations per cubic polynomial, 24 operations
    # for the 4 weights.  Precomputing the powers of t, t^2 and t^3, takes 2
    # operations, then 6 operations per cubic polynomial are needed.
    #
    # Using factorizations, I manage to only use 15 operations.
    const h = frac(T,1,2)
    const p = frac(T,2,3)
    const q = frac(T,1,6)
    r = 1 - t
    r2 = r*r
    t2 = t*t
    w1 = q*r2*r
    w2 = p + (h*t - 1)*t2
    w3 = p - h*r2*(t + 1)
    w4 = q*t2*t
    return w1, w2, w3, w4
end

(::CubicSpline′{T,B})(x::T) where {T,B} =
    -2 < x < 2 ? (
        x < -1 ? frac(T,1,2)*square(x + 2) :
        x <  0 ? frac(T,-3,2)*x*(x + frac(T,4,3)) :
        x <  1 ? frac(T,+3,2)*x*(x - frac(T,4,3)) :
        frac(T,-1,2)*square(x - 2)
    ) : zero(T)

@inline function getweights(ker::CubicSpline′{T,B},
                            t::T) where {T,B}
    return (frac(T,-1,2)*(t - 1)^2,
            frac(T, 3,2)*(t - frac(T,4,3))*t,
            frac(T, 1,2) + (1 - frac(T,3,2)*t)*t,
            frac(T, 1,2)*t^2)
end

#------------------------------------------------------------------------------
# Catmull-Rom kernel is a special case of Mitchell & Netravali kernel.

struct CatmullRomSpline{T,B} <: Kernel{T,4,B}; end

iscardinal(::Union{K,Type{K}}) where {K<:CatmullRomSpline} = true

isnormalized(::Union{K,Type{K}}) where {K<:CatmullRomSpline} = true

Base.summary(::CatmullRomSpline) = "CatmullRomSpline()"

function (::CatmullRomSpline{T,B})(x::T) where {T<:AbstractFloat,B}
    a = abs(x)
    return (a ≥ T(2) ? T(0) :
            a ≤ T(1) ? (frac(T,3,2)*a - frac(T,5,2))*a*a + T(1) :
            ((frac(T,5,2) - frac(T,1,2)*a)*a - T(4))*a + T(2))
end

@inline function getweights(::CatmullRomSpline{T,B},
                            t::T) where {T<:AbstractFloat,B}
    # 10 operations
    s = T(1) - t
    q = T(-1/2)*t*s
    w1 = q*s
    w4 = q*t
    r = w4 - w1
    w2 = s - w1 + r
    w3 = t - w4 - r
    return (w1, w2, w3, w4)
end

#------------------------------------------------------------------------------
"""
```julia
CardinalCubicSpline([T=Float64,] c, B=Flat) -> ker
```

yields a cardinal cubic spline interpolation kernel for floating-point type `T`
tension parameter `c` and boundary conditions `B`.  The slope at `x = ±1` is
`∓(1 - c)/2`.  Usually `c ≤ 1`, choosing `c = 0` yields a Catmull-Rom spline,
`c = 1` yields all zero tangents, `c = -1` yields a truncated approximation of
a cardinal sine, `c = -1/2` yields an interpolating cubic spline with
continuous second derivatives (inside its support).

"""
struct CardinalCubicSpline{T,B} <: Kernel{T,4,B}
    c::T
    p::T
    q::T

    function (::Type{CardinalCubicSpline{T,B}})(c_::Real) where {T,B}
        c = convert(T, c_)
        new{T,B}(c, (c - 1)/2, (c + 1)/2)
    end
end

function CardinalCubicSpline(::Type{T}, c::Real,
                             ::Type{B} = Flat) where {T<:AbstractFloat,
                                                      B<:Boundaries}
    CardinalCubicSpline{T,B}(c)
end

CardinalCubicSpline(c::Real, ::Type{B} = Flat) where {B<:Boundaries} =
    CardinalCubicSpline(Float64, c, B)

iscardinal(::Union{K,Type{K}}) where {K<:CardinalCubicSpline} = true

isnormalized(::Union{K,Type{K}}) where {K<:CardinalCubicSpline} = true

Base.show(io::IO, ker::CardinalCubicSpline) =
    print(io, "CardinalCubicSpline(", @sprintf("%.1f", ker.c), ")")

#Base.summary(ker::CardinalCubicSpline) =
#    @sprintf("CardinalCubicSpline(%.1f)", ker.p + ker.q)

function convert(::Type{CardinalCubicSpline{T,B}},
                 ker::CardinalCubicSpline) where {T<:AbstractFloat,
                                                  B<:Boundaries}
    CardinalCubicSpline(T, ker.c, B)
end

function (ker::CardinalCubicSpline{T,B})(x::T) where {T<:AbstractFloat,B}
    a = abs(x)
    return (a ≥ 2 ? zero(T) :
            a ≥ 1 ? ker.p*(a - 1)*square(2 - a) :
            ((ker.q*a + a)*a - a - 1)*(a - 1))
end

@inline function getweights(ker::CardinalCubicSpline{T,B},
                            t::T) where {T<:AbstractFloat,B}
    p, q = ker.p, ker.q
    # Computation of:
    #     w1 = p t u²
    #     w2 = u + t u² - q t² u
    #     w3 = t + t² u - q t u²
    #     w4 = p t² u
    # with u = 1 - t in 13 operations.
    u = 1 - t
    tu = t*u
    ptu = p*tu
    return (ptu*u,
            (u - q*t)*tu + u,
            (t - q*u)*tu + t,
            ptu*t)
end

# Prime = ′    \prime + [tab]
# Second = ″
# Third = ‴

# First derivative of the cardinal cubic spline.

struct CardinalCubicSpline′{T,B} <: Kernel{T,4,B}
    c::T
    p::T
    q::T
    r::T
    s::T

    function (::Type{CardinalCubicSpline′{T,B}})(c_::Real) where {T,B}
        c = convert(T, c_)
        t = 3c + 9
        return new{T,B}(c,
                        (3c - 3)/2,
                        t/2,
                        (2c + 10)/t,
                        (c - 1)/t)
    end
end

function CardinalCubicSpline′(::Type{T}, c::Real,
                              ::Type{B} = Flat) where {T<:AbstractFloat,
                                                       B<:Boundaries}
    CardinalCubicSpline′{T,B}(c)
end

CardinalCubicSpline′(c::Real, ::Type{B} = Flat) where {B<:Boundaries} =
    CardinalCubicSpline′(Float64, c, B)

(ker::CardinalCubicSpline′{T,B})(x::T) where {T<:AbstractFloat,B} =
    x < 0 ? (
        x ≤ -2 ? zero(T) :
        x < -1 ? -(x + 2)*(x + frac(T,4,3))*ker.p :
        -(x + ker.r)*x*ker.q
    ) : (
        x ≥ 2 ? zero(T) :
        x > 1 ? (x - 2)*(x - frac(T,4,3))*ker.p :
        (x - ker.r)*x*ker.q
    )

iscardinal(ker::CardinalCubicSpline′) = (ker.c == 1)

isnormalized(::CardinalCubicSpline′) = false

Base.show(io::IO, ker::CardinalCubicSpline′) =
    print(io, "CardinalCubicSpline′(", @sprintf("%.1f", ker.c), ")")

Base.ctranspose(ker::CardinalCubicSpline{T,B}) where {T,B} =
    CardinalCubicSpline′{T,B}(ker.c)

function convert(::Type{CardinalCubicSpline′{T,B}},
                 ker::CardinalCubicSpline′) where {T<:AbstractFloat,
                                                   B<:Boundaries}
    CardinalCubicSpline′(T, ker.c, B)
end

@inline function getweights(ker::CardinalCubicSpline′{T,B},
                            t::T) where {T<:AbstractFloat,B}

    # Computation of:
    #     w1 = p*(t - 1)*(t - 1/3)
    #     w2 = q*(t - r)*t
    #     w3 = q*(t - 1)*(s - t)
    #     w4 = p*t*(2/3 - t)
    # in 13 operations.
    u = t - 1
    return (ker.p*u*(t - frac(T,1,3)),
            ker.q*(t - ker.r)*t,
            ker.q*u*(ker.s - t),
            ker.p*t*(frac(T,2,3) - t))
end

#------------------------------------------------------------------------------
"""
# Mitchell & Netravali Kernels

```julia
MitchellNetravaliSpline([T=Float64,] [b=1/3, c=1/3,] B=Flat) -> ker
```

yields an interpolation kernel of the Mitchell & Netravali family of kernels
for floating-point type `T`, parameters `b` and `c` and boundary conditions
`B`.

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

* Mitchell & Netravali, "*Reconstruction Filters in Computer Graphics*",
  in Computer Graphics, Vol. 22, Num. 4 (1988).
  http://www.cs.utexas.edu/users/fussell/courses/cs384g/lectures/mitchell/Mitchell.pdf.

"""
struct MitchellNetravaliSpline{T,B} <: Kernel{T,4,B}
    b ::T
    c ::T
    p0::T
    p2::T
    p3::T
    q0::T
    q1::T
    q2::T
    q3::T
    function (::Type{MitchellNetravaliSpline{T,B}})(b::Real,
                                                    c::Real) where {T,B}
        new{T,B}(
            b, c,
            (   6 -  2*b       )/6,
            ( -18 + 12*b +  6*c)/6,
            (  12 -  9*b -  6*c)/6,
            (        8*b + 24*c)/6,
            (     - 12*b - 48*c)/6,
            (        6*b + 30*c)/6,
            (     -    b -  6*c)/6)
    end
end

function MitchellNetravaliSpline(::Type{T}, b::Real, c::Real,
                                 ::Type{B} = Flat) where {T<:AbstractFloat,
                                                          B<:Boundaries}
    MitchellNetravaliSpline{T,B}(b, c)
end

function MitchellNetravaliSpline(b::Real, c::Real,
                                 ::Type{B} = Flat) where {B<:Boundaries}
    MitchellNetravaliSpline(Float64, b, c, B)
end

# Create Mitchell-Netravali kernel with default "good" parameters.
function MitchellNetravaliSpline(::Type{T} = Float64,
                                 ::Type{B} = Flat) where {T<:AbstractFloat,
                                                          B<:Boundaries}
    MitchellNetravaliSpline{T,B}(frac(T,1,3), frac(T,1,3))
end

iscardinal(ker::MitchellNetravaliSpline{T,B}) where {T<:AbstractFloat,B} =
    (ker.b == T(0))

isnormalized(::Union{K,Type{K}}) where {K<:MitchellNetravaliSpline} = true

Base.summary(ker::MitchellNetravaliSpline{T,B}) where {T,B} =
    @sprintf("MitchellNetravaliSpline(%.1f,%.1f)", ker.b, ker.c)

function convert(::Type{MitchellNetravaliSpline{T,B}},
                 ker::MitchellNetravaliSpline) where {T<:AbstractFloat,
                                                      B<:Boundaries}
    MitchellNetravaliSpline(T, ker.b, ker.c, B)
end

@inline _p(ker::MitchellNetravaliSpline{T,B}, x::T) where {T,B} =
    (ker.p3*x + ker.p2)*x*x + ker.p0

@inline _q(ker::MitchellNetravaliSpline{T,B}, x::T) where {T,B} =
    ((ker.q3*x + ker.q2)*x + ker.q1)*x + ker.q0

function (ker::MitchellNetravaliSpline{T,B})(x::T) where {T<:AbstractFloat,B}
    a = abs(x)
    return (a ≥ T(2) ? T(0) : a ≤ T(1) ? _p(ker, a) : _q(ker, a))
end

@inline function getweights(ker::MitchellNetravaliSpline{T,B},
                            t::T) where {T<:AbstractFloat,B}
    return (_q(ker, t + T(1)),
            _p(ker, t),
            _p(ker, T(1) - t),
            _q(ker, T(2) - t))
end

#------------------------------------------------------------------------------
"""
# Keys cardinal kernels

```julia
KeysSpline([T=Float64,] a, B=Flat) -> ker
```

yields an interpolation kernel of the Keys family of cardinal kernels for
floating-point type `T`, parameter `a` and boundary conditions `B`.

These kernels are piecewise normalized cardinal cubic spline which depend on
one parameter `a`.

Reference:

* Keys, Robert, G., "Cubic Convolution Interpolation for Digital Image
  Processing", IEEE Trans. Acoustics, Speech, and Signal Processing,
  Vol. ASSP-29, No. 6, December 1981, pp. 1153-1160.

"""
struct KeysSpline{T,B} <: Kernel{T,4,B}
    a ::T
    p0::T
    p2::T
    p3::T
    q0::T
    q1::T
    q2::T
    q3::T
    function (::Type{KeysSpline{T,B}})(a::Real) where {T,B}
        new{T,B}(a, 1, -a - 3, a + 2, -4*a, 8*a, -5*a, a)
    end
end

function KeysSpline(::Type{T}, a::Real,
                    ::Type{B} = Flat) where {T<:AbstractFloat, B<:Boundaries}
    KeysSpline{T,B}(a)
end

KeysSpline(a::Real, ::Type{B} = Flat) where {B<:Boundaries} =
    KeysSpline(Float64, a, B)

iscardinal(::Union{K,Type{K}}) where {K<:KeysSpline} = true
isnormalized(::Union{K,Type{K}}) where {K<:KeysSpline} = true
Base.summary(ker::KeysSpline) = @sprintf("KeysSpline(%.1f)", ker.a)

function convert(::Type{KeysSpline{T,B}},
                 ker::KeysSpline) where {T<:AbstractFloat, B<:Boundaries}
    KeysSpline(T, ker.a, B)
end

@inline _p(ker::KeysSpline{T,B}, x::T) where {T,B} =
    (ker.p3*x + ker.p2)*x*x + ker.p0

@inline _q(ker::KeysSpline{T,B}, x::T) where {T,B} =
    ((ker.q3*x + ker.q2)*x + ker.q1)*x + ker.q0

function (ker::KeysSpline{T,B})(x::T) where {T<:AbstractFloat,B}
    a = abs(x)
    return (a ≥ T(2) ? T(0) : a ≤ T(1) ? _p(ker, a) : _q(ker, a))
end

@inline function getweights(ker::KeysSpline{T,B},
                            t::T) where {T<:AbstractFloat,B}
    return (_q(ker, t + T(1)),
            _p(ker, t),
            _p(ker, T(1) - t),
            _q(ker, T(2) - t))
end

#------------------------------------------------------------------------------
"""
# Lanczos Resampling Kernel

```julia
LanczosKernel([T=Float64,] S, B=Flat)
```

yields a Lanczos kernel of support size `S` (which must be even), for
floating-point type `T` and boundary conditions `B`..

The Lanczos kernels doe not have the partition of unity property.  However,
Lanczos kernels tend to be normalized for large support size.

See also: [link](https://en.wikipedia.org/wiki/Lanczos_resampling).

"""
struct LanczosKernel{T,S,B} <: Kernel{T,S,B}
    a::T   # 1/2 support
    b::T   # a/pi^2
    c::T   # pi/a
    function (::Type{LanczosKernel{T,S,B}})() where {T,S,B}
        @assert typeof(S) == Int && S > 0 && iseven(S)
        a = S/2
        new{T,S,B}(a, a/pi^2, pi/a)
    end
end

function LanczosKernel(::Type{T}, s::Integer,
                       ::Type{B} = Flat) where {T<:AbstractFloat,
                                                B<:Boundaries}
    LanczosKernel{T,Int(s),B}()
end

function LanczosKernel(s::Integer, ::Type{B} = Flat) where {B<:Boundaries}
    LanczosKernel{Float64,Int(s),B}()
end

iscardinal(::Union{K,Type{K}}) where {K<:LanczosKernel} = true
isnormalized(::Union{K,Type{K}}) where {K<:LanczosKernel} = false
Base.summary(::LanczosKernel{T,S,B}) where {T,S,B} = "LanczosKernel($S)"

# `convert` should give something which is almost equivalent, so here we
# enforce the same support size.
function convert(::Type{LanczosKernel{T,S,B}},
                 ::LanczosKernel{<:AbstractFloat,S,<:Boundaries}
                 ) where {T<:AbstractFloat,S,B<:Boundaries}
    LanczosKernel{T,S,B}()
end

# Expression for non-zero argument in the range (-S/2,S/2).
@inline _p(ker::LanczosKernel{T,S,B}, x::T) where {T,S,B} =
    ker.b*sin(pi*x)*sin(ker.c*x)/(x*x)

(ker::LanczosKernel{T,S,B})(x::T) where {T,S,B} =
    (abs(x) ≥ ker.a ? T(0) : x == T(0) ? T(1) : _p(ker, x))

@generated function getweights(ker::LanczosKernel{T,S,B}, t::T) where {T,S,B}
    c = (S >> 1) # central index
    W = [Symbol("w",i) for i in 1:S] # all weights
    Expr(:block,
         Expr(:meta, :inline),
         Expr(:local, [:($w::T) for w in W]...),
         Expr(:if, :(t == zero(T)),
              Expr(:block, [:($(W[i]) = $(i == c ? 1 : 0)) for i in 1:S]...),
              Expr(:block,
                   [:($(W[i]) = _p(ker, t + T($(c - i)))) for i in 1:c-1]...,
                   :($(W[c]) = _p(ker, t)),
                   [:($(W[i]) = _p(ker, t - T($(i - c)))) for i in c+1:S]...)),
         Expr(:return, Expr(:tuple, W...)))
end

#------------------------------------------------------------------------------

Base.show(io::IO, ::MIME"text/plain", ker::Kernel) = show(io, ker)

# Provide methods for parameter-less kernels.
for K in (:RectangularSpline, :RectangularSpline′,
          :LinearSpline, :LinearSpline′,
          :QuadraticSpline, :QuadraticSpline′,
          :CubicSpline, :CubicSpline′,
          :CatmullRomSpline)
    @eval begin

        # Constructors.
        function $K(::Type{T} = Float64, ::Type{B} = Flat
                    ) where {T<:AbstractFloat,B<:Boundaries}
            $K{T,B}()
        end

        function $K(::Type{B}, ::Type{T} = Float64
                    ) where {T<:AbstractFloat,B<:Boundaries}
            $K{T,B}()
        end

        # Conversion to different types.
        function convert(::Type{$K{T,B}}, ::$K
                         ) where {T<:AbstractFloat,B<:Boundaries}
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
    #     (ker::$K{T,B})(x::Real) where {T<:AbstractFloat,B<:Boundaries} =
    #         ker(T(x))
    #
    # leads to ambiguities, the following is ugly but works...
    for T in subtypes(AbstractFloat), R in (subtypes(AbstractFloat)..., Integer)
        if R != T
            if K <: LanczosKernel
                @eval @inline (ker::$K{$T,S,B})(x::$R) where {S,B<:Boundaries} =
                    ker($T(x))
            else
                @eval @inline (ker::$K{$T,B})(x::$R) where {B<:Boundaries} =
                    ker($T(x))
            end
        end
    end

    # Change type.
    for R in (:Float16, :Float32, :Float64)
        if K <: LanczosKernel
            @eval Base.$R(ker::$K{T,S,B}) where {T,S,B} =
                convert($K{$R,S,B}, ker)
        else
            @eval Base.$R(ker::$K{T,B}) where {T,B} =
                convert($K{$R,B}, ker)
        end
    end

    # Change boundary conditions.
    for C in (:Flat, :SafeFlat)
        if K <: LanczosKernel
            @eval $C(ker::$K{T,S,B}) where {T,S,B} =
                convert($K{T,S,$C}, ker)
        else
            @eval $C(ker::$K{T,B}) where {T,B} =
                convert($K{T,$C}, ker)
        end
    end

    # Calling the kernel on an array.  FIXME: should be deprecated!
    if K <: LanczosKernel
        @eval function (ker::$K{T,S,B})(A::AbstractArray
                                        ) where {T<:AbstractFloat,S,
                                                 B<:Boundaries}
            map((x) -> ker(x), A)
        end
    else
        @eval function (ker::$K{T,B})(A::AbstractArray
                                      ) where {T<:AbstractFloat,B<:Boundaries}
            map((x) -> ker(x), A)
        end
    end

    # Calling the kernel as a function to convert to another floating-point
    # type and/or other boundary conditions.
    if K <: LanczosKernel
        @eval begin
            (ker::$K{T,S,B})(::Type{newT}, ::Type{newB}=B) where {
                T, S, B, newT<:AbstractFloat, newB<:Boundaries
            } = convert($K{newT,S,newB}, ker)

            (ker::$K{T,S,B})(::Type{newB}, ::Type{newT}=T) where {
                T, S, B, newT<:AbstractFloat, newB<:Boundaries
            } = convert($K{newT,S,newB}, ker)

            Base.convert(::Type{$K{T,S,B}}, ker::$K{T,S,B}) where {T,S,B} = ker
        end
    else
        @eval begin
            (ker::$K{T,B})(::Type{newT}, ::Type{newB}=B) where {
                T, B, newT<:AbstractFloat, newB<:Boundaries
            } = convert($K{newT,newB}, ker)

            (ker::$K{T,B})(::Type{newB}, ::Type{newT}=T) where {
                T, B, newT<:AbstractFloat, newB<:Boundaries
            } = convert($K{newT,newB}, ker)

            Base.convert(::Type{$K{T,B}}, ker::$K{T,B}) where {T,B} = ker
        end
    end

end

end # module

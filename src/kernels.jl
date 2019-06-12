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
# Copyright (C) 2016-2019, Éric Thiébaut.
#

module Kernels

import Base: convert

export
    Boundaries,
    CardinalCubicSpline,
    CardinalCubicSplinePrime,
    CatmullRomSpline,
    CatmullRomSplinePrime,
    CubicSpline,
    CubicSplinePrime,
    Flat,
    Kernel,
    KeysSpline,
    KeysSplinePrime,
    LanczosKernel,
    LanczosKernelPrime,
    LinearSpline,
    LinearSplinePrime,
    MitchellNetravaliSpline,
    QuadraticSpline,
    QuadraticSplinePrime,
    RectangularSpline,
    RectangularSplinePrime,
    SafeFlat,
    boundaries,
    getweights,
    iscardinal,
    isnormalized,
    brief

# Deal with compatibility issues.
using Compat
using Compat.Printf
using Compat.InteractiveUtils
@static if isdefined(Base, :adjoint)
    import Base: adjoint
else
    import Base: ctranspose
    const adjoint = ctranspose
end

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
    (T(num)/T(den))

@inline square(x) = x*x
@inline cube(x) = x*x*x

@inline signabs(x::Real) = ifelse(x < 0, (oftype(one(x),-1), -x), (one(x), x))

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

```julia
eltype(ker) -> T
```

yields the floating-point type for calculations,

```julia
length(ker) -> S
size(ker)   -> (S,)
```

yield the size the support of `ker` which is also the number of neighbors
involved in an interpolation by this kernel,

```julia
boundaries(ker) -> B
```

yields the type of the boundary conditions applied for extrapolation; finally:

```julia
getweights(ker, t) -> w1, w2, ..., wS
```

yields the `S` interpolation weights for offset `t`

```
t = x - floor(x)        if s is even
    x - round(x)        if s is odd
```

 `t ∈ [0,1]` if `S` is even or
for offset `t ∈ [-1/2,+1/2]` if `S` is odd.

"""
abstract type Kernel{T<:AbstractFloat,S,B<:Boundaries} <: Function end

Base.eltype(::Kernel{T,S,B})         where {T,S,B} = T
Base.eltype(::Type{<:Kernel{T,S,B}}) where {T,S,B} = T
Base.length(::Kernel{T,S,B})         where {T,S,B} = S
Base.length(::Type{<:Kernel{T,S,B}}) where {T,S,B} = S
Base.size(::Kernel{T,S,B})           where {T,S,B} = (S,)
Base.size(::Type{<:Kernel{T,S,B}})   where {T,S,B} = (S,)

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

A rectangular spline instance is created by:

```julia
RectangularSpline([T=Float64,] B=Flat)
```

Its derivative is created by:

```julia
RectangularSplinePrime([T=Float64,] B=Flat)
```

"""
struct RectangularSpline{T,B} <: Kernel{T,1,B}; end
struct RectangularSplinePrime{T,B} <: Kernel{T,1,B}; end
@doc @doc(RectangularSpline) RectangularSplinePrime

iscardinal(::Union{K,Type{K}}) where {K<:RectangularSpline} = true
iscardinal(::Union{K,Type{K}}) where {K<:RectangularSplinePrime} = false

isnormalized(::Union{K,Type{K}}) where {K<:RectangularSpline} = true
isnormalized(::Union{K,Type{K}}) where {K<:RectangularSplinePrime} = false

Base.show(io::IO, ::RectangularSpline) = print(io, "RectangularSpline()")
Base.show(io::IO, ::RectangularSplinePrime) = print(io, "RectangularSplinePrime()")

(::RectangularSpline{T,B})(x::T) where {T,B} =
    frac(T,-1,2) ≤ x < frac(T,1,2) ? one(T) : zero(T)

@inline getweights(::RectangularSpline{T,B}, t::T) where {T,B} = one(T)

(::RectangularSplinePrime{T,B})(x::T) where {T,B} = zero(T)

@inline getweights(::RectangularSplinePrime{T,B}, t::T) where {T,B} = zero(T)

#------------------------------------------------------------------------------
"""
# Linear Spline

The linear spline (also known as triangle kernel or Bartlett window or Fejér
window) is the 2nd order (linear) B-spline.

A linear spline instance is created by:

```julia
LinearSpline([T=Float64,] B=Flat)
```

Its derivative is created by:

```julia
LinearSplinePrime([T=Float64,] B=Flat)
```

"""
struct LinearSpline{T,B} <: Kernel{T,2,B}; end
struct LinearSplinePrime{T,B} <: Kernel{T,2,B}; end
@doc @doc(LinearSpline) LinearSplinePrime

iscardinal(::Union{K,Type{K}}) where {K<:LinearSpline} = true
iscardinal(::Union{K,Type{K}}) where {K<:LinearSplinePrime} = false

isnormalized(::Union{K,Type{K}}) where {K<:LinearSpline} = true
isnormalized(::Union{K,Type{K}}) where {K<:LinearSplinePrime} = false

Base.show(io::IO, ::LinearSpline) = print(io, "LinearSpline()")
Base.show(io::IO, ::LinearSplinePrime) = print(io, "LinearSplinePrime()")

(::LinearSpline{T,B})(x::T) where {T<:AbstractFloat,B} =
    (a = abs(x); a < 1 ? 1 - a : zero(T))

@inline getweights(::LinearSpline{T,B}, t::T) where {T<:AbstractFloat,B} =
    (1 - t, t)

# The derivative of the linear B-spline must be non-symmetric for tests to
# succeed.  In particular we want that interpolating with the derivative of the
# linear B-spline amounts to taking the finite difference when 0 ≤ t < 1.
# This implies that f'(x) = 1 for x ∈ [-1,0), f'(x) = -1 for x ∈ [0,1), and
# f'(x) = 0 elsewhere.
(::LinearSplinePrime{T,B})(x::T) where {T<:AbstractFloat,B} =
    -1 ≤ x < 1 ? (x < 0 ? one(T) : -one(T)) : zero(T)

@inline getweights(::LinearSplinePrime{T,B}, t::T) where {T<:AbstractFloat,B} =
    (-one(T), one(T))

#------------------------------------------------------------------------------
"""
# Quadratic Spline

The quadratic spline is the 3rd order (quadratic) B-spline.

A quadratic spline instance is created by:

```julia
QuadraticSpline([T=Float64,] B=Flat)
```

Its derivative is created by:

```julia
QuadraticSplinePrime([T=Float64,] B=Flat)
```

"""
struct QuadraticSpline{T,B} <: Kernel{T,3,B}; end
struct QuadraticSplinePrime{T,B} <: Kernel{T,3,B}; end
@doc @doc(QuadraticSpline) QuadraticSplinePrime

iscardinal(::Union{K,Type{K}}) where {K<:QuadraticSpline} = false
iscardinal(::Union{K,Type{K}}) where {K<:QuadraticSplinePrime} = false

isnormalized(::Union{K,Type{K}}) where {K<:QuadraticSpline} = true
isnormalized(::Union{K,Type{K}}) where {K<:QuadraticSplinePrime} = false

Base.show(io::IO, ::QuadraticSpline) = print(io, "QuadraticSpline()")
Base.show(io::IO, ::QuadraticSplinePrime) = print(io, "QuadraticSplinePrime()")

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
        h = frac(T,1,2)
        return (h*square(h - t), frac(T,3,4) - t*t, h*square(h + t))
    end
else
    # Same result, but with 7 operations.
    @inline function getweights(::QuadraticSpline{T,B},
                                t::T) where {T<:AbstractFloat,B}
        # c1 = 1/sqrt(8)
        c1 = T(0.35355339059327376220042218105242451964241796884424)
        # c2 = 2/sqrt(8)
        c2 = T(0.70710678118654752440084436210484903928483593768847)
        # c2 = 3/4
        c3 = frac(T,3,4)
        c2t = c2*t
        q1 = c1 - c2t
        q3 = c1 + c2t
        return (q1*q1, c3 - t*t, q3*q3)
    end
end

(::QuadraticSplinePrime{T,B})(x::T) where {T,B} =
    frac(T,-3,2) < x < frac(T,3,2) ? (
        x < frac(T,-1,2) ? x + frac(T,3,2) :
        x ≤ frac(T,1,2) ? -2x : x - frac(T,3,2)
    ) : zero(T)

@inline function getweights(::QuadraticSplinePrime{T,B},
                            t::T) where {T<:AbstractFloat,B}
    h = frac(T,1,2)
    return (t - h, -2t, t + h)
end

#------------------------------------------------------------------------------
"""
# Cubic Spline

    CubicSpline([T=Float64,] B=Flat)

where `T <: AbstractFloat` and `B <: Boundaries` yields a cubic spline kernel
which operates with floating-point type `T` and use boundary conditions `B`
(any of which can be omitted and their order is irrelevant).

The 4th order (cubic) B-spline kernel is also known as Parzen window or de la
Vallée Poussin window.

Its derivative is given by:

```julia
CubicSplinePrime([T=Float64,] B=Flat)
```

"""
struct CubicSpline{T,B} <: Kernel{T,4,B}; end
struct CubicSplinePrime{T,B} <: Kernel{T,4,B}; end
@doc @doc(CubicSpline) CubicSplinePrime

iscardinal(::Union{K,Type{K}}) where {K<:CubicSpline} = false
iscardinal(::Union{K,Type{K}}) where {K<:CubicSplinePrime} = false

isnormalized(::Union{K,Type{K}}) where {K<:CubicSpline} = true
isnormalized(::Union{K,Type{K}}) where {K<:CubicSplinePrime} = false

Base.show(io::IO, ::CubicSpline) = print(io, "CubicSpline()")
Base.show(io::IO, ::CubicSplinePrime) = print(io, "CubicSplinePrime()")

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
    h = frac(T,1,2)
    p = frac(T,2,3)
    q = frac(T,1,6)
    r = 1 - t
    r2 = r*r
    t2 = t*t
    w1 = q*r2*r
    w2 = p + (h*t - 1)*t2
    w3 = p - h*r2*(t + 1)
    w4 = q*t2*t
    return w1, w2, w3, w4
end

(::CubicSplinePrime{T,B})(x::T) where {T,B} =
    -2 < x < 2 ? (
        x < -1 ? frac(T,1,2)*square(x + 2) :
        x <  0 ? frac(T,-3,2)*x*(x + frac(T,4,3)) :
        x <  1 ? frac(T,+3,2)*x*(x - frac(T,4,3)) :
        frac(T,-1,2)*square(x - 2)
    ) : zero(T)

@inline function getweights(ker::CubicSplinePrime{T,B},
                            t::T) where {T,B}
    return (frac(T,-1,2)*(t - 1)^2,
            frac(T, 3,2)*(t - frac(T,4,3))*t,
            frac(T, 1,2) + (1 - frac(T,3,2)*t)*t,
            frac(T, 1,2)*t^2)
end

#------------------------------------------------------------------------------
# Catmull-Rom kernel is a special case of Mitchell & Netravali kernel.

"""
```julia
CatmullRomSpline([T=Float64,] B=Flat) -> ker
```

yields a Catmull-Rom interpolation kernel for floating-point type `T` and
boundary conditions `B`.

Catmull-Rom interpolation kernel is a piecewise cardinal cubic spline defined
by:

```
ker(x) = ((3/2)*|x| - (5/2))*x^2 + 1             if |x| ≤ 1
         (((5/2) - (1/2)*|x|)*|x| - 4)*|x| + 2   if 1 ≤ |x| ≤ 2
         0                                       if |x| ≥ 2
```

It derivative is given by:

```julia
CatmullRomSplinePrime([T=Float64,] B=Flat) -> ker′
```

with:

```
ker′(x) = ((9/2)*|x| - 5)*x                      if a = |x| ≤ 1
          (5 - (3/2)*|x|)*x - 4*sign(x)          if 1 ≤ |x| ≤ 2
          0                                      if |x| ≥ 2
```

"""
struct CatmullRomSpline{T,B} <: Kernel{T,4,B}; end
struct CatmullRomSplinePrime{T,B} <: Kernel{T,4,B}; end
@doc @doc(CatmullRomSpline) CatmullRomSplinePrime

iscardinal(::Union{K,Type{K}}) where {K<:CatmullRomSpline} = true
iscardinal(::Union{K,Type{K}}) where {K<:CatmullRomSplinePrime} = false

isnormalized(::Union{K,Type{K}}) where {K<:CatmullRomSpline} = true
isnormalized(::Union{K,Type{K}}) where {K<:CatmullRomSplinePrime} = false

Base.summary(::CatmullRomSpline) = "CatmullRomSpline()"
Base.summary(::CatmullRomSplinePrime) = "CatmullRomSplinePrime()"

@inline function (::CatmullRomSpline{T,B})(x::T) where {T<:AbstractFloat,B}
    a = abs(x)
    return (a ≥ 2 ? zero(T) :
            a ≤ 1 ? (frac(T,3,2)*a - frac(T,5,2))*a*a + T(1) :
            ((frac(T,5,2) - frac(T,1,2)*a)*a - T(4))*a + T(2))
end

@inline function (::CatmullRomSplinePrime{T,B})(x::T) where {T<:AbstractFloat,B}
    return (x < 0 ?
            (x ≤ -2 ? zero(T) :
             x ≥ -1 ? (frac(T,-9,2)*x - T(5))*x :
             (T(5) + frac(T,3,2)*x)*x + T(4)) :
            (x ≥ 2 ? zero(T) :
             x ≤ 1 ? (frac(T,9,2)*x - T(5))*x :
             (T(5) - frac(T,3,2)*x)*x - T(4)))
end

@inline function getweights(::CatmullRomSpline{T,B},
                            t::T) where {T<:AbstractFloat,B}
    # 10 operations
    s = 1 - t
    q = frac(T,-1,2)*t*s
    w1 = q*s
    w4 = q*t
    r = w4 - w1
    w2 = s - w1 + r
    w3 = t - w4 - r
    return (w1, w2, w3, w4)
end

@inline function getweights(::CatmullRomSplinePrime{T,B},
                            t::T) where {T<:AbstractFloat,B}
    # Weights (18 operations):
    #   w1 = -(3/2)*t^2 + 2*t - (1/2);
    #   w2 =  (9/2)*t^2 - 5*t;
    #   w3 = -(9/2)*t^2 + 4*t + (1/2);
    #   w4 =  (3/2)*t^2 - t;
    #
    #   w4 = ((3/2)*t - 1)*t;
    #   w1 = t - w4 - (1/2);
    #   w2 = 3*w4 - 2*t;
    #   w3 = t - 3*w4 + (1/2);
    #
    # 11 operations:
    w4 = frac(T,3,2)*t*t - t;
    w1 = t - w4 - frac(T,1,2);
    w2 = T(3)*w4 - 2*t;
    w3 = t - T(3)*w4 + frac(T,1,2);
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

Its derivative is given by:

```julia
CardinalCubicSplinePrime([T=Float64,] c, B=Flat) -> ker
```

"""
struct CardinalCubicSpline{T,B} <: Kernel{T,4,B}
    c::T
    p::T
    q::T

    function CardinalCubicSpline{T,B}(c_::Real) where {T,B}
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

# First derivative of the cardinal cubic spline.

struct CardinalCubicSplinePrime{T,B} <: Kernel{T,4,B}
    c::T
    p::T
    q::T
    r::T
    s::T

    function CardinalCubicSplinePrime{T,B}(c_::Real) where {T,B}
        c = convert(T, c_)
        t = 3c + 9
        return new{T,B}(c,
                        (3c - 3)/2,
                        t/2,
                        (2c + 10)/t,
                        (c - 1)/t)
    end
end

@doc @doc(CardinalCubicSpline) CardinalCubicSplinePrime

function CardinalCubicSplinePrime(::Type{T}, c::Real,
                              ::Type{B} = Flat) where {T<:AbstractFloat,
                                                       B<:Boundaries}
    CardinalCubicSplinePrime{T,B}(c)
end

CardinalCubicSplinePrime(c::Real, ::Type{B} = Flat) where {B<:Boundaries} =
    CardinalCubicSplinePrime(Float64, c, B)

(ker::CardinalCubicSplinePrime{T,B})(x::T) where {T<:AbstractFloat,B} =
    x < 0 ? (
        x ≤ -2 ? zero(T) :
        x < -1 ? -(x + 2)*(x + frac(T,4,3))*ker.p :
        -(x + ker.r)*x*ker.q
    ) : (
        x ≥ 2 ? zero(T) :
        x > 1 ? (x - 2)*(x - frac(T,4,3))*ker.p :
        (x - ker.r)*x*ker.q
    )

iscardinal(ker::CardinalCubicSplinePrime) = (ker.c == 1)

isnormalized(::CardinalCubicSplinePrime) = false

Base.show(io::IO, ker::CardinalCubicSplinePrime) =
    print(io, "CardinalCubicSplinePrime(", @sprintf("%.1f", ker.c), ")")

function convert(::Type{CardinalCubicSplinePrime{T,B}},
                 ker::CardinalCubicSplinePrime) where {T<:AbstractFloat,
                                                   B<:Boundaries}
    CardinalCubicSplinePrime(T, ker.c, B)
end

@inline function getweights(ker::CardinalCubicSplinePrime{T,B},
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
    function MitchellNetravaliSpline{T,B}(_b::Real,
                                          _c::Real) where {T,B}
        b, c = T(_b), T(_c)
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

iscardinal(ker::MitchellNetravaliSpline{T,B}) where {T,B} = (ker.b == 0)

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
    return (a ≥ 2 ? zero(T) : a ≤ 1 ? _p(ker, a) : _q(ker, a))
end

@inline function getweights(ker::MitchellNetravaliSpline{T,B},
                            t::T) where {T<:AbstractFloat,B}
    return (_q(ker, t + 1),
            _p(ker, t),
            _p(ker, 1 - t),
            _q(ker, 2 - t))
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
one parameter `a` which is the slope of the spline at abscissa 1.

Keys splines are defined by:

```
ker(x) = p(abs(x))   if abs(x) ≤ 1
         q(abs(x))   if 1 ≤ abs(x) ≤ 2
         0           if abs(x) ≥ 2
```

with:

```
p(x) = 1 - (a + 3)*x^2 + (a + 2)*x^3
q(x) = -4a + 8a*x - 5a*x^2 + a*x^3
```

Their derivatives are given by:

```julia
KeysSplinePrime([T=Float64,] a, B=Flat) -> ker′
```

defined by:

```
ker′(x) = p′(abs(x))*sign(x)   if abs(x) ≤ 1
          q′(abs(x))*sign(x)   if 1 ≤ abs(x) ≤ 2
          0                    if abs(x) ≥ 2
```

with:
```
p(x) = -2*(a + 3)*x + 3*(a + 2)*x^2
q(x) = 8a - 10a*x + 3a*x^2
```

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
    function KeysSpline{T,B}(a::Real) where {T,B}
        new{T,B}(a,
                 1, -a - 3, a + 2,
                 -4a, 8a, -5a, a)
    end
end

struct KeysSplinePrime{T,B} <: Kernel{T,4,B}
    a::T
    b::T # 2*(a + 3)
    c::T # a + 2
    p1::T
    p2::T
    q0::T
    q1::T
    q2::T
    function KeysSplinePrime{T,B}(a::Real) where {T,B}
        new{T,B}(a, 2a + 6, a + 2,
                 -2a - 6, 3a + 6,
                 8a, -10a, 3a)
    end
end

@doc @doc(KeysSpline) KeysSplinePrime

function KeysSpline(::Type{T}, a::Real,
                    ::Type{B} = Flat) where {T<:AbstractFloat, B<:Boundaries}
    KeysSpline{T,B}(a)
end

KeysSpline(a::Real, ::Type{B} = Flat) where {B<:Boundaries} =
    KeysSpline(Float64, a, B)

function KeysSplinePrime(::Type{T}, a::Real,
                    ::Type{B} = Flat) where {T<:AbstractFloat, B<:Boundaries}
    KeysSplinePrime{T,B}(a)
end

KeysSplinePrime(a::Real, ::Type{B} = Flat) where {B<:Boundaries} =
    KeysSplinePrime(Float64, a, B)

iscardinal(::Union{K,Type{K}}) where {K<:KeysSpline} = true
isnormalized(::Union{K,Type{K}}) where {K<:KeysSpline} = true
Base.summary(ker::KeysSpline) = @sprintf("KeysSpline(%.1f)", ker.a)

iscardinal(::Union{K,Type{K}}) where {K<:KeysSplinePrime} = false
isnormalized(::Union{K,Type{K}}) where {K<:KeysSplinePrime} = false
Base.summary(ker::KeysSplinePrime) = @sprintf("KeysSplinePrime(%.1f)", ker.a)

function convert(::Type{KeysSpline{T,B}},
                 ker::KeysSpline) where {T<:AbstractFloat, B<:Boundaries}
    KeysSpline(T, ker.a, B)
end

function convert(::Type{KeysSplinePrime{T,B}},
                 ker::KeysSplinePrime) where {T<:AbstractFloat, B<:Boundaries}
    KeysSplinePrime(T, ker.a, B)
end

@inline _p(ker::KeysSpline{T,B}, x::T) where {T,B} =
    (ker.p3*x + ker.p2)*x*x + ker.p0

@inline _q(ker::KeysSpline{T,B}, x::T) where {T,B} =
    ((ker.q3*x + ker.q2)*x + ker.q1)*x + ker.q0

function (ker::KeysSpline{T,B})(x::T) where {T<:AbstractFloat,B}
    a = abs(x)
    return (a ≥ 2 ? zero(T) : a ≤ 1 ? _p(ker, a) : _q(ker, a))
end

@inline _p(ker::KeysSplinePrime{T,B}, x::T, a::T) where {T,B} =
    (ker.p2*a + ker.p1)*x

@inline _q(ker::KeysSplinePrime{T,B}, x::T, s::T, a::T) where {T,B} =
    ((ker.q2*a + ker.q1)*x + ker.q0*s)

function (ker::KeysSplinePrime{T,B})(x::T) where {T<:AbstractFloat,B}
    s, a = signabs(x)
    return (a ≥ 2 ? zero(T) : a ≤ 1 ? _p(ker, x, a) : _q(ker, x, s, a))
end

@inline function getweights(ker::KeysSpline{T,B},
                            t::T) where {T<:AbstractFloat,B}
    # t ∈ [0,1), S=4
    # w1 = f(t1) = f4(t1) = q(t1)     t1 = t + 1 ∈ [1,2)
    # w2 = f(t2) = f3(t2) = p(t2)     t2 = t     ∈ [0,1)
    # w3 = f(t3) = f2(t3) = p(-t3)    t3 = t - 1 ∈ [-1,0)
    # w4 = f(t4) = f1(t4) = q(-t4)    t4 = t - 2 ∈ [-2,0)
    #
    # w1 = a*(1 - t)^2*t
    # w2 = 1 - (3 + a)*t^2 + (2 + a)*t^3
    # w3 = -t*(a*(1 - t)^2 + t*(2*t - 3))
    # w4 = a*(1 - t)*t^2
    #
    # in 12 operations (instead of 25 with the polynomials):
    a = ker.a
    r = 1 - t
    s = (ker.p3*r + 1)*t*t
    art = a*r*t
    w1 = art*r
    w4 = art*t
    w2 = 1 - s
    w3 = s - w1 - w4
    return w1, w2, w3, w4
end

@inline function getweights(ker::KeysSplinePrime{T,B},
                            t::T) where {T<:AbstractFloat,B}
    # t ∈ [0,1), S=4
    # w1 = f(t1) = f4(t1) =  q(t1)     t1 = t + 1 ∈ [1,2)
    # w2 = f(t2) = f3(t2) =  p(t2)     t2 = t     ∈ [0,1)
    # w3 = f(t3) = f2(t3) = -p(-t3)    t3 = t - 1 ∈ [-1,0)
    # w4 = f(t4) = f1(t4) = -q(-t4)    t4 = t - 2 ∈ [-2,0)
    #
    # w1 = a*(1 - t)*(1 - 3*t)
    # w2 = (3*(2 + a)*t - 2*(3 + a))*t
    # w3 = (1 - t)*(3*(2 + a)*t - a)
    # w4 = a*(2 - 3*t)*t
    #
    # in 13 operations:
    a = ker.a
    b = ker.b
    c = ker.c
    q = 3*t
    r = 1 - t
    s = c*q
    w1 = (a - a*q)*r
    w2 = (s - b)*t
    w3 = (s - a)*r
    w4 = (2 - q)*t*a
    return w1, w2, w3, w4
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

The derivative of Lanczos kernel of support size `S` is given by:

```julia
LanczosKernelPrime([T=Float64,] S, B=Flat)
```

See also: [link](https://en.wikipedia.org/wiki/Lanczos_resampling).

"""
struct LanczosKernel{T,S,B} <: Kernel{T,S,B}
    a::T   # 1/2 support
    b::T   # a/pi^2
    c::T   # pi/a
    function LanczosKernel{T,S,B}() where {T,S,B}
        @assert typeof(S) == Int && S > 0 && iseven(S)
        a = frac(T,S,2)
        new{T,S,B}(a, a/T(π)^2, T(π)/a)
    end
end

struct LanczosKernelPrime{T,S,B} <: Kernel{T,S,B}
    a::T   # 1/2 support
    c::T   # pi/a
    function LanczosKernelPrime{T,S,B}() where {T,S,B}
        @assert typeof(S) == Int && S > 0 && iseven(S)
        a = T(S)/T(2)
        new{T,S,B}(a, T(π)/a)
    end
end

function LanczosKernel(::Type{T}, s::Integer,
                       ::Type{B} = Flat) where {T<:AbstractFloat,
                                                B<:Boundaries}
    return LanczosKernel{T,Int(s),B}()
end

function LanczosKernelPrime(::Type{T}, s::Integer,
                            ::Type{B} = Flat) where {T<:AbstractFloat,
                                                     B<:Boundaries}
    return LanczosKernelPrime{T,Int(s),B}()
end

LanczosKernel(s::Integer, ::Type{B} = Flat) where {B<:Boundaries} =
    LanczosKernel{Float64,Int(s),B}()

LanczosKernelPrime(s::Integer, ::Type{B} = Flat) where {B<:Boundaries} =
    LanczosKernelPrime{Float64,Int(s),B}()

iscardinal(::Union{K,Type{K}}) where {K<:LanczosKernel} = true
isnormalized(::Union{K,Type{K}}) where {K<:LanczosKernel} = false
Base.summary(::LanczosKernel{T,S,B}) where {T,S,B} = "LanczosKernel($S)"

iscardinal(::Union{K,Type{K}}) where {K<:LanczosKernelPrime} = false
isnormalized(::Union{K,Type{K}}) where {K<:LanczosKernelPrime} = false
Base.summary(::LanczosKernelPrime{T,S,B}) where {T,S,B} = "LanczosKernelPrime($S)"

# `convert` should give something which is almost equivalent, so here we
# enforce the same support size.
function convert(::Type{LanczosKernel{T,S,B}},
                 ::LanczosKernel{<:AbstractFloat,S,<:Boundaries}
                 ) where {T<:AbstractFloat,S,B<:Boundaries}
    LanczosKernel{T,S,B}()
end

function convert(::Type{LanczosKernelPrime{T,S,B}},
                 ::LanczosKernelPrime{<:AbstractFloat,S,<:Boundaries}
                 ) where {T<:AbstractFloat,S,B<:Boundaries}
    LanczosKernelPrime{T,S,B}()
end

# Expression for non-zero argument in the range (-S/2,S/2).
@inline _p(ker::LanczosKernel{T,S,B}, x::T) where {T,S,B} =
    ker.b*sin(π*x)*sin(ker.c*x)/(x*x)

# Expression for non-zero argument in the range (-S/2,S/2).
@inline function _p(ker::LanczosKernelPrime{T,S,B}, x::T) where {T,S,B}
    x1 = π*x
    s1, c1 = sin(x1), cos(x1)
    r1 = s1/x1
    x2 = ker.c*x # π*x/a
    s2, c2 = sin(x2), cos(x2)
    r2 = s2/x2
    return (c1*r2 + c2*r1 - 2*r1*r2)/x
end

(ker::LanczosKernel{T,S,B})(x::T) where {T,S,B} =
    (abs(x) ≥ ker.a ? zero(T) : x == 0 ? one(T) : _p(ker, x))

(ker::LanczosKernelPrime{T,S,B})(x::T) where {T,S,B} =
    (abs(x) ≥ ker.a ? zero(T) : x == 0 ? one(T) : _p(ker, x))

@generated function getweights(ker::LanczosKernel{T,S,B}, t::T) where {T,S,B}
    c = (S >> 1) # central index
    W = [Symbol("w",i) for i in 1:S] # all weights
    Expr(:block,
         Expr(:meta, :inline),
         Expr(:local, [:($w::T) for w in W]...),
         Expr(:if, :(t == zero(T)),
              Expr(:block, [:($(W[i]) = $(i == c ? 1 : 0)) for i in 1:S]...),
              Expr(:block,
                   [:($(W[i]) = _p(ker, t + $(c - i))) for i in 1:c-1]...,
                   :($(W[c]) = _p(ker, t)),
                   [:($(W[i]) = _p(ker, t - $(i - c))) for i in c+1:S]...)),
         Expr(:return, Expr(:tuple, W...)))
end

@generated function getweights(ker::LanczosKernelPrime{T,S,B}, t::T) where {T,S,B}
    c = (S >> 1) # central index
    W = [Symbol("w",i) for i in 1:S] # all weights
    Expr(:block,
         #Expr(:meta, :inline),
         Expr(:local, [:($w::T) for w in W]...),
         [:($(W[i]) = ker(t + $(c - i))) for i in 1:c-1]...,
         :($(W[c]) = ker(t)),
         [:($(W[i]) = ker(t - $(i - c))) for i in c+1:S]...,
         Expr(:return, Expr(:tuple, W...)))
end

#------------------------------------------------------------------------------

# Manage to call the short version of `show` for MIME output.
Base.show(io::IO, ::MIME"text/plain", ker::Kernel) = show(io, ker)

"""
```julia
brief(ker)
```

yields the name of kernel `ker`.

""" brief

for (T, str) in (
    (:RectangularSpline, "rectangular B-spline"),
    (:RectangularSplinePrime, "derivative of rectangular B-spline"),
    (:LinearSpline, "linear B-spline"),
    (:LinearSplinePrime, "derivative of linear B-spline"),
    (:QuadraticSpline, "quadratic B-spline"),
    (:QuadraticSplinePrime, "derivative of quadratic B-spline"),
    (:CubicSpline, "cubic B-spline"),
    (:CubicSplinePrime, "derivative of cubic B-spline"),
    (:CardinalCubicSpline, "cardinal cubic spline"),
    (:CardinalCubicSplinePrime, "derivative of cardinal cubic spline"),
    (:CatmullRomSpline, "Catmull & Rom cubic spline"),
    (:CatmullRomSplinePrime, "derivative of Catmull & Rom cubic spline"),
    (:MitchellNetravaliSpline, "Mitchell & Netravali cubic spline"),
    (:KeysSpline, "Keys cubic spline"),
    (:KeysSplinePrime, "derivative of Keys cubic spline"))
    @eval brief(::$T) = $str
end

brief(::LanczosKernel{T,S,B}) where {T,S,B} =
    "Lanczos resampling kernel of size $S"

brief(::LanczosKernelPrime{T,S,B}) where {T,S,B} =
    "derivative of Lanczos resampling kernel of size $S"

# Manage to yield the derivative of (some) kernels when the notation `ker'` is
# used.
for T in (:RectangularSpline, :LinearSpline, :QuadraticSpline,
          :CubicSpline, :CatmullRomSpline)
    @eval adjoint(ker::$T{T,B}) where {T,B} = $(Symbol(T,:Prime)){T,B}()
end

adjoint(ker::CardinalCubicSpline{T,B}) where {T,B} =
    CardinalCubicSplinePrime{T,B}(ker.c)

adjoint(ker::LanczosKernel{T,S,B}) where {T,S,B} =
    LanczosKernelPrime{T,S,B}()


# Provide methods for parameter-less kernels.
for K in (:RectangularSpline, :RectangularSplinePrime,
          :LinearSpline, :LinearSplinePrime,
          :QuadraticSpline, :QuadraticSplinePrime,
          :CubicSpline, :CubicSplinePrime,
          :CatmullRomSpline, :CatmullRomSplinePrime)
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

    lanczos = (K <: LanczosKernel || K <: LanczosKernelPrime)

    # We want that calling the kernel on a different type of real argument than
    # the floting-point type of the kernel convert the argument.
    # Unfortunately, defining:
    #
    #     (ker::$K{T,B})(x::Real) where {T<:AbstractFloat,B<:Boundaries} =
    #         ker(convert(T,x))
    #
    # leads to ambiguities, the following is ugly but works...
    for T in subtypes(AbstractFloat), R in (subtypes(AbstractFloat)..., Integer)
        if R != T
            if lanczos
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
        if lanczos
            @eval Base.$R(ker::$K{T,S,B}) where {T,S,B} =
                convert($K{$R,S,B}, ker)
        else
            @eval Base.$R(ker::$K{T,B}) where {T,B} =
                convert($K{$R,B}, ker)
        end
    end

    # Change boundary conditions.
    for C in (:Flat, :SafeFlat)
        if lanczos
            @eval $C(ker::$K{T,S,B}) where {T,S,B} =
                convert($K{T,S,$C}, ker)
        else
            @eval $C(ker::$K{T,B}) where {T,B} =
                convert($K{T,$C}, ker)
        end
    end

    # Calling the kernel on an array.  FIXME: should be deprecated!
    if lanczos
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
    if lanczos
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

# Deprecations.
# Prime = ′    \prime + [tab]
# Second = ″
# Third = ‴

@deprecate          KeysSpline′          KeysSplinePrime
@deprecate         CubicSpline′         CubicSplinePrime
@deprecate        LinearSpline′        LinearSplinePrime
@deprecate       LanczosKernel′       LanczosKernelPrime
@deprecate     QuadraticSpline′     QuadraticSplinePrime
@deprecate    CatmullRomSpline′    CatmullRomSplinePrime
@deprecate CardinalCubicSpline′ CardinalCubicSplinePrime

end # module

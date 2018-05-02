#
# kernels.jl --
#
# Kernel functions used for linear filtering, windowing or linear
# interpolation.
#
#------------------------------------------------------------------------------
#
# This file is part of the LazyInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2016-2018, Éric Thiébaut.
#

module Kernels

import Base: convert

export
    Boundaries,
    CardinalCubicSpline,
    CatmullRomSpline,
    CubicSpline,
    Flat,
    Kernel,
    KeysSpline,
    LanczosKernel,
    LinearSpline,
    MitchellNetravaliSpline,
    QuadraticSpline,
    RectangularSpline,
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

iscardinal(::Union{K,Type{K}}) where {K<:RectangularSpline} = true
isnormalized(::Union{K,Type{K}}) where {K<:RectangularSpline} = true
Base.summary(::RectangularSpline) = "RectangularSpline()"

(::RectangularSpline{T,B})(x::T) where {T<:AbstractFloat,B} =
    T(-1/2) ≤ x < T(1/2) ? T(1) : T(0)

@inline getweights(::RectangularSpline{T,B}, t::T) where {T<:AbstractFloat,B} =
    one(T)

#------------------------------------------------------------------------------
"""
# Linear Spline

The linear spline (also known as triangle kernel or Bartlett window or Fejér
window) is the 2nd order (linear) B-spline.

"""
struct LinearSpline{T,B} <: Kernel{T,2,B}; end

iscardinal(::Union{K,Type{K}}) where {K<:LinearSpline} = true
isnormalized(::Union{K,Type{K}}) where {K<:LinearSpline} = true
Base.summary(::LinearSpline) = "LinearSpline()"

(::LinearSpline{T,B})(x::T) where {T<:AbstractFloat,B} =
    (a = abs(x); a < T(1) ? T(1) - a : T(0))

@inline getweights(::LinearSpline{T,B}, t::T) where {T<:AbstractFloat,B} =
    T(1) - t, t

#------------------------------------------------------------------------------
"""
# Quadratic Spline

The quadratic spline is the 3rd order (quadratic) B-spline.
"""
struct QuadraticSpline{T,B} <: Kernel{T,3,B}; end

iscardinal(::Union{K,Type{K}}) where {K<:QuadraticSpline} = false
isnormalized(::Union{K,Type{K}}) where {K<:QuadraticSpline} = true
Base.summary(::QuadraticSpline) = "QuadraticSpline()"

function (::QuadraticSpline{T,B})(x::T) where {T<:AbstractFloat,B<:Boundaries}
    a = abs(x)
    return (a ≥ T(3/2) ? T(0) :
            a ≤ T(1/2) ? T(3/4) - a*a :
            square(a - T(3/2))*T(1/2))
end

@inline function getweights(::QuadraticSpline{T,B},
                            t::T) where {T<:AbstractFloat,B}
    #return (T(1/8)*(T(1) - T(2)*t)^2,
    #        T(3/4) - t^2,
    #        T(1/8)*(T(1) + T(2)*t)^2)
    const c1 = T(0.35355339059327376220042218105242451964241796884424) # 1/sqrt(8)
    const c2 = T(0.70710678118654752440084436210484903928483593768847) # 2/sqrt(8)
    const c3 = T(3/4)
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
struct CubicSpline{T,B} <: Kernel{T,4,B}; end

iscardinal(::Union{K,Type{K}}) where {K<:CubicSpline} = false
isnormalized(::Union{K,Type{K}}) where {K<:CubicSpline} = true
Base.summary(::CubicSpline) = "CubicSpline()"

function (::CubicSpline{T,B})(x::T) where {T<:AbstractFloat,B}
    a = abs(x)
    return (a ≥ T(2) ? T(0) :
            a ≥ T(1) ? cube(T(2) - a)*T(1/6) :
            (T(1/2)*a - T(1))*a*a + T(2/3))
end

@inline function getweights(ker::CubicSpline{T,B},
                            t::T) where {T<:AbstractFloat,B}
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
    const q = T(1/6)
    r = T(1) - t
    r2 = r*r
    t2 = t*t
    w1 = q*r2*r
    w2 = T(2/3) + (T(1/2)*t - T(1))*t2
    w3 = T(4/6) - T(1/2)*r2*(t + T(1))
    w4 = q*t2*t
    return w1, w2, w3, w4
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
            a ≤ T(1) ? (T(3/2)*a - T(5/2))*a*a + T(1) :
            ((T(5/2) - T(1/2)*a)*a - T(4))*a + T(2))
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
a cardinal sine.

"""
struct CardinalCubicSpline{T,B} <: Kernel{T,4,B}
    α::T
    β::T

    (::Type{Kernels.CardinalCubicSpline{T,B}})(α::Real, β::Real) where {T,B} =
        new{T,B}(α, β)

    (::Type{Kernels.CardinalCubicSpline{T,B}})(c::Real) where {T,B} =
        new{T,B}((c - 1)/2, (c + 1)/2)
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
Base.summary(ker::CardinalCubicSpline) =
    @sprintf("CardinalCubicSpline(%.1f)", ker.α + ker.β)

function convert(::Type{CardinalCubicSpline{T,B}},
                 ker::CardinalCubicSpline) where {T<:AbstractFloat,
                                                  B<:Boundaries}
    CardinalCubicSpline(T, ker.α + ker.β, B)
end

function (ker::CardinalCubicSpline{T,B})(x::T) where {T<:AbstractFloat,B}
    a = abs(x)
    return (a ≥ T(2) ? T(0) :
            a ≥ T(1) ? ker.α*(a - T(1))*square(T(2) - a) :
            ((ker.β*a + a)*a - a - T(1))*(a - T(1)))
end

@inline function getweights(ker::CardinalCubicSpline{T,B},
                            t::T) where {T<:AbstractFloat,B}
    α = ker.α
    β = ker.β
    # Computation of:
    #     w1 = α s² t
    #     w2 = s + t s² - β s t²
    #     w3 = t + t² s - β s² t
    #     w4 = α s t²
    # with s = 1 - t in 13 operations.
    s = T(1) - t
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
    MitchellNetravaliSpline{T,B}(T(1/3), T(1/3))
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

@inline function getweights(ker::LanczosKernel{T,2,B}, t::T) where {T,B}
    local w1::T, w2::T
    if t == T(0)
        w1, w2 = 1, 0
    else
        w1 = _p(ker, t)
        w2 = _p(ker, T(1) - t)
    end
    return w1, w2
end

@inline function getweights(ker::LanczosKernel{T,4,B}, t::T) where {T,B}
    local w1::T, w2::T, w3::T, w4::T
    if t == T(0)
        w1, w2, w3, w4 = 0, 1, 0, 0
    else
        w1 = _p(ker, t + T(1))
        w2 = _p(ker, t)
        w3 = _p(ker, T(1) - t)
        w4 = _p(ker, T(2) - t)
    end
    return w1, w2, w3, w4
end

@inline function getweights(ker::LanczosKernel{T,6,B}, t::T) where {T,B}
    local w1::T, w2::T, w3::T, w4::T, w5::T, w6::T
    if t == T(0)
        w1, w2, w3, w4, w5, w6 = 0, 0, 1, 0, 0, 0
    else
        w1 = _p(ker, t + T(2))
        w2 = _p(ker, t + T(1))
        w3 = _p(ker, t)
        w4 = _p(ker, T(1) - t)
        w5 = _p(ker, T(2) - t)
        w6 = _p(ker, T(3) - t)
    end
    return w1, w2, w3, w4, w5, w6
end

@inline function getweights(ker::LanczosKernel{T,8,B}, t::T) where {T,B}
    local w1::T, w2::T, w3::T, w4::T, w5::T, w6::T, w7::T, w8::T
    if t == T(0)
        w1, w2, w3, w4, w5, w6, w7, w8 = 0, 0, 0, 1, 0, 0, 0, 0
    else
        w1 = _p(ker, t + T(3))
        w2 = _p(ker, t + T(2))
        w3 = _p(ker, t + T(1))
        w4 = _p(ker, t)
        w5 = _p(ker, T(1) - t)
        w6 = _p(ker, T(2) - t)
        w7 = _p(ker, T(3) - t)
        w8 = _p(ker, T(4) - t)
    end
    return w1, w2, w3, w4, w5, w6, w7, w8
end

#------------------------------------------------------------------------------

# Provide methods for parameter-less kernels.
for K in (:RectangularSpline, :LinearSpline, :QuadraticSpline,
          :CubicSpline, :CatmullRomSpline)
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

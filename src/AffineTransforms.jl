#
# AffineTransforms.jl --
#
# Implementation of affine transforms which are notably useful for coordinate
# transforms.
#
#------------------------------------------------------------------------------
#
# This file is part of the LazyInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2016-2018, Éric Thiébaut.
#

module AffineTransforms

export
    AffineTransform2D,
    compose,
    rotate,
    scale,
    translate,
    intercept,
    jacobian

@static if isdefined(Base, :scale)
    import Base.scale
end

"""
# Affine 2D Transforms

An affine 2D transform `C` is defined by 6 real coefficients, `Cxx`, `Cxy`,
`Cx`, `Cyx`, `Cyy` and `Cy`.  Such a transform maps `(x,y)` as `(xp,yp)` given
by:
```
  xp = Cxx*x + Cxy*y + Cx
  yp = Cyx*x + Cyy*y + Cy
```

The immutable type `AffineTransform2D` is used to store an affine 2D transform
`C`, it can be created by:
```
  C = AffineTransform2D{T}() # yields the identity with type T
  C = AffineTransform2D{T}(Cxx, Cxy, Cx, Cyx, Cyy, Cy)
```
The `{T}` above is used to specify the floating-point type for the
coefficients; if omitted, `T = Cdouble` is assumed.

Many operations are available to manage or apply affine transforms:
```julia
(xp, yp) = A(x, y)         # idem
(xp, yp) = A(xy)           # idem
(xp, yp) = A*xy            # idem

B = convert(T, A)       # convert coefficients of transform A to be of type T

C = compose(A, B, ...)  # compose 2 (or more) transforms, C = apply B then A
C = A∘B                 # idem
C = A*B                 # idem
C = A⋅B                 # idem

B = translate(tx, ty, A)   # B = apply A then translate by (tx,ty)
B = translate(t, A)        # idem with t = (tx,ty)
B = t + A                  # idem

B = translate(A, tx, ty)   # B = translate by (tx,ty) then apply A
B = translate(A, t)        # idem with t = (tx,ty)
B = A + t                  # idem

B = rotate(θ, A)   # B = apply A then rotate by angle θ
C = rotate(A, θ)   # C = rotate by angle θ then apply A

B = scale(ρ, A)    # B = apply A then scale by ρ
B = ρ*A            # idem
C = scale(A, ρ)    # C = scale by ρ then apply A
C = A*ρ            # idem

B = inv(A)         # reciprocal coordinate transform
C = A/B            # right division, same as: C = compose(A, inv(B))
C = A\\B            # left division, same as: C = compose(inv(A), B)
```

"`∘`" and "`⋅`" can be typed by `\\circ<tab>` and `\\cdot<tab>`.

"""
struct AffineTransform2D{T<:AbstractFloat}
    xx::T
    xy::T
    x ::T
    yx::T
    yy::T
    y ::T
    (::Type{AffineTransform2D{T}}){T}() = new{T}(1,0,0, 0,1,0)
    (::Type{AffineTransform2D{T}}){T}(a11::Real, a12::Real, a13::Real,
                                      a21::Real, a22::Real, a23::Real) =
                                          new{T}(a11,a12,a13, a21,a22,a23)
end

# Use Cdouble type by default.
AffineTransform2D() = AffineTransform2D{Cdouble}()
AffineTransform2D(a11::Real, a12::Real, a13::Real,
                  a21::Real, a22::Real, a23::Real) =
                      AffineTransform2D{Cdouble}(a11,a12,a13, a21,a22,a23)

@deprecate(
    AffineTransform2D{T<:AbstractFloat}(::Type{T}),
    AffineTransform2D{T}())

@deprecate(
    AffineTransform2D{T<:AbstractFloat}(::Type{T},
                                        a11::Real, a12::Real, a13::Real,
                                        a21::Real, a22::Real, a23::Real),
    AffineTransform2D{T}(a11,a12,a13, a21,a22,a23))

# The following is a no-op when the destination type matches that of the
# source.
#
# The trick is to have a more restrictive signature than the general case
# above.  So the template type T for the result must have the same
# restrictions as in the general case.
#
# Another trick to remember: you can call a specific constructor, e.g.
# AffineTransform2D{Float16}, but this is not allowed for methods for which
# you must rely on Julia dispatching rules.
#
# When you make specialized versions of methods beware of infinite loops
# resulting from recursively calling the same method.  The diagnostic is a
# stack overflow.
#
function Base.convert(::Type{AffineTransform2D{T}},
                      A::AffineTransform2D{T}) where {T<:AbstractFloat}
    return A
end

function Base.convert(::Type{AffineTransform2D{T}},
                      A::AffineTransform2D) where {T<:AbstractFloat}
    return AffineTransform2D{T}(A.xx, A.xy, A.x, A.yx, A.yy, A.y)
end

#------------------------------------------------------------------------------
# apply the transform to some coordinates:

(A::AffineTransform2D{T})(x::T, y::T) where {T<:AbstractFloat} =
    (A.xx*x + A.xy*y + A.x,
     A.yx*x + A.yy*y + A.y)

(A::AffineTransform2D{T})(x::Real, y::Real) where {T<:AbstractFloat} =
    A(convert(T, x), convert(T, y))

(A::AffineTransform2D)(v::Tuple{Real,Real}) = A(v[1], v[2])

#------------------------------------------------------------------------------
# Combine a translation with an affine transform.

# Left-translating results in translating the output of the transform.
translate(x::T, y::T, A::AffineTransform2D{T}) where {T<:AbstractFloat} =
    AffineTransform2D{T}(A.xx, A.xy, A.x + x,
                         A.yx, A.yy, A.y + y)

# Right-translating results in translating the input of the transform.
translate(A::AffineTransform2D{T}, x::T, y::T) where {T<:AbstractFloat} =
    AffineTransform2D{T}(A.xx, A.xy, A.xx*x + A.xy*y + A.x,
                         A.yx, A.yy, A.yx*x + A.yy*y + A.y)

translate(A::AffineTransform2D{T}, x::Real, y::Real) where {T<:AbstractFloat} =
    translate(A, convert(T, x), convert(T, y))

translate(A::AffineTransform2D{T},t::NTuple{2,T}) where {T<:AbstractFloat} =
    translate(A, t[1], t[2])

function translate(A::AffineTransform2D{T},
                   t::Tuple{T1,T2}) where {T<:AbstractFloat,T1<:Real,T2<:Real}
    return translate(A, convert(T, t[1]), convert(T, t[2]))
end

function translate(x::T1, y::T2,
                   A::AffineTransform2D{T}) where {T<:AbstractFloat,
                                                   T1<:Real,T2<:Real}
    return translate(convert(T, x), convert(T, y), A)
end

translate(t::NTuple{2,T}, A::AffineTransform2D{T}) where {T<:AbstractFloat} =
    translate(t[1], t[2], A)

function translate(t::Tuple{T1,T2},
                   A::AffineTransform2D{T}) where {T<:AbstractFloat,
                                                   T1<:Real,T2<:Real}
    return translate(convert(T, t[1]), convert(T, t[2]), A)
end

#------------------------------------------------------------------------------
"""
### Scaling an affine transform

There are two ways to combine a scaling by a factor `ρ` with an affine
transform `A`.  Left-scaling as in:
```
    B = scale(ρ, A)
```
results in scaling the output of the transform; while right-scaling as in:
```
    C = scale(A, ρ)
```
results in scaling the input of the transform.  The above examples yield
transforms which behave as:
```
    B*t = ρ*(A*t) = ρ*A(t)
    C*t = A*(ρ*t) = A(ρ*t)
```
where `t` is any 2-element tuple.
"""
function scale(ρ::T, A::AffineTransform2D{T}) where {T<:AbstractFloat}
    return AffineTransform2D{T}(ρ*A.xx, ρ*A.xy, ρ*A.x,
                                ρ*A.yx, ρ*A.yy, ρ*A.y)
end
function scale(A::AffineTransform2D{T}, ρ::T) where {T<:AbstractFloat}
    return AffineTransform2D{T}(ρ*A.xx, ρ*A.xy, A.x,
                                ρ*A.yx, ρ*A.yy, A.y)
end

#------------------------------------------------------------------------------
"""
### Rotating an affine transform

There are two ways to combine a rotation by angle `θ` (in radians
counterclockwise) with an affine transform `A`.  Left-rotating as in:
```
    B = rotate(θ, A)
```
results in rotating the output of the transform; while right-rotating as in:
```
    C = rotate(A, θ)
```
results in rotating the input of the transform.  The above examples are
similar to:
```
    B = R*A
    C = A*R
```
where `R` implements rotation by angle `θ` around `(0,0)`.
"""
function rotate(θ::T, A::AffineTransform2D{T}) where {T<:AbstractFloat}
    cs = cos(θ)
    sn = sin(θ)
    return AffineTransform2D{T}(cs*A.xx - sn*A.yx,
                                cs*A.xy - sn*A.yy,
                                cs*A.x  - sn*A.y,
                                cs*A.yx + sn*A.xx,
                                cs*A.yy + sn*A.xy,
                                cs*A.y  + sn*A.x)
end

function rotate(A::AffineTransform2D{T}, θ::T) where {T<:AbstractFloat}
    cs = cos(θ)
    sn = sin(θ)
    return AffineTransform2D{T}(A.xx*cs + A.xy*sn,
                                A.xy*cs - A.xx*sn,
                                A.x,
                                A.yx*cs + A.yy*sn,
                                A.yy*cs - A.yx*sn,
                                A.y)
end

for func in (:scale, :rotate)
    @eval begin
        function $func(q::S,
                       A::AffineTransform2D{T}) where {S<:Real,T<:AbstractFloat}
            return $func(convert(T, q), A)
        end
        function $func(A::AffineTransform2D{T},
                       q::S) where {S<:Real,T<:AbstractFloat}
            return $func(A, convert(T, q))
        end
    end
end

#------------------------------------------------------------------------------

"""
`det(A)` returns the determinant of the linear part of the affine
transform `A`.
"""
Base.det(A::AffineTransform2D) = A.xx*A.yy - A.xy*A.yx

"""
`jacobian(A)` returns the Jacobian of the affine transform `A`, that is the
absolute value of the determinant of its linear part.
"""
jacobian(A::AffineTransform2D) = abs(det(A))

"""
`inv(A)` returns the inverse of the affine transform `A`.
"""
function Base.inv(A::AffineTransform2D{T}) where {T<:AbstractFloat}
    d = det(A)
    d == zero(T) && error("transformation is not invertible")
    Txx =  A.yy/d
    Txy = -A.xy/d
    Tyx = -A.yx/d
    Tyy =  A.xx/d
    return AffineTransform2D{T}(Txx, Txy, -Txx*A.x - Txy*A.y,
                                Tyx, Tyy, -Tyx*A.x - Tyy*A.y)
end

"""

`compose(A,B)` yields the affine transform which combines the two affine
transforms `A` and `B`, that is the affine transform which applies `B` and then
`A`.  Composition is accessible via: `A∘B`, `A*B` or `A⋅B` ("`∘`" and "`⋅`" can
be typed by `\\circ<tab>` and `\\cdot<tab>`).

It is possible to compose more than two affine transforms.  For instance,
`compose(A,B,C)` yields the affine transform which applies `C` then `B`, then
`A`.

"""
compose() = error("missing argument(s)")

compose(A::AffineTransform2D) = A

compose(A::AffineTransform2D, B::AffineTransform2D) = __compose(A, B)

compose(args::AffineTransform2D...) =
     compose(__compose(args[1], args[2]), args[3:end]...)

function __compose(A::AffineTransform2D{Ta},
                   B::AffineTransform2D{Tb}) where {Ta, Tb}
    T = promote_type(Ta, Tb)
    return AffineTransform2D{T}(A.xx*B.xx + A.xy*B.yx,
                                A.xx*B.xy + A.xy*B.yy,
                                A.xx*B.x  + A.xy*B.y + A.x,
                                A.yx*B.xx + A.yy*B.yx,
                                A.yx*B.xy + A.yy*B.yy,
                                A.yx*B.x  + A.yy*B.y + A.y)
end

"""

`rightdivide(A,B)` yields `A/B`, the right division of the affine
transform `A` by the affine transform `B`.

"""
function rightdivide(A::AffineTransform2D{T},
                     B::AffineTransform2D{T}) where {T<:AbstractFloat}
    d = det(B)
    d == zero(T) && error("right operand is not invertible")
    Rxx = (A.xx*B.yy - A.xy*B.yx)/d
    Rxy = (A.xy*B.xx - A.xx*B.xy)/d
    Ryx = (A.yx*B.yy - A.yy*B.yx)/d
    Ryy = (A.yy*B.xx - A.yx*B.xy)/d
    return AffineTransform2D{T}(Rxx, Rxy, A.x - (Rxx*B.x + Rxy*B.y),
                                Ryx, Ryy, A.y - (Ryx*B.y + Ryy*B.y))

end

"""
`leftdivide(A,B)` yields `A\\B`, the left division of the affine
transform `A` by the affine transform `B`.
"""
function leftdivide(A::AffineTransform2D{T},
                    B::AffineTransform2D{T}) where {T<:AbstractFloat}
    d = det(A)
    d == zero(T) && error("left operand is not invertible")
    Txx =  A.yy/d
    Txy = -A.xy/d
    Tyx = -A.yx/d
    Tyy =  A.xx/d
    Tx = B.x - A.x
    Ty = B.y - A.y
    return AffineTransform2D{T}(Txx*B.xx + Txy*B.yx,
                                Txx*B.xy + Txy*B.yy,
                                Txx*Tx   + Txy*Ty,
                                Tyx*B.xx + Tyy*B.yx,
                                Tyx*B.xy + Tyy*B.yy,
                                Tyx*Tx   + Tyy*Ty)
end

for func in (:rightdivide, :leftdivide)
    @eval begin
        function $func(A::AffineTransform2D{Ta},
                       B::AffineTransform2D{Tb}) where {Ta<:AbstractFloat,
                                                        Tb<:AbstractFloat}
            T = AffineTransform2D{promote_type(Ta, Tb)}
            return $func(convert(T, A), convert(T, B))
        end
    end
end

"""

`intercept(A)` returns the tuple `(x,y)` such that `A(x,y) = (0,0)`.

"""
function intercept(A::AffineTransform2D{T}) where {T<:AbstractFloat}
    d = det(A)
    d == zero(T) && error("transformation is not invertible")
    return ((A.xy*A.y - A.yy*A.x)/d, (A.yx*A.x - A.xx*A.y)/d)
end


Base.:+(t::NTuple{2}, A::AffineTransform2D) = translate(t, A)

Base.:+(A::AffineTransform2D, t::NTuple{2}) = translate(A, t)

for op in (:(∘), :(*), :(⋅))
    @eval begin
        Base.$op(A::AffineTransform2D, B::AffineTransform2D) = compose(A, B)
    end
end

Base.:*(A::AffineTransform2D, t::NTuple{2}) = A(t)

Base.:*(ρ::Real, A::AffineTransform2D) = scale(ρ, A)

Base.:*(A::AffineTransform2D, ρ::Real) = scale(A, ρ)

Base.:\(A::AffineTransform2D, B::AffineTransform2D) = leftdivide(A, B)

Base.:/(A::AffineTransform2D, B::AffineTransform2D) = rightdivide(A, B)

function Base.show(io::IO, A::AffineTransform2D)
    println(io, typeof(A), ":")
    println(io, "  ", A.xx, "  ", A.xy, " | ", A.x)
    println(io, "  ", A.yx, "  ", A.yy, " | ", A.y)
end

function runtests()
    B = AffineTransform2D(1, 0, -3, 0.1, 1, +2)
    show(B)
    println()
    A = inv(B)
    show(A)
    println()
    C = compose(A, B)
    show(C)
    println()
    U = convert(AffineTransform2D{Float16},C)
    show(U)
    println()
    V = convert(AffineTransform2D{Float64},C)
    show(V)
    println()
    show(B(1, 4))
    println()
    show(B(1f0, 4))
    println()
    show(B(1.0, 4.0))
    println()
    show(B(1.0, 4))
    println()
    show(B((1f0, 4f0)))
    println()
    show(B((1.0, 4f0)))
    println()
    xy = intercept(B)
    xpyp = B*xy
    println("$xy --> $xpyp")
    nothing
end

@deprecate combine compose
@deprecate multiply compose

end # module

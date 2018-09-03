#
# interp/tabulated.jl -
#
# Implement unidimensional interpolation operator using precomputed indices and
# coefficients.
#
#------------------------------------------------------------------------------
#
# This file is part of the LinearInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2016-2018, Éric Thiébaut.
#

module TabulatedInterpolators

export
    TabulatedInterpolator

using ...Kernels
using ...Interpolations
import ...Interpolations.Meta

using LazyAlgebra
import LazyAlgebra: vcreate, apply!, apply

struct TabulatedInterpolator{T<:AbstractFloat,S,D} <: LinearMapping
    nrows::Int     # length of output dimension
    ncols::Int     # length of input dimension
    J::Matrix{Int} # indices along input dimension
    W::Matrix{T}   # interpolation weights
    d::D           # dimension to interpolate or nothing
end

# Interpolator can be used as a function.
(A::TabulatedInterpolator)(x::AbstractArray) = apply(A, x)

nrows(A::TabulatedInterpolator) = A.nrows
ncols(A::TabulatedInterpolator) = A.ncols
width(A::TabulatedInterpolator{T,S,D}) where {T,S,D} = S
coefficients(A::TabulatedInterpolator) = A.W
columns(A::TabulatedInterpolator) = A.J

"""
```julia
TabulatedInterpolator([T,] [d = nothing,] ker, pos, nrows, ncols)
```

yields a linear map to interpolate with kernel `ker` along a dimension of
length `ncols` to produce a dimension of length `nrows`.  The function `pos(i)`
for `i ∈ 1:nrows` gives the positions to interpolate in fractional index unit
along a dimension of length `ncols`.  Argument `d` is the rank of the dimension
along which to interpolate when the operator is applied to a multidimensional
array.  If it is unspecifed when the operator is created, it will have to be
specified each time the operator is applied (see below).

The positions to interpolate can also be specified by a vector `x` as in:

```julia
TabulatedInterpolator([T,] [d = nothing,] ker, x, ncols)
```

to produce an interpolator whose output dimension is `nrows = length(x)`.

Optional argument `T` is the floating-point type of the coefficients of the
linear map.  By default, it is given by the promotion of the element type of
the arguments `ker` and, if specified, `x`.

The tabulated operator, say `A`, can be applied to an argument `x`:

```julia
apply!(α, P::Operations, A, [d,] x, β, y)
```

to overwrite `y` with `α*P(A)⋅x + β*y`.  If `x` and `y` are multi-dimensional,
the dimension `d` to interpolate must be specified.

"""
function TabulatedInterpolator(::Type{T},
                               d::D,
                               ker::Kernel{T,S},
                               pos::Function,
                               nrows::Integer,
                               ncols::Integer) where {T<:AbstractFloat,
                                                      D<:Union{Nothing,Int},S}
    nrows ≥ 1 || error("number of rows too small")
    ncols ≥ 1 || error("number of columns too small")
    J, W = __maketables(ker, pos, convert(Int, nrows), convert(Int, ncols))
    return TabulatedInterpolator{T,S,D}(nrows, ncols, J, W, d)
end

function TabulatedInterpolator(::Type{T},
                               d::D,
                               ker::Kernel{T,S},
                               x::AbstractVector{<:Real},
                               ncols::Integer) where {T<:AbstractFloat,
                                                      D<:Union{Nothing,Int},S}
    nrows = length(x)
    nrows ≥ 1 || error("number of positions too small")
    ncols ≥ 1 || error("number of columns too small")
    J, W = __maketables(ker, x, nrows, convert(Int, ncols))
    return TabulatedInterpolator{T,S,D}(nrows, ncols, J, W, d)
end

# This one forces the kernel to have the requested precision.
function TabulatedInterpolator(::Type{T},
                               d::D,
                               ker::Kernel{K,S,B},
                               args...) where {T<:AbstractFloat,
                                               D<:Union{Nothing,Int},K,S,B}
    TabulatedInterpolator(T, d, T(ker), args...)
end

function TabulatedInterpolator(d::D,
                               ker::Kernel{T,S},
                               pos::Function,
                               nrows::Integer,
                               ncols::Integer) where {D<:Union{Nothing,Int},T,S}
    TabulatedInterpolator(float(T), d, ker, pos, nrows, ncols)
end

function TabulatedInterpolator(d::D,
                               ker::Kernel{Tk,S},
                               x::AbstractVector{Tx},
                               ncols::Integer) where {D<:Union{Nothing,Int},
                                                      Tk<:AbstractFloat,S,
                                                      Tx<:Real}
    TabulatedInterpolator(float(promote_type(Tk, Tx)), d, ker, x, ncols)
end

TabulatedInterpolator(T::DataType, ker::Kernel, args...) =
    TabulatedInterpolator(T, nothing, ker, args...)

TabulatedInterpolator(ker::Kernel, args...) =
    TabulatedInterpolator(nothing, ker, args...)

TabulatedInterpolator(T::DataType, d::Integer, ker::Kernel, args...) =
    TabulatedInterpolator(T, convert(Int, d), ker, args...)

TabulatedInterpolator(d::Integer, ker::Kernel, args...) =
    TabulatedInterpolator(convert(Int, d), ker, args...)

@generated function __maketables(ker::Kernel{T,S},
                                 X::AbstractVector,
                                 nrows::Int,
                                 ncols::Int) where {T,S}
    _J, _W = Meta.make_varlist(:_j, S), Meta.make_varlist(:_w, S)
    code = (Meta.generate_getcoefs(_J, _W, :ker, :lim, :x),
            [:( J[$s,i] = $(_J[s]) ) for s in 1:S]...,
            [:( W[$s,i] = $(_W[s]) ) for s in 1:S]...)
    quote
        J = Array{Int}(undef, S, nrows)
        W = Array{T}(undef, S, nrows)
        lim = limits(ker, ncols)
        @inbounds for i in 1:nrows
            x = convert(T, X[i])
            $(code...)
        end
        return J, W
    end
end

@generated function __maketables(ker::Kernel{T,S},
                                 pos::Function,
                                 nrows::Int,
                                 ncols::Int) where {T,S}
    _J, _W = Meta.make_varlist(:_j, S), Meta.make_varlist(:_w, S)
    code = (Meta.generate_getcoefs(_J, _W, :ker, :lim, :x),
            [:( J[$s,i] = $(_J[s]) ) for s in 1:S]...,
            [:( W[$s,i] = $(_W[s]) ) for s in 1:S]...)
    quote
        J = Array{Int}(undef, S, nrows)
        W = Array{T}(undef, S, nrows)
        lim = limits(ker, ncols)
        @inbounds for i in 1:nrows
            x = convert(T, pos(i))
            $(code...)
        end
        return J, W
    end
end

# Source and destination arrays may have elements which are either reals or
# complexes.
const RorC{T<:AbstractFloat} = Union{T,Complex{T}}

@inline __errdims(msg::String) = throw(DimensionMismatch(msg))

@inline function __check(A::TabulatedInterpolator{T,S}) where {T,S}
    if !(size(A.J) == size(A.W) == (S, A.nrows))
        error("corrupted interpolation table")
    end
end

@inline function __fullcheck(A::TabulatedInterpolator{T,S}) where {T,S}
    # The loop below is a bit faster than using `extrema`.
    __check(A)
    J, ncols = A.J, A.ncols
    @inbounds for k in eachindex(J)
        if !(1 ≤ J[k] ≤ ncols)
            error("out of bound index in interpolation table")
        end
    end
end

# For vectors, the dimension to interpolate is 1.
function vcreate(::Type{P},
                 A::TabulatedInterpolator{R,S,Nothing},
                 src::AbstractVector{T}) where {P<:Operations,
                                                R<:AbstractFloat,S,
                                                T<:RorC{R}}
    __vcreate(P, A, 1, src)
end

function vcreate(::Type{P},
                 A::TabulatedInterpolator{R,S,Nothing},
                 src::AbstractArray{T,N}) where {P<:Operations,
                                                 R<:AbstractFloat,S,
                                                 T<:RorC{R},N}
    error("dimension to interpolate must be provided")
end

function vcreate(::Type{P},
                 A::TabulatedInterpolator{R,S,Nothing},
                 d::Integer,
                 src::AbstractArray{T,N}) where {P<:Operations,
                                                 R<:AbstractFloat,S,
                                                 T<:RorC{R},N}
    __vcreate(P, A, convert(Int, d), src)
end

function apply!(alpha::Real,
                ::Type{P},
                A::TabulatedInterpolator{R,S,Nothing},
                d::Integer,
                src::AbstractArray{T,N},
                beta::Real,
                dst::AbstractArray{T,N}) where {P<:Operations,
                                                R<:AbstractFloat,S,
                                                T<:RorC{R},N}
    __apply!(alpha, P, A, convert(Int, d), src, beta, dst)
end

# For vectors, the dimension to interpolate is 1.
function apply!(alpha::Real,
                ::Type{P},
                A::TabulatedInterpolator{R,S,Nothing},
                src::AbstractVector{T},
                beta::Real,
                dst::AbstractVector{T}) where {P<:Operations,
                                               R<:AbstractFloat,S,
                                               T<:RorC{R}}
    __apply!(alpha, P, A, 1, src, beta, dst)
end

function apply!(alpha::Real,
                ::Type{P},
                A::TabulatedInterpolator{R,S,Nothing},
                src::AbstractArray{T,N},
                beta::Real,
                dst::AbstractArray{T,N}) where {P<:Operations,
                                                R<:AbstractFloat,S,
                                                T<:RorC{R},N}
    error("dimension to interpolate must be provided")
end

function vcreate(::Type{P},
                 A::TabulatedInterpolator{R,S,Int},
                 src::AbstractArray{T,N}) where {P<:Operations,
                                                 R<:AbstractFloat,S,
                                                 T<:RorC{R},N}
    __vcreate(P, A, A.d, src)
end

function apply!(alpha::Real,
                ::Type{P},
                A::TabulatedInterpolator{R,S,Int},
                src::AbstractArray{T,N},
                beta::Real,
                dst::AbstractArray{T,N}) where {P<:Operations,
                                                R<:AbstractFloat,S,
                                                T<:RorC{R},N}
    __apply!(alpha, P, A, A.d, src, beta, dst)
end

function __vcreate(::Type{Direct},
                   A::TabulatedInterpolator{R,S,D},
                   d::Int,
                   src::AbstractArray{T,N}) where {R<:AbstractFloat,S,D,
                                                   T<:RorC{R},N}
    1 ≤ d ≤ N || error("out of range dimension $d")
    nrows = A.nrows
    srcdims = size(src)
    dstdims = ntuple(i -> (i == d ? nrows : srcdims[i]), Val(N))
    return Array{T}(undef, dstdims)
end

function __vcreate(::Type{Adjoint},
                   A::TabulatedInterpolator{R,S,D},
                   d::Int,
                   src::AbstractArray{T,N}) where {R<:AbstractFloat,S,D,
                                                   T<:RorC{R},N}
    1 ≤ d ≤ N || error("out of range dimension $d")
    ncols = A.ncols
    srcdims = size(src)
    dstdims = ntuple(i -> (i == d ? ncols : srcdims[i]), Val(N))
    return Array{T}(undef, dstdims)
end

function  __apply!(alpha::Real,
                   ::Type{Direct},
                   A::TabulatedInterpolator{R,S,D},
                   d::Int,
                   src::AbstractArray{T,N},
                   beta::Real,
                   dst::AbstractArray{T,N}) where {R<:AbstractFloat,S,D,
                                                   T<:RorC{R},N}
    # Check arguments.
    __fullcheck(A)
    1 ≤ d ≤ N || error("out of range dimension $d")
    nrows, ncols = A.nrows, A.ncols
    srcdims, dstdims = size(src), size(dst)
    dstdims[d] == nrows ||
        __errdims("dimension $d of destination must be $nrows")
    srcdims[d] == ncols ||
        __errdims("dimension $d of source must be $ncols")
    predims = srcdims[1:d-1]
    postdims = srcdims[d+1:end]
    (dstdims[1:d-1] == predims && dstdims[d+1:end] == postdims) ||
        __errdims("incompatible destination and source dimensions")

    # Get rid of the case alpha = 0 and apply direct interpolation.
    if alpha == 0
        vscale!(dst, beta)
    else
        __direct!(convert(R, alpha), A, src,
                  CartesianIndices(predims), nrows, CartesianIndices(postdims),
                  convert(R, beta), dst)
    end
    return dst
end

function __apply!(alpha::Real,
                  ::Type{Adjoint},
                  A::TabulatedInterpolator{R,S,D},
                  d::Int,
                  src::AbstractArray{T,N},
                  beta::Real,
                  dst::AbstractArray{T,N}) where {R<:AbstractFloat,S,D,
                                                  T<:RorC{R},N}
    # Check arguments.
    __fullcheck(A)
    1 ≤ d ≤ N || error("out of range dimension $d")
    nrows, ncols = A.nrows, A.ncols
    srcdims, dstdims = size(src), size(dst)
    dstdims[d] == ncols ||
        __errdims("dimension $d of destination must be $ncols")
    srcdims[d] == nrows ||
        __errdims("dimension $d of source must be $nrows")
    predims = srcdims[1:d-1]
    postdims = srcdims[d+1:end]
    (dstdims[1:d-1] == predims && dstdims[d+1:end] == postdims) ||
        __errdims("incompatible destination and source dimensions")

    # Scale destination by beta, and apply adjoint interpolation unless
    # alpha = 0.
    vscale!(dst, beta)
    if alpha != 0
        __adjoint!(convert(R, alpha), A, src, CartesianIndices(predims), nrows,
                   CartesianIndices(postdims), dst)
    end
    return dst
end

function __direct!(α::R,
                   A::TabulatedInterpolator{R,S,D},
                   src::AbstractArray{T,N},
                   Ipre::CartesianIndices,
                   nrows::Int,
                   Ipost::CartesianIndices,
                   β::R,
                   dst::AbstractArray{T,N}) where {R<:AbstractFloat,S,D,
                                                   T<:RorC{R},N}
    J, W = A.J, A.W
    # We already know that α != 0.
    if α == 1 && β == 0
        @inbounds for ipost in Ipost, ipre in Ipre, i in 1:nrows
            a = zero(T)
            @simd for s in 1:S
                j, w = J[s,i], W[s,i]
                a += src[ipre,j,ipost]*w
            end
            dst[ipre,i,ipost] = a
        end
    elseif β == 0
        @inbounds for ipost in Ipost, ipre in Ipre, i in 1:nrows
            a = zero(T)
            @simd for s in 1:S
                j, w = J[s,i], W[s,i]
                a += src[ipre,j,ipost]*w
            end
            dst[ipre,i,ipost] = α*a
        end
    else
        @inbounds for ipost in Ipost, ipre in Ipre, i in 1:nrows
            a = zero(T)
            @simd for s in 1:S
                j, w = J[s,i], W[s,i]
                a += src[ipre,j,ipost]*w
            end
            dst[ipre,i,ipost] = α*a + β*dst[ipre,i,ipost]
        end
    end
end

function __adjoint!(α::R,
                    A::TabulatedInterpolator{R,S,D},
                    src::AbstractArray{T,N},
                    Ipre::CartesianIndices,
                    nrows::Int,
                    Ipost::CartesianIndices,
                    dst::AbstractArray{T,N}) where {R<:AbstractFloat,S,D,
                                                    T<:RorC{R},N}
    J, W = A.J, A.W
    # We already know that α != 0.
    if α == 1
        @inbounds for ipost in Ipost, ipre in Ipre, i in 1:nrows
            x = src[ipre,i,ipost]
            if x != zero(T)
                @simd for s in 1:S
                    j, w = J[s,i], W[s,i]
                    dst[ipre,j,ipost] += w*x
                end
            end
        end
    else
        @inbounds for ipost in Ipost, ipre in Ipre, i in 1:nrows
            x = α*src[ipre,i,ipost]
            if x != zero(T)
                @simd for s in 1:S
                    j, w = J[s,i], W[s,i]
                    dst[ipre,j,ipost] += w*x
                end
            end
        end
    end
end

end # module

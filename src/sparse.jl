#
# interp/sparse.jl --
#
# Implement sparse linear interpolator.
#
#------------------------------------------------------------------------------
#
# This file is part of the LinearInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# Copyright (C) 2016-2021, Éric Thiébaut.
#

# All code is in a module to "hide" private methods.
module SparseInterpolators

export
    SparseInterpolator,
    SparseUnidimensionalInterpolator

using InterpolationKernels

using LazyAlgebra
using LazyAlgebra.Foundations
import LazyAlgebra: apply, apply!, vcreate, output_size, input_size

import Base: axes, eltype, size
import SparseArrays: sparse

using ..LinearInterpolators
using ..LinearInterpolators: limits, getcoefs
import ..LinearInterpolators.Meta
import ..LinearInterpolators: coefficients, columns, rows,
    fit, regularize, regularize!

abstract type AbstractSparseInterpolator{T<:AbstractFloat} <: LinearMapping end

eltype(A::AbstractSparseInterpolator) = eltype(typeof(A))
eltype(::Type{<:AbstractSparseInterpolator{T}}) where {T} = T

struct SparseInterpolator{T<:AbstractFloat,S,N} <: AbstractSparseInterpolator{T}
    C::Vector{T}
    J::Vector{Int}
    nrows::Int
    ncols::Int
    dims::Dims{N} # dimensions of result
    function SparseInterpolator{T,S,N}(C::Vector{T},
                                       J::Vector{Int},
                                       dims::Dims{N},
                                       ncols::Int) where {T,S,N}
        @assert S ≥ 1
        @assert minimum(dims) ≥ 1
        nrows = prod(dims)
        nvals = S*nrows       # number of non-zero coefficients
        @assert length(C) == nvals
        @assert length(J) == nvals
        new{T,S,N}(C, J, nrows, ncols, dims)
    end
end

# Interpolator can be used as a function.
(A::SparseInterpolator)(x::AbstractVector) = apply(A, x)

output_size(A::SparseInterpolator) = A.dims
input_size(A::SparseInterpolator) = (A.ncols,)
width(A::SparseInterpolator{T,S,N}) where {T,S,N} = S
coefficients(A::SparseInterpolator) = A.C
columns(A::SparseInterpolator) = A.J
function rows(A::SparseInterpolator{T,S,N}) where {T,S,N}
    nrows = A.nrows
    nvals = S*nrows       # number of non-zero coefficients
    @assert length(A.C) == nvals
    @assert length(A.J) == nvals
    I = Array{Int}(undef, nvals)
    k0 = 0
    for i in 1:nrows
        for s in 1:S
            k = k0 + s
            @inbounds I[k] = i
        end
        k0 += S
    end
    return I
end

# Convert to a sparse matrix.
sparse(A::SparseInterpolator) =
    sparse(rows(A), columns(A), coefficients(A), A.nrows, A.ncols)

"""
    A = SparseInterpolator{T=eltype(ker)}(ker, pos, grd)

yields a sparse linear interpolator suitable for interpolating with kernel
`ker` at positions `pos` a function sampled on the grid `grd`.  Optional
parameter `T` is the floating-point type of the coefficients of the operator
`A`.  Call `eltype(A)` to query the type of the coefficients of the sparse
interpolator `A`.

Then `y = apply(A, x)` or `y = A(x)` or `y = A*x` yield the result of
interpolation array `x`.  The shape of `y` is the same as that of `pos`.
Formally, this amounts to computing:

    y[i] = sum_j ker((pos[i] - grd[j])/step(grd))*x[j]

with `step(grd)` the (constant) step size between the nodes of the grid `grd`
and `grd[j]` the `j`-th position of the grid.

"""
SparseInterpolator(ker::Kernel{T}, args...) where {T<:AbstractFloat} =
    SparseInterpolator{T}(ker, args...)

@deprecate SparseInterpolator(T::Type{<:AbstractFloat}, ker::Kernel, args...) SparseInterpolator{T}(ker, args...)

SparseInterpolator{T}(ker::Kernel, args...) where {T<:AbstractFloat} =
    SparseInterpolator{T}(T(ker), args...)

function SparseInterpolator{T}(ker::Kernel{T},
                               pos::AbstractArray{<:Real},
                               grd::AbstractRange) where {T<:AbstractFloat}
    SparseInterpolator{T}(ker, fractional_index(T, pos, grd),
                          CartesianIndices(axes(pos)), length(grd))
end

function SparseInterpolator{T}(ker::Kernel{T},
                               pos::AbstractArray{<:Real},
                               len::Integer) where {T<:AbstractFloat}
    SparseInterpolator{T}(ker, fractional_index(T, pos),
                          CartesianIndices(axes(pos)), len)
end

function SparseInterpolator{T}(ker::Kernel{T,S},
                               f::Function,
                               R::CartesianIndices{N},
                               ncols::Integer) where {T<:AbstractFloat,S,N}
    C, J = _sparsecoefs(R, Int(ncols), ker, f)
    return SparseInterpolator{T,S,N}(C, J, size(R), ncols)
end

@generated function _sparsecoefs(R::CartesianIndices{N},
                                 ncols::Int,
                                 ker::Kernel{T,S},
                                 f::Function) where {T,S,N}

    J_ = [Symbol(:j_,s) for s in 1:S]
    C_ = [Symbol(:c_,s) for s in 1:S]
    code = (Meta.generate_getcoefs(J_, C_, :ker, :lim, :x),
            [:( J[k+$s] = $(J_[s]) ) for s in 1:S]...,
            [:( C[k+$s] = $(C_[s]) ) for s in 1:S]...)

    quote
        lim = limits(ker, ncols)
        nvals = S*length(R)
        J = Array{Int}(undef, nvals)
        C = Array{T}(undef, nvals)
        k = 0
        @inbounds for i in R
            x = convert(T, f(i))
            $(code...)
            k += S
        end
        return C, J
    end
end

function _check(A::SparseInterpolator{T,S,N},
                out::AbstractArray{T,N},
                inp::AbstractVector{T}) where {T,S,N}
    nvals = S*A.nrows # number of non-zero coefficients
    J, ncols = A.J, A.ncols
    length(A.C) == nvals ||
        error("corrupted sparse interpolator (bad number of coefficients)")
    length(J) == nvals ||
        error("corrupted sparse interpolator (bad number of indices)")
    length(inp) == ncols ||
        error("bad vector length (expecting $(A.ncols), got $(length(inp)))")
    size(out) == A.dims ||
        error("bad output array size (expecting $(A.dims), got $(size(out)))")
    length(out) == A.nrows ||
        error("corrupted sparse interpolator (bad number of \"rows\")")
    @inbounds for k in 1:nvals
        1 ≤ J[k] ≤ ncols ||
            error("corrupted sparse interpolator (out of bound indices)")
    end
end

function vcreate(::Type{Direct},
                 A::SparseInterpolator{T,S,N},
                 x::AbstractVector{T},
                 scratch::Bool=false) where {T,S,N}
    return Array{T}(undef, output_size(A))
end

function vcreate(::Type{Adjoint},
                 A::SparseInterpolator{T,S,N},
                 x::AbstractArray{T,N},
                 scratch::Bool=false) where {T,S,N}
    return Array{T}(undef, input_size(A))
end

@generated function direct_sumprod(W, A, J, off::Int, ::Val{S})
    ex = Expr(:call, :(+), [:(W[off + $k]*A[J[off + $k]]) for k in 1:S]...)
end

function apply!(α::Number,
                ::Type{Direct},
                A::SparseInterpolator{Ta,L,1,1,O,S},
                x::AbstractArray{Tx,Nx},
                scratch::Bool,
                β::Number,
                y::AbstractArray{Ty,Ny}) where {Ta,Tx,
                                               Ty<:AbstractFloat,S,N}
    _check(A, y, x)
    if iszero(α)
        vscale!(y, β)
    else
        T = float(promote_type(Ta, Tx))
        alpha = convert(T, α)
        nrows, ncols = A.nrows, A.ncols
        W, J = coefficients(A), columns(A)
        k0 = 0
        if β == 0
            @inbounds for i in 1:nrows
                s = sumprod(W, x, J, off::Int, Val(S))
                @simd for s in 1:S
                    k = k0 + s
                    j = J[k]
                    sum += C[k]*x[j]
                end
                y[i] = axpby(α, s, β, y[i])
                off += S
            end
        else
            beta = convert(Ty, β)
            @inbounds for i in 1:nrows
                sum = zero(T)
                @simd for s in 1:S
                    k = k0 + s
                    j = J[k]
                    sum += C[k]*x[j]
                end
                y[i] = alpha*sum + beta*y[i]
                k0 += S
            end
        end
    end
    return y
end

function apply!(α::Real,
                ::Type{Adjoint},
                A::SparseInterpolator{Ta,S,N},
                x::AbstractArray{Tx,N},
                scratch::Bool,
                β::Real,
                y::AbstractVector{Ty}) where {Ta,Tx<:Real,
                                              Ty<:AbstractFloat,S,N}
    _check(A, x, y)
    vscale!(y, β)
    if α != 0
        T = float(promote_type(Ta, Tx))
        alpha = convert(T, α)
        nrows, ncols = A.nrows, A.ncols
        C, J = coefficients(A), columns(A)
        k0 = 0
        @inbounds for i in 1:nrows
            c = alpha*x[i]
            if c != 0
                @simd for s in 1:S
                    k = k0 + s
                    j = J[k]
                    y[j] += C[k]*c
                end
            end
            k0 += S
        end
    end
    return y
end

# Yields a function that takes an index and returns the corresponding
# interpolation position as fractional index into the source array.
function fractional_index(T::Type{<:AbstractFloat},
                          pos::AbstractArray{<:Real},
                          grd::AbstractRange)
    # Use the central position of the grid to minimize rounding errors.
    c = (convert(T, first(grd)) + convert(T, last(grd)))/2
    q = 1/convert(T, step(grd))
    r = convert(T, 1 + length(grd))/2
    return i -> q*(convert(T, pos[i]) - c) + r
end

function fractional_index(T::Type{<:AbstractFloat},
                          pos::AbstractArray{<:Real})
    return i -> T(pos[i])
end

#------------------------------------------------------------------------------

"""
    SparseUnidimensionalInterpolator{T<:AbstractFloat,S,D} <: AbstractSparseInterpolator{T}
* `T` is the floating-point type of the coefficients,
* `S` is the size of the kernel
  (number of nodes to combine for a single interpolator)
* `D` is the dimension of interpolation.
"""
struct SparseUnidimensionalInterpolator{T<:AbstractFloat,S,D} <: AbstractSparseInterpolator{T}
    nrows::Int     # number of rows
    ncols::Int     # number of columns
    C::Vector{T}   # coefficients along the dimension of interpolation
    J::Vector{Int} # columns indices along the dimension of interpolation
end

(A::SparseUnidimensionalInterpolator)(x) = apply(A, x)

interp_dim(::SparseUnidimensionalInterpolator{T,S,D}) where {T,S,D} = D

coefficients(A::SparseUnidimensionalInterpolator) = A.C
columns(A::SparseUnidimensionalInterpolator) = A.J
size(A::SparseUnidimensionalInterpolator) = (A.nrows, A.ncols)
size(A::SparseUnidimensionalInterpolator, i::Integer) =
    (i == 1 ? A.nrows :
     i == 2 ? A.ncols : error("out of bounds dimension"))

"""
    SparseUnidimensionalInterpolator{T=eltype(ker)}(ker, d, pos, grd)

yields a linear mapping which interpolates the `d`-th dimension of an array
with kernel `ker` at positions `pos` along the dimension of interpolation `d`
and assuming the input array has grid coordinates `grd` along the the `d`-th
dimension of interpolation.  Argument `pos` is a vector of positions, argument
`grd` may be a range or the length of the dimension of interpolation.  Optional
parameter `T` is the floating-point type of the coefficients of the operator.

This kind of interpolator is suitable for separable multi-dimensional
interpolation with precomputed interpolation coefficients.  Having precomputed
coefficients is mostly interesting when the operator is to be applied multiple
times (for instance in iterative methods).  Otherwise, separable operators
which compute the coefficients *on the fly* may be preferable.

A combination of instances of `SparseUnidimensionalInterpolator` can be built
to achieve sperable multi-dimensional interpolation.  For example:

    using LinearInterpolators
    ker = CatmullRomSpline()
    n1, n2 = 70, 50
    x1 = linspace(1, 70, 201)
    x2 = linspace(1, 50, 201)
    A1 = SparseUnidimensionalInterpolator(ker, 1, x1, 1:n1)
    A2 = SparseUnidimensionalInterpolator(ker, 2, x2, 1:n2)
    A = A1*A2

"""
SparseUnidimensionalInterpolator(ker::Kernel{T}, args...) where {T<:AbstractFloat} =
    SparseUnidimensionalInterpolator{T}(ker, args...)

@deprecate SparseUnidimensionalInterpolator(T::Type{<:AbstractFloat}, ker::Kernel, args...) SparseUnidimensionalInterpolator{T}(ker, args...)

SparseUnidimensionalInterpolator{T}(ker::Kernel, args...) where {T<:AbstractFloat} =
    SparseUnidimensionalInterpolator{T}(T(ker), args...)

function SparseUnidimensionalInterpolator{T}(ker::Kernel{T},
                                             d::Integer,
                                             pos::AbstractVector{<:Real},
                                             len::Integer) where {T<:AbstractFloat}
    len ≥ 1 || throw(ArgumentError("invalid dimension length"))
    return SparseUnidimensionalInterpolator{T}(ker, d, pos, 1:Int(len))
end

# FIXME: not type-stable
function SparseUnidimensionalInterpolator{T}(ker::Kernel{T,S},
                                             d::Integer,
                                             pos::AbstractVector{<:Real},
                                             grd::AbstractRange
                                             ) where {T<:AbstractFloat,S}
     SparseUnidimensionalInterpolator{T,S,Int(d)}(ker, pos, grd)
end

function SparseUnidimensionalInterpolator{T,S,D}(ker::Kernel{T,S},
                                                 pos::AbstractVector{<:Real},
                                                 grd::AbstractRange
                                                 ) where {D,T<:AbstractFloat,S}
    isa(D, Int) || throw(ArgumentError("invalid type for dimension of interpolation"))
    D ≥ 1 || throw(ArgumentError("invalid dimension of interpolation"))
    nrows = length(pos)
    ncols = length(grd)
    C, J = _sparsecoefs(CartesianIndices((nrows,)), ncols, ker,
                        fractional_index(T, pos, grd))
    return SparseUnidimensionalInterpolator{T,S,D}(nrows, ncols, C, J)
end

function vcreate(::Type{Direct},
                 A::SparseUnidimensionalInterpolator,
                 x::AbstractArray,
                 scratch::Bool=false)
    nrows, ncols = size(A)
    return _vcreate(nrows, ncols, A, x)
end

function vcreate(::Type{Adjoint},
                 A::SparseUnidimensionalInterpolator,
                 x::AbstractArray,
                 scratch::Bool=false)
    nrows, ncols = size(A)
    return _vcreate(ncols, nrows, A, x)
end

function _vcreate(ny::Int, nx::Int,
                  A::SparseUnidimensionalInterpolator{Ta,S,D},
                  x::AbstractArray{Tx,N}) where {Ta,Tx<:Real,S,D,N}
    xdims = size(x)
    1 ≤ D ≤ N ||
        throw(DimensionMismatch("out of range dimension of interpolation"))
    xdims[D] == nx ||
        throw(DimensionMismatch("dimension $D of `x` must be $nx"))
    Ty = float(promote_type(Ta, Tx))
    ydims = [(d == D ? ny : xdims[d]) for d in 1:N]
    return Array{Ty,N}(undef, ydims...)
end

function apply!(α::Real, ::Type{Direct},
                A::SparseUnidimensionalInterpolator{Ta,S,D},
                x::AbstractArray{Tx,N},
                scratch::Bool,
                β::Real,
                y::AbstractArray{Ty,N}) where {Ta<:AbstractFloat,
                                               Tx<:Real,
                                               Ty<:AbstractFloat,S,D,N}
    # Check arguments.
    _check(A, N)
    xdims = size(x)
    ydims = size(y)
    nrows, ncols = size(A)
    xdims[D] == ncols ||
        throw(DimensionMismatch("dimension $D of `x` must be $ncols"))
    ydims[D] == nrows ||
        throw(DimensionMismatch("dimension $D of `y` must be $nrows"))
    for k in 1:N
        k == D || xdims[k] == ydims[k] ||
            throw(DimensionMismatch("`x` and `y` have incompatible dimensions"))
    end

    # Apply operator.
    if α == 0
        vscale!(y, β)
    else
        C = coefficients(A)
        J = columns(A)
        I_head = CartesianIndices(xdims[1:D-1])
        I_tail = CartesianIndices(xdims[D+1:N])
        T = promote_type(Ta,Tx)
        alpha = convert(T, α)
        if β == 0
            _apply_direct!(T, Val{S}, C, J, alpha, x, y,
                           I_head, nrows, I_tail)
        else
            beta = convert(Ty, β)
            _apply_direct!(T, Val{S}, C, J, alpha, x, beta, y,
                           I_head, nrows, I_tail)
        end
    end
    return y
end

function apply!(α::Real, ::Type{Adjoint},
                A::SparseUnidimensionalInterpolator{Ta,S,D},
                x::AbstractArray{Tx,N},
                scratch::Bool,
                β::Real,
                y::AbstractArray{Ty,N}) where {Ta<:AbstractFloat,
                                               Tx<:Real,
                                               Ty<:AbstractFloat,S,D,N}
    # Check arguments.
    _check(A, N)
    xdims = size(x)
    ydims = size(y)
    nrows, ncols = size(A)
    xdims[D] == nrows ||
        throw(DimensionMismatch("dimension $D of `x` must be $nrows"))
    ydims[D] == ncols ||
        throw(DimensionMismatch("dimension $D of `y` must be $ncols"))
    for k in 1:N
        k == D || xdims[k] == ydims[k] ||
            throw(DimensionMismatch("`x` and `y` have incompatible dimensions"))
    end

    # Apply adjoint operator.
    vscale!(y, β)
    if α != 0
        T = promote_type(Ta,Tx)
        _apply_adjoint!(Val{S}, coefficients(A), columns(A),
                        convert(T, α), x, y,
                        CartesianIndices(xdims[1:D-1]), nrows,
                        CartesianIndices(xdims[D+1:N]))
    end
    return y
end

# The 3 following private methods are needed to achieve type invariance and win
# a factor ~1000 in speed!  Also note the way the innermost loop is written
# with a constant range and an offset k0 which is updated; this is critical for
# saving a factor 2-3 in speed.
#
# The current version takes ~ 4ms (7 iterations of linear conjugate gradients)
# to fit a 77×77 array of weights interpolated by Catmull-Rom splines to
# approximate a 256×256 image.

function _apply_direct!(::Type{T},
                        ::Type{Val{S}},
                        C::Vector{<:AbstractFloat},
                        J::Vector{Int},
                        α::AbstractFloat,
                        x::AbstractArray{<:Real,N},
                        y::AbstractArray{<:AbstractFloat,N},
                        I_head::CartesianIndices{N_head},
                        len::Int,
                        I_tail::CartesianIndices{N_tail}
                        ) where {T<:AbstractFloat,S,N,N_tail,N_head}
    @assert N == N_tail + N_head + 1
    @inbounds for i_tail in I_tail
        for i_head in I_head
            k0 = 0
            for i in 1:len
                sum = zero(T)
                @simd for s in 1:S
                    k = k0 + s
                    sum += C[k]*x[i_head,J[k],i_tail]
                end
                y[i_head,i,i_tail] = α*sum
                k0 += S
            end
        end
    end
end

function _apply_direct!(::Type{T},
                        ::Type{Val{S}},
                        C::Vector{<:AbstractFloat},
                        J::Vector{Int},
                        α::AbstractFloat,
                        x::AbstractArray{<:Real,N},
                        β::AbstractFloat,
                        y::AbstractArray{<:AbstractFloat,N},
                        I_head::CartesianIndices{N_head},
                        len::Int,
                        I_tail::CartesianIndices{N_tail}
                        ) where {T<:AbstractFloat,S,N,N_tail,N_head}
    @assert N == N_tail + N_head + 1
    @inbounds for i_tail in I_tail
        for i_head in I_head
            k0 = 0
            for i in 1:len
                sum = zero(T)
                @simd for s in 1:S
                    k = k0 + s
                    sum += C[k]*x[i_head,J[k],i_tail]
                end
                y[i_head,i,i_tail] = α*sum + β*y[i_head,i,i_tail]
                k0 += S
            end
        end
    end
end

function _apply_adjoint!(::Type{Val{S}},
                         C::Vector{<:AbstractFloat},
                         J::Vector{Int},
                         α::AbstractFloat,
                         x::AbstractArray{<:Real,N},
                         y::AbstractArray{<:AbstractFloat,N},
                         I_head::CartesianIndices{N_head},
                         len::Int,
                         I_tail::CartesianIndices{N_tail}
                         ) where {S,N,N_tail,N_head}
    @assert N == N_tail + N_head + 1
    @inbounds for i_tail in I_tail
        for i_head in I_head
            k0 = 0
            for i in 1:len
                c = α*x[i_head,i,i_tail]
                @simd for s in 1:S
                    k = k0 + s
                    y[i_head,J[k],i_tail] += C[k]*c
                end
                k0 += S
            end
        end
    end
end

function _check(A::SparseUnidimensionalInterpolator{T,S,D},
                N::Int) where {T<:AbstractFloat,S,D}
    1 ≤ D ≤ N ||
        throw(DimensionMismatch("out of range dimension of interpolation"))
    nrows, ncols = size(A)
    nvals = S*nrows
    C = coefficients(A)
    J = columns(A)
    length(C) == nvals ||
        throw(DimensionMismatch("array of coefficients must have $nvals elements (has $(length(C)))"))
    length(J) == nvals ||
        throw(DimensionMismatch("array of indices must have $nvals elements (has $(length(C)))"))
    for k in eachindex(J)
        1 ≤ J[k] ≤ ncols || throw(ErrorException("out of bounds indice(s)"))
    end
end

end # module

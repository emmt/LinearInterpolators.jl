#
# sparse.jl -
#
# Implement sparse linear interpolator.
#

# All code is in a module to "hide" private methods.
module SparseInterpolators

export
    SparseInterpolator,
    SparseSeparableInterpolator

using InterpolationKernels
using InterpolationKernels: compute_offset_and_weights

using LinearAlgebra

using LazyAlgebra
using LazyAlgebra.Foundations
import LazyAlgebra: apply, apply!, vcreate, nrows, ncols, output_size, input_size

import Base: axes, eltype, size, getindex
using Base: OneTo, @propagate_inbounds
import SparseArrays: sparse, nonzeros, nnz

using ..LinearInterpolators
using ..LinearInterpolators:
    AbstractInterpolator,
    check_axes,
    check_size,
    compute_indices,
    sum_of_terms,
    to_axis
import ..LinearInterpolators:
    coefficients,
    columns,
    rows,
    fit,
    regularize,
    regularize!,
    dimension_mismatch

# A SparseInterpolator is stored in Compressed Sparse Row (CSR) format.
struct SparseInterpolator{T<:AbstractFloat,S,N} <: AbstractInterpolator{T}
    C::Vector{T}   # coefficients
    J::Vector{Int} # column indices
    nrows::Int     # equivalent number of rows
    ncols::Int     # equivalent number of columns
    dims::Dims{N}  # dimensions of result
    function SparseInterpolator{T,S,N}(C::Vector{T},
                                       J::Vector{Int},
                                       dims::Dims{N},
                                       ncols::Int) where {T,S,N}
        (isa(S, Int) && S ≥ 1) || bad_type_parameter(:S, S, Int)
        nrows = check_size(dims)
        nrows ≥ 1 || error("all dimensions must be ≥ 1")
        nvals = S*nrows # number of non-zero coefficients
        length(C) == nvals || error("array of coefficients has invalid length")
        length(J) == nvals || error("array of column indices has invalid length")
        new{T,S,N}(C, J, nrows, ncols, dims)
    end
end

# Interpolator can be used as a function.
(A::SparseInterpolator)(x::AbstractVector) = apply(A, x)

nrows(A::SparseInterpolator) = A.nrows
ncols(A::SparseInterpolator) = A.ncols
output_size(A::SparseInterpolator) = A.dims
input_size(A::SparseInterpolator) = (ncols(A),)
nonzeros(A::SparseInterpolator) = A.C
coefficients(A::SparseInterpolator) = nonzeros(A)
nnz(A::SparseInterpolator{T,S}) where {T,S} = S*nrows(A)
columns(A::SparseInterpolator) = A.J
function rows(A::SparseInterpolator{T,S,N}) where {T,S,N}
    I = Array{Int}(undef, nnz(A))
    k0 = 0
    @inbounds for i in 1:nrows(A)
        for k in k0+1:k0+S
            I[k] = i
        end
        k0 += S
    end
    return I
end

# Convert to a sparse matrix.
sparse(A::SparseInterpolator) =
    sparse(rows(A), columns(A), nonzeros(A), nrows(A), ncols(A))

"""
    A = SparseInterpolator{T=eltype(ker)}(ker, pos, grd, B=Flat)

yields a sparse linear interpolator `A` suitable for interpolating with kernel
`ker` at positions `pos` a function sampled on the grid `grd`.  Optional
parameter `T` is the floating-point type of the coefficients of the operator
`A`.

Then `y = apply(A, x)`, `y = A(x)`, or `y = A*x` yield the result of the
interpolation of array `x` with `A`.  The shape of `y` is the same as that of
`pos`.  Formally, this amounts to computing (forgetting boundary conditions):

    y[i] = sum_j ker((pos[i] - grd[j])/step(grd))*x[j]

with `step(grd)` the (constant) step size between the nodes of the grid `grd`
and `grd[j]` the `j`-th position of the grid.

The grid `grd` can be specified as a step-range, like `a:b:c`, or can just be
the length, say `len`, of the array to interpolate; in this latter case, the
grid is assumed to be `1:len`.

Optional argument `B` is to specify which boundary conditions to apply for
extrapolating values.

Call `eltype(A)` to query the type of the coefficients of the sparse
interpolator `A`.

Call `sparse(A)` (from package `SparseArrays`) to convert the sparse
interpolator `A` into a sparse array.

"""
SparseInterpolator(ker::Kernel{T}, args...) where {T<:AbstractFloat} =
    SparseInterpolator{T}(ker, args...)

SparseInterpolator{T}(ker::Kernel, args...) where {T<:AbstractFloat} =
    SparseInterpolator{T}(Kernel{T}(ker), args...)

function SparseInterpolator{T}(ker::Kernel{T},
                               pos::AbstractArray{<:Real},
                               grd::AbstractRange,
                               B::Type{<:Boundaries} = Flat
                               ) where {T<:AbstractFloat}
    SparseInterpolator{T}(ker, FractionalIndex{T}(pos, grd),
                          CartesianIndices(axes(pos)), length(grd), B)
end

function SparseInterpolator{T}(ker::Kernel{T},
                               pos::AbstractArray{<:Real},
                               len::Integer,
                               B::Type{<:Boundaries} = Flat
                               ) where {T<:AbstractFloat}
    SparseInterpolator{T}(ker, fractional_index(T, pos),
                          CartesianIndices(axes(pos)), len, B)
end

function SparseInterpolator{T}(ker::Kernel{T,S},
                               f::Function,
                               R::CartesianIndices{N},
                               ncols::Integer,
                               B::Type{<:Boundaries} = Flat
                               ) where {T<:AbstractFloat,S,N}
    C, J = sparse_fields(R, Int(ncols), ker, f, B)
    return SparseInterpolator{T,S,N}(C, J, size(R), ncols)
end

# Compute the fields of a sparse interpolator.
@generated function sparse_fields(I,
                                  ncols::Int,
                                  ker::Kernel{T,S},
                                  f::Function,
                                  B::Type{<:Boundaries}) where {T,S,N}
    store_indices = [:(J[k+$s] = inds[$s]) for s in 1:S]
    store_weights = [:(C[k+$s] = wgts[$s]) for s in 1:S]
    quote
        bounds = B(ker, ncols)
        nvals = S*length(I)
        J = Array{Int}(undef, nvals)
        C = Array{T}(undef, nvals)
        k = 0
        @inbounds for i in I
            x = convert(T, f(i))
            off, wgts = compute_offset_and_weights(ker, x)
            inds = compute_indices(bounds, off)
            $(store_weights...)
            $(store_indices...)
            k += S
        end
        return C, J
    end
end

function check(A::SparseInterpolator{<:AbstractFloat,S,N};
               full::Bool=false) where {S,N}
    @noinline throw_corrupted(args...) =
        error("corrupted sparse interpolator (", args..., ")")
    nvals = nnz(A) # number of non-zero coefficients
    C = coefficients(A)
    length(C) == nvals || throw_corrupted("bad number of coefficients")
    J = columns(A)
    length(J) == nvals || throw_corrupted("bad number of indices")
    if full
        flag = false
        @inbounds @simd for k in 1:nvals
            flag |= (J[k] < 1)|(J[k] > ncols)
        end
        flag && throw_corrupted("out of bound indices")
    end
end

function vcreate(::Type{LazyAlgebra.Direct},
                 A::SparseInterpolator{T,S,N},
                 x::AbstractVector{T},
                 scratch::Bool=false) where {T,S,N}
    return Array{T}(undef, output_size(A))
end

function vcreate(::Type{LazyAlgebra.Adjoint},
                 A::SparseInterpolator{T,S,N},
                 x::AbstractArray{T,N},
                 scratch::Bool=false) where {T,S,N}
    return Array{T}(undef, input_size(A))
end

@inline @generated function interp_direct(A::SparseInterpolator{<:AbstractFloat,S},
                                          x::AbstractVector,
                                          off::Int) where {S}
    ex = sum_of_terms([:(A.C[off+$k]*x[A.J[off+$k]]) for k in 1:S])
    quote
        $(Expr(:meta, :inline))
        return $(ex)
    end
end

# FIXME: optimize for other values of α and β
function apply!(α::Number,
                ::Type{LazyAlgebra.Direct},
                A::SparseInterpolator{<:AbstractFloat,S,N},
                x::AbstractVector{<:Real},
                scratch::Bool,
                β::Number,
                y::AbstractArray{<:AbstractFloat,N}) where {S,N}
    check(A)
    check_axes(x, input_size(A)) ||
        dimension_mismatch("argument `x` has incompatible indices")
    check_axes(y, output_size(A)) ||
        dimension_mismatch("argument `y` has incompatible indices")
    if α == 0
        vscale!(y, β)
    else
        T = promote_type(eltype(A), eltype(x))
        alpha = promote_multiplier(α, T)
        nrows, ncols = A.nrows, A.ncols
        C, J = nonzeros(A), columns(A)
        k0 = 0
        if β == 0
            @inbounds for i in 1:nrows
                #y[i] = alpha*interp_direct(A, x, k0)
                #sum = zero(T)
                #@simd for k in k0+1:k0+S # FIXME: use @generated to unroll
                #    j = J[k]
                #    sum += C[k]*x[j]
                #end
                #y[i] = alpha*sum
                sum = zero(T)
                for k in k0+1:k0+S # FIXME: use @generated to unroll
                    j = J[k]
                    sum += C[k]*x[j]
                end
                y[i] = alpha*sum
                k0 += S
            end
        else
            beta = promote_multiplier(β, y)
            @inbounds for i in 1:nrows
                sum = zero(T)
                @simd for k in k0+1:k0+S # FIXME: use @generated to unroll
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

function apply!(α::Number,
                ::Type{LazyAlgebra.Adjoint},
                A::SparseInterpolator{<:AbstractFloat,S,N},
                x::AbstractArray{<:Real,N},
                scratch::Bool,
                β::Number,
                y::AbstractVector{<:AbstractFloat}) where {S,N}
    check(A)
    check_axes(x, output_size(A)) ||
        dimension_mismatch("argument `x` has incompatible indices")
    check_axes(y, input_size(A)) ||
        dimension_mismatch("argument `y` has incompatible indices")
    vscale!(y, β)
    if α != 0
        T = promote_type(eltype(A), eltype(x))
        alpha = promote_multiplier(α, T)
        nrows, ncols = A.nrows, A.ncols
        C, J = nonzeros(A), columns(A)
        k0 = 0
        @inbounds for i in 1:nrows
            c = alpha*x[i]
            if c != 0
                @simd for k in k0+1:k0+S # FIXME: use @generated to unroll
                    j = J[k]
                    y[j] += C[k]*c
                end
            end
            k0 += S
        end
    end
    return y
end

"""
    lhs_matrix(M) -> A

yields the left-hand-side (LHS) matrix of the normal equations `A = M'⋅M` for a
model matrix `M`.  Independent nonnegative weights `w` may be specified:

    lhs_matrix(M, w) -> A

yields the matrix `A = M'⋅W⋅M` with `W = Diag(w)`.

"""
function lhs_matrix(M::SparseInterpolator{T}) where {T}
    ncols = M.ncols
    return lhs_matrix!(Array{T}(undef, ncols, ncols), M)
end
function lhs_matrix(M::SparseInterpolator{<:AbstractFloat,S,N},
                    w::AbstractArray{<:Real,N}) where {S,N}
    T = promote_type(eltype(M), eltype(w))
    ncols = M.ncols
    return lhs_matrix!(Array{T}(undef, ncols, ncols), M, w)
end

"""
    lhs_matrix!(A, M) -> A

overwrites `A` with the left-hand-side (LHS) matrix of the normal equations
`A = M'⋅M` for a sparse linear interpolator `M`.  Independent nonnegative
weights `w` may be specified:

    lhs_matrix!(A, M, w) -> A

yields the matrix `A = M'⋅W⋅M` with `W = Diag(w)`.

""" lhs_matrix!

# FIXME: compute only one of the triangular parts and return Hermitian result
function lhs_matrix!(A::AbstractArray{T,2},
                     M::SparseInterpolator{T,S,N}) where {T,S,N}
    nrows, ncols = M.nrows, M.ncols
    @assert size(A) == (ncols, ncols)
    fill!(A, zero(T))
    C, J = nonzeros(M), columns(M)
    k0 = 0
    @assert length(J) == length(C)
    @inbounds for i in 1:nrows
        K = k0+1:k0+S
        for k in K
            1 ≤ J[k] ≤ ncols || error("corrupted interpolator table")
        end
        for k1 in K
            j1, c1 = J[k1], C[k1]
            @simd for k2 in K
                j2, c2 = J[k2], C[k2]
                A[j1,j2] += c1*c2
            end
        end
        k0 += S
    end
    return A
end

# FIXME: compute only one of the triangular parts and return Hermitian result
function lhs_matrix!(A::AbstractArray{T,2},
                     M::SparseInterpolator{T,S,N},
                     wgt::AbstractArray{T,N}) where {T,S,N}
    nrows, ncols = M.nrows, M.ncols
    @assert size(A) == (ncols, ncols)
    @assert size(wgt) == output_size(M)
    fill!(A, zero(T))
    C, J = nonzeros(M), columns(M)
    k0 = 0
    @assert length(J) == length(C)
    @inbounds for i in 1:nrows
        K = k0+1:k0+S
        for k in K
            1 ≤ J[k] ≤ ncols || error("corrupted interpolator table")
        end
        w = wgt[i]
        for k1 in K
            j1 = J[k1]
            w_c1 = w*C[k1]
            @simd for k2 in K
                j2 = J[k2]
                A[j1,j2] += C[k2]*w_c1
            end
        end
        k0 += S
    end
    return A
end

"""
    rhs_vector(M, y) -> b

yields the right-hand-side (RHS) vector of the normal equations `b = M'⋅y` for
a sparse linear interpolator `M` and data `y`.  Independent nonnegative weights
`w` may be specified:

    rhs_vector(M, y, w) -> b

yields the vector `b = M'⋅W⋅y` with `W = Diag(w)`.

"""
function rhs_vector(M::SparseInterpolator{T,S,N},
                    y::AbstractArray{T,N}) where {T,S,N}
    return M'*y
end

function rhs_vector(M::SparseInterpolator{T,S,N},
                    y::AbstractArray{T,N},
                    w::AbstractArray{T,N}) where {T,S,N}
    return M'*(w.*y)
end

# Default regularization levels.
const RGL_EPS = 1e-9
const RGL_MU = 0.0

"""
    fit(M, y [, w]; epsilon=1e-9, mu=0.0) -> x

performs a linear fit of `y` by the model `M*x` with `M` a linear interpolator.
The returned value `x` minimizes:

    sum(w.*(M*x - y).^2)

where `w` are given weights.  If `w` is not specified, all weights are assumed
to be equal to one; otherwise `w` must be an array of nonnegative values and of
same size as `y`.

Keywords `epsilon` and `mu` may be specified to regularize the solution and
minimize:

    sum(w.*(M*x - y).^2) + rho*(epsilon*norm(x)^2 + mu*norm(D*x)^2)

where `D` is a finite difference operator, `rho` is the maximum diagonal
element of `M'*Diag(w)*M` and `norm` is the Euclidean norm.

"""
function fit(M::SparseInterpolator{T,S,N},
             y::AbstractArray{T,N},
             w::AbstractArray{T,N};
             kwds...) where {T,S,N}
    A = lhs_matrix(M, w)    # Compute LHS matrix A = M'*W*M with W = diag(w).
    b = rhs_vector(M, y, w) # Compute RHS vector b = M'*W*y with W = Diag(w).
    return solve_regularized_normal_equations!(A, b; kwds...)
end

function fit(M::SparseInterpolator{T,S,N},
             y::AbstractArray{T,N};
             kwds...) where {T,S,N}
    A = lhs_matrix(M)    # Compute LHS matrix A = M'*W*M with W = diag(w).
    b = rhs_vector(M, y) # Compute RHS vector b = M'*y.
    return solve_regularized_normal_equations!(A, b; kwds...)
end

"""
    solve_normal_equations!(A, b)

overwrites `b` with the solution `x = A\b` of the normal equations `A*x = b`
where `A` is a symmetric positive definite matrix.  Unless `A` is already a
Cholesky decomposition, its contents is overwritten by its Cholesky
decomposition.

"""
function solve_normal_equations!(A::AbstractMatrix{<:AbstractFloat},
                                 b::AbstractVector{<:AbstractFloat})
    return solve_normal_equations!(cholesky!(Hermitian(A), Val(false)), b)
end

function solve_normal_equations!(A::Cholesky{<:AbstractFloat,<:AbstractMatrix},
                                 b::AbstractVector{<:AbstractFloat})
    return ldiv!(A, b)
end

"""
    solve_regularized_normal_equations!(A, b; epsilon=1e-9, mu=0.0)

overwrites `b` with the solution `x = R\b` of the regularized normal equations
`R*x = b` with:

    R = A + ρ*(ϵ*I + μ*D'*D)

where `A` is a symmetric positive definite matrix whose contents is overwritten
by the Cholesky decomposition of `R`.  The scaling parameter `ρ > 0` is the
maximum diagonal element of `A`.  The regularization parameters `ϵ ≥ 0` and `µ
≥ 0` are respectively specified by the keywords `epsilon` and `mu`.

"""
function solve_regularized_normal_equations!(A::AbstractMatrix{<:AbstractFloat},
                                             b::AbstractVector{<:AbstractFloat};
                                             epsilon::Real = RGL_EPS,
                                             mu::Real = RGL_MU)
    regularize!(A, epsilon, mu)
    return solve_normal_equations!(A, b)
end

"""
    regularize(A, ϵ, μ) -> R

regularizes the symmetric matrix `A` to produce the matrix:

    R = A + ρ*(ϵ*I + μ*D'*D)

where `I` is the identity, `D` is a finite difference operator and `ρ` is the
maximum diagonal element of `A`.

"""
regularize(A::AbstractArray{T,2}, args...) where {T<:AbstractFloat} =
    regularize!(copyto!(Array{T}(undef, size(A)), A), args...)

"""
    regularize!(A, ϵ, μ) -> A

stores the regularized matrix in `A` (and returns it).  This is the in-place
version of [`LinearInterpolators.SparseInterpolators.regularize`].

"""
function regularize!(A::AbstractArray{T,2},
                     eps::Real = RGL_EPS,
                     mu::Real = RGL_MU) where {T<:AbstractFloat}
    regularize!(A, T(eps), T(mu))
end

function regularize!(A::AbstractArray{T,2},
                     eps::T, mu::T) where {T<:AbstractFloat}
    local rho::T
    @assert eps ≥ zero(T)
    @assert mu ≥ zero(T)
    @assert size(A,1) == size(A,2)
    n = size(A,1)
    if eps > zero(T) || mu > zero(T)
        rho = A[1,1]
        for j in 2:n
            d = A[j,j]
            rho = max(rho, d)
        end
        rho > zero(T) || error("we have a problem!")
    end
    if eps > zero(T)
        q = eps*rho
        for j in 1:n
            A[j,j] += q
        end
    end
    if mu > zero(T)
        q = rho*mu
        if n ≥ 2
            r = q + q
            A[1,1] += q
            A[2,1] -= q
            for i in 2:n-1
                A[i-1,i] -= q
                A[i,  i] += r
                A[i+1,i] -= q
            end
            A[n-1,n] -= q
            A[n,  n] += q
        elseif n == 1
            A[1,1] += q
        end
    end
    return A
end


# A fractional index is a lazy map to convert a position into a fractional
# index along a dimension.
abstract type FractionalIndex{T<:AbstractFloat,N,
                              A<:AbstractArray{<:Real,N}} <: AbstractArray{T,N} end

struct FractionalIndexSimple{T<:AbstractFloat,N,
                             A<:AbstractArray{<:Real,N}} <: FractionalIndex{T,N,A}
    pos::A
end

struct FractionalIndexOffset{T<:AbstractFloat,N,
                             A<:AbstractArray{<:Real,N}} <: FractionalIndex{T,N,A}
    pos::A
    off::T
end

struct FractionalIndexAffine{T<:AbstractFloat,N,
                             A<:AbstractArray{<:Real,N}} <: FractionalIndex{T,N,A}
    # Fractional index is computed as `(pos[i] - c)*q + r` where `c` is the
    # mean index value in the target dimension to reduce rounding errors.
    pos::A
    c::T
    q::T
    r::T
end

Base.IndexStyle(::Type{<:FractionalIndex{T,N,A}}) where {T,N,A} = IndexStyle(A)

@inline @propagate_inbounds getindex(A::FractionalIndexSimple{T}, I...) where {T} =
    T(A.pos[I...])

@inline @propagate_inbounds getindex(A::FractionalIndexOffset{T}, I...) where {T} =
    T(A.pos[I...]) + A.off

@inline @propagate_inbounds getindex(A::FractionalIndexAffine{T}, I...) where {T} =
    (T(A.pos[I...]) - A.c)*A.q + A.r


function FractionalIndex{T}(pos::A,
                            len::Integer) where {T<:AbstractFloat,N,
                                                 A<:AbstractArray{<:Real,N}}
    return FractionalIndexSimple{T,N,A}(pos)
end

function FractionalIndex{T}(pos::A,
                            grd::OneTo) where {T<:AbstractFloat,N,
                                               A<:AbstractArray{<:Real,N}}
    return FractionalIndexSimple{T,N,A}(pos)
end

function FractionalIndex{T}(pos::AbstractArray{<:Real},
                            grd::AbstractUnitRange) where {T<:AbstractFloat}
    #
    # The fractional index is defined so that the argument of the kernel
    # function be given by:
    #
    #     (pos[i] - grd[j])/step(grd) = fractional_index[i] - j
    #
    # For a unit step grid `grd`, the j-th grid coordinate is given by:
    #
    #     grd[j] = first[j] + (j - firstindex(grd))
    #
    # Thus the fractional index is given by the expression:
    #
    #     fractional_index[i] = pos[i] + off
    #
    # with `off = firstindex(grd) - first(grd)`.
    #
    return FractionalIndexOffset{T,N,A}(pos, firstindex(grd) - first(grd))
end

function FractionalIndex{T}(pos::A,
                            grd::AbstractRange{<:Real}) where {T<:AbstractFloat,N,
                                                               A<:AbstractArray{<:Real,N}}
    #
    # The fractional index is defined so that the argument of the kernel
    # function be given by:
    #
    #     (pos[i] - grd[j])/step(grd) = fractional_index[i] - j
    #
    # For a constant step grid `grd`, the j-th grid coordinate is given by:
    #
    #     grd[j] = first[j] + (j - firstindex(grd))*step(grd)
    #
    # hence and the fractional index writes:
    #
    #     fractional_index[i] = (pos[i] - first(grd))/step(grd) + firstindex(grd)
    #
    # For fast computaions, we want the fractional index to be expressed as:
    #
    #     fractional_index[i] = (pos[i] - c)*q + r
    #
    # with `q = 1/step(grd)` while `c` and `r` must be such that:
    #
    #     r - c*q = firstindex(grd) - first(grd)/step(grd)
    #
    # holds.  Taking `c` as the mean grid coordinate to minimize rounding
    # errors yields:
    #
    #     c = (first(grd) + last(grd))/2
    #     r = firstindex(grd) + q*(c - first(grd))
    #       = firstindex(grd) + q*(last(grd) - first(grd))/2
    #
    grd_first = T(first(grd))
    grd_last = T(last(grd))
    grd_step = T(step(grd))
    c = (grd_first + grd_last)/2
    q = 1/grd_step
    r = firstindex(grd) + q*(grd_last - grd_first)/2
    return FractionalIndexSimple{T,N,A}(pos, c, q, r)
end

#------------------------------------------------------------------------------

# Parameter `T` is the floating-point type of the coefficients, parameter `S`
# is the size of the kernel (number of nodes to combine for a single
# interpolator) and parameter `D` is the dimension of interpolation.
struct SparseSeparableInterpolator{D,T<:AbstractFloat,S} <: AbstractInterpolator{T}
    nrows::Int     # number of rows
    ncols::Int     # number of columns
    C::Vector{T}   # coefficients along the dimension of interpolation
    J::Vector{Int} # columns indices along the dimension of interpolation
    function SparseSeparableInterpolator{D,T,S}(nrows::Integer,
                                                ncols::Integer,
                                                C::AbstractVector{<:Real},
                                                J::AbstractVector{<:Integer}
                                                ) where {D,T<:AbstractFloat,S}
        nrows ≥ 1 || error("invalid number of rows")
        ncols ≥ 1 || error("invalid number of columns")
        (isa(S, Int) && S ≥ 1) || bad_type_parameter(:S, S, Int)
        (isa(D, Int) && D ≥ 1) || bad_type_parameter(:D, D, Int)
        nvals = S*nrows # number of non-zero coefficients
        axes(C,1) == 1:nvals || error("invalid indices for vector of coefficients")
        axes(J,1) == 1:nvals || error("invalid indices for vector of column indices")
        @inbounds for k in eachindex(J)
            1 ≤ j ≤ ncols || error("out of bounds column index")
        end
        return new{D,T,S}(nrows, ncols, C, J)
    end
end

# Interpolator can be used as a function.
(A::SparseSeparableInterpolator)(x) = apply(A, x)

dimension_of_interest(x::SparseSeparableInterpolator) = dimension_of_interest(typeof(x))
dimension_of_interest(::Type{<:SparseSeparableInterpolator{D}}) where {D} = D

nrows(A::SparseSeparableInterpolator) = A.nrows
ncols(A::SparseSeparableInterpolator) = A.ncols
nonzeros(A::SparseSeparableInterpolator) = A.C
nnz(A::SparseSeparableInterpolator{D,T,S}) where {D,T,S} = S*nrows(A)
coefficients(A::SparseSeparableInterpolator) = nonzeros(A)
columns(A::SparseSeparableInterpolator) = A.J
size(A::SparseSeparableInterpolator) = (nrows(A), ncols(A)) # FIXME: useful?
size(A::SparseSeparableInterpolator, i::Integer) =          # FIXME: useful?
    (i == 1 ? nrows(A) :
     i == 2 ? ncols(A) : error("out of bounds dimension"))

"""
    A = SparseSeparableInterpolator{D,T=eltype(C),S=div(length(C),m)}(m, n, C, J)

yields a linear mapping `A` which *"interpolates"* the `D`-th dimension of an
array of length `n` along that dimension, to produce an array of length `m`
along that dimension.  Arguments `C` and `J` are vectors of `S*m` entries
specifying the interpolation coefficients and *"column"* indices along the
`D`-th dimension.  The indices in `J` must all be in the range `1:n`.  Optional
parameters `T` and `S` are the floating-point type of the interpolation
coefficients and the number of neighbors involved in the interpolation at a
given position.  Applying `A` to an array `x` yields an array `y` such that:

    y[i_pre,i,i_post] = sum_{k ∈ 1:S} C[S*(i-1)+k]*x[i_pre,J[S*(i-1)+k],i_post]

where `i ∈ 1:m` is the output index along the dimension of interpolation while
`i_pre` and `i_post` are multi-dimensional indices along the leading and
trailing dimensions.

Easier to use constructor is provided by:

    SparseSeparableInterpolator{D,T=eltype(ker)}(ker, pos, grd)

which yields a linear mapping to interpolate the `D`-th dimension of an array
with kernel `ker` at positions `pos` along the dimension of interpolation `D`
and assuming the input array has grid coordinates `grd` along the the `D`-th
dimension of interpolation.  Argument `pos` is a vector of positions, argument
`grd` may be a range or the length of the dimension of interpolation.  Optional
parameter `T` is the floating-point type of the coefficients of the operator.

This kind of interpolator is suitable for separable multi-dimensional
interpolation with precomputed interpolation coefficients.  Having precomputed
coefficients is mostly interesting when the operator is to be applied multiple
times (for instance in iterative methods).  Otherwise, a separable operator
which computes the coefficients *on the fly* may be preferable
(see [`LazySeparableInterpolator`](@ref).

A combination of instances of `SparseSeparableInterpolator` can be built to
achieve sperable multi-dimensional interpolation.  For example:

    using LinearInterpolators
    ker = CatmullRomSpline()
    n1, n2 = 70, 50
    x1 = range(1, 70; length=201)
    x2 = range(1, 50; length=201)
    A1 = SparseSeparableInterpolator(ker, 1, x1, 1:n1)
    A2 = SparseSeparableInterpolator(ker, 2, x2, 1:n2)
    A = A1*A2

""" SparseSeparableInterpolator

function SparseSeparableInterpolator{D}(m::Integer,
                                        n::Integer,
                                        C::AbstractVector{<:AbstractFloat},
                                        J::AbstractVector{<:Integer}
                                        ) where {D}
    T = eltype(C)
    return SparseSeparableInterpolator{D,T}(m, n, C, J)
end

function SparseSeparableInterpolator{D,T}(m::Integer,
                                          n::Integer,
                                          C::AbstractVector{<:Real},
                                          J::AbstractVector{<:Integer}
                                          ) where {D,T<:AbstractFloat}
    S = convert(Int, div(length(C), m))
    return SparseSeparableInterpolator{D,T,S}(m, n, C, J)
end

SparseSeparableInterpolator{D}(ker::Kernel{T}, args...) where {D,T<:AbstractFloat} =
    SparseSeparableInterpolator{D,T}(ker, args...)

SparseSeparableInterpolator{D,T}(ker::Kernel, args...) where {D,T<:AbstractFloat} =
    SparseSeparableInterpolator{D,T}(T(ker), args...)

function SparseSeparableInterpolator{D,T}(ker::Kernel{T},
                                          pos::AbstractVector{<:Real},
                                          len::Integer) where {D,T<:AbstractFloat}
    len ≥ 1 || throw(ArgumentError("invalid dimension length"))
    return SparseSeparableInterpolator{D,T}(ker, d, pos, 1:Int(len))
end

# FIXME: `grd` -> `axis`, using `to_axis`, to allow for interpolating offset
# arrays or a (contiguous) sub-part of an array.
function SparseSeparableInterpolator{D,T}(ker::Kernel{T,S},
                                          pos::AbstractVector{<:Real},
                                          grd::AbstractRange) where {D,T<:AbstractFloat,S}
     SparseSeparableInterpolator{D,T,S}(ker, pos, grd)
end

function SparseSeparableInterpolator{D,T,S}(ker::Kernel{T,S},
                                            pos::AbstractVector{<:Real},
                                            grd::AbstractRange) where {D,T<:AbstractFloat,S}
    nrows = length(pos)
    ncols = length(grd)
    C, J = sparse_fields(CartesianIndices((nrows,)), ncols, ker,
                         fractional_index(T, pos, grd))
    return SparseSeparableInterpolator{D,T,S}(nrows, ncols, C, J)
end

function vcreate(::Type{LazyAlgebra.Direct},
                 A::SparseSeparableInterpolator,
                 x::AbstractArray,
                 scratch::Bool=false)
    return _vcreate(nrows(A), ncols(A), A, x)
end

function vcreate(::Type{LazyAlgebra.Adjoint},
                 A::SparseSeparableInterpolator,
                 x::AbstractArray,
                 scratch::Bool=false)
    return _vcreate(ncols(A), nrows(A), A, x)
end

function _vcreate(ny::Int, nx::Int,
                  A::SparseSeparableInterpolator{D,Ta,S},
                  x::AbstractArray{Tx,N}) where {Ta,Tx<:Real,S,D,N}
    xdims = size(x)
    1 ≤ D ≤ N ||
        throw(DimensionMismatch("out of range dimension of interpolation"))
    xdims[D] == nx ||
        throw(DimensionMismatch("dimension $D of argument must be $nx"))
    Ty = float(promote_type(Ta, Tx))
    ydims = [(d == D ? ny : xdims[d]) for d in 1:N] # FIXME: ntuple?
    return Array{Ty,N}(undef, ydims...)
end

function apply!(α::Real,
                ::Type{LazyAlgebra.Direct},
                A::SparseSeparableInterpolator{D,Ta,S},
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
        C = nonzeros(A)
        J = columns(A)
        I_pre = CartesianIndices(xdims[1:D-1])
        I_post = CartesianIndices(xdims[D+1:N])
        T = promote_type(Ta,Tx)
        alpha = convert(T, α)
        if β == 0
            _apply_direct!(T, Val{S}, C, J, alpha, x, y,
                           I_pre, nrows, I_post)
        else
            beta = convert(Ty, β)
            _apply_direct!(T, Val{S}, C, J, alpha, x, beta, y,
                           I_pre, nrows, I_post)
        end
    end
    return y
end

function apply!(α::Real,
                ::Type{LazyAlgebra.Adjoint},
                A::SparseSeparableInterpolator{D,Ta,S},
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
        _apply_adjoint!(Val{S}, nonzeros(A), columns(A),
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
                        I_pre::CartesianIndices{N_pre},
                        len::Int,
                        I_post::CartesianIndices{N_post}
                        ) where {T<:AbstractFloat,S,N,N_post,N_pre}
    @assert N == N_post + N_pre + 1
    @inbounds for i_post in I_post
        for i_pre in I_pre
            k0 = 0
            for i in 1:len
                sum = zero(T)
                @simd for s in 1:S
                    k = k0 + s
                    sum += C[k]*x[i_pre,J[k],i_post]
                end
                y[i_pre,i,i_post] = α*sum
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
                        I_pre::CartesianIndices{N_pre},
                        len::Int,
                        I_post::CartesianIndices{N_post}
                        ) where {T<:AbstractFloat,S,N,N_post,N_pre}
    @assert N == N_post + N_pre + 1
    @inbounds for i_post in I_post
        for i_pre in I_pre
            k0 = 0
            for i in 1:len
                sum = zero(T)
                @simd for s in 1:S
                    k = k0 + s
                    sum += C[k]*x[i_pre,J[k],i_post]
                end
                y[i_pre,i,i_post] = α*sum + β*y[i_pre,i,i_post]
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
                         I_pre::CartesianIndices{N_pre},
                         len::Int,
                         I_post::CartesianIndices{N_post}
                         ) where {S,N,N_post,N_pre}
    @assert N == N_post + N_pre + 1
    @inbounds for i_post in I_post
        for i_pre in I_pre
            k0 = 0
            for i in 1:len
                c = α*x[i_pre,i,i_post]
                @simd for s in 1:S
                    k = k0 + s
                    y[i_pre,J[k],i_post] += C[k]*c
                end
                k0 += S
            end
        end
    end
end

function _check(A::SparseSeparableInterpolator{D,T,S},
                N::Int) where {D,T<:AbstractFloat,S}
    1 ≤ D ≤ N ||
        throw(DimensionMismatch("out of range dimension of interpolation"))
    nrows, ncols = size(A)
    nvals = S*nrows
    C = nonzeros(A)
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

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

import Base: axes
import SparseArrays: sparse

using ...Interpolations
import ...Interpolations: Meta, coefficients, columns, rows,
    fit, regularize, regularize!

struct SparseInterpolator{T<:AbstractFloat,S,N} <: LinearMapping
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

Base.eltype(::SparseInterpolator{T,S,N}) where {T,S,N} = T
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
    A = SparseInterpolator([T=eltype(ker),] ker, pos, grd)

yields a sparse linear interpolator suitable for interpolating with kernel
`ker` a function sampled on the grid `grd` at positions `pos`.  Optional
argument `T` is the floating-point type of the coefficients of the operator
`A`.  Call `eltype(A)` to query the type of the coefficients of the sparse
interpolator `A`.

Then `y = apply(A, x)` or `y = A(x)` or `y = A*x` yields the interpolated
values for interpolation weights `x`.  The shape of `y` is the same as that of
`pos`.  Formally, this amounts to computing:

    y[i] = sum_j ker((pos[i] - grd[j])/step(grd))*x[j]

with `step(grd)` the (constant) step size between the nodes of the grid `grd`
and `grd[j]` the `j`-th position of the grid.

"""
function SparseInterpolator(::Type{T}, ker::Kernel,
                            args...) where {T<:AbstractFloat}
    return SparseInterpolator(T(ker), args...)
end

function SparseInterpolator(ker::Kernel{T,S},
                            pos::AbstractArray{<:Real,N},
                            grd::AbstractRange) where {T<:AbstractFloat,S,N}

    # Parameters to convert the interpolated position into a frational grid
    # index. FIXME: Use the central position of the grid to minimize the error.
    delta = T(step(grd))
    alpha = one(T)/delta
    beta = T(first(grd)) - delta
    SparseInterpolator(ker, i -> (T(pos[i]) - beta)*alpha,
                       CartesianIndices(axes(pos)), length(grd))
end

function SparseInterpolator(ker::Kernel{T,S},
                            pos::AbstractArray{<:Real,N},
                            len::Integer) where {T<:AbstractFloat,S,N}
    SparseInterpolator(ker, i -> T(pos[i]), CartesianIndices(axes(pos)), len)
end

function SparseInterpolator(ker::Kernel{T,S},
                            pos::Function,
                            R::CartesianIndices{N},
                            ncols::Integer) where {T<:AbstractFloat,S,N}
    C, J = _sparsecoefs(R, Int(ncols), ker, pos)
    return SparseInterpolator{T,S,N}(C, J, size(R), ncols)
end

@generated function _sparsecoefs(R::CartesianIndices{N},
                                 ncols::Int,
                                 ker::Kernel{T,S},
                                 pos::Function) where {T,S,N}

    _J = Meta.make_varlist(:_j, S)
    _C = Meta.make_varlist(:_c, S)
    code = (Meta.generate_getcoefs(_J, _C, :ker, :lim, :x),
            [:( J[k+$s] = $(_J[s]) ) for s in 1:S]...,
            [:( C[k+$s] = $(_C[s]) ) for s in 1:S]...)

    quote
        lim = limits(ker, ncols)
        nvals = S*length(R)
        J = Array{Int}(undef, nvals)
        C = Array{T}(undef, nvals)
        k = 0
        @inbounds for i in R
            x = convert(T, pos(i))
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

function apply!(α::Real,
                ::Type{Direct},
                A::SparseInterpolator{Ta,S,N},
                x::AbstractVector{Tx},
                scratch::Bool,
                β::Real,
                y::AbstractArray{Ty,N}) where {Ta,Tx<:Real,
                                               Ty<:AbstractFloat,S,N}
    _check(A, y, x)
    if α == 0
        vscale!(y, β)
    else
        T = float(promote_type(Ta, Tx))
        alpha = convert(T, α)
        nrows, ncols = A.nrows, A.ncols
        C, J = coefficients(A), columns(A)
        k0 = 0
        if β == 0
            @inbounds for i in 1:nrows
                sum = zero(T)
                @simd for s in 1:S
                    k = k0 + s
                    j = J[k]
                    sum += C[k]*x[j]
                end
                y[i] = alpha*sum
                k0 += S
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

"""

`AtWA(A,w)` yields the matrix `A'*W*A` from a sparse linear operator `A` and
weights `W = diag(w)`.

"""
function AtWA(A::SparseInterpolator{T,S,N},
              w::AbstractArray{T,N}) where {T,S,N}
    ncols = A.ncols
    AtWA!(Array{T}(undef, ncols, ncols), A, w)
end

"""

`AtA(A)` yields the matrix `A'*A` from a sparse linear operator `A`.

"""
function AtA(A::SparseInterpolator{T,S,N}) where {T,S,N}
    ncols = A.ncols
    AtA!(Array{T}(undef, ncols, ncols), A)
end

# Build the `A'*A` matrix from a sparse linear operator `A`.
function AtA!(dst::AbstractArray{T,2},
              A::SparseInterpolator{T,S,N}) where {T,S,N}
    nrows, ncols = A.nrows, A.ncols
    @assert size(dst) == (ncols, ncols)
    fill!(dst, zero(T))
    C, J = coefficients(A), columns(A)
    k0 = 0
    @assert length(J) == length(C)
    @inbounds for i in 1:nrows
        for s in 1:S
            k = k0 + s
            1 ≤ J[k] ≤ ncols || error("corrupted interpolator table")
        end
        for s1 in 1:S
            k1 = k0 + s1
            j1, c1 = J[k1], C[k1]
            @simd for s2 in 1:S
                k2 = k0 + s2
                j2, c2 = J[k2], C[k2]
                dst[j1,j2] += c1*c2
            end
        end
        k0 += S
    end
    return dst
end

# Build the `A'*W*A` matrix from a sparse linear operator `A` and weights `W`.
function AtWA!(dst::AbstractArray{T,2}, A::SparseInterpolator{T,S,N},
               wgt::AbstractArray{T,N}) where {T,S,N}
    nrows, ncols = A.nrows, A.ncols
    @assert size(dst) == (ncols, ncols)
    @assert size(wgt) == output_size(A)
    fill!(dst, zero(T))
    C, J = coefficients(A), columns(A)
    k0 = 0
    @assert length(J) == length(C)
    @inbounds for i in 1:nrows
        for s in 1:S
            k = k0 + s
            1 ≤ J[k] ≤ ncols || error("corrupted interpolator table")
        end
        w = wgt[i]
        for s1 in 1:S
            k1 = k0 + s1
            j1 = J[k1]
            wc1 = w*C[k1]
            @simd for s2 in 1:S
                k2 = k0 + s2
                j2 = J[k2]
                dst[j1,j2] += C[k2]*wc1
            end
        end
        k0 += S
    end
    return dst
end

# Default regularization levels.
const RGL_EPS = 1e-9
const RGL_MU = 0.0

"""
    fit(A, y [, w]; epsilon=1e-9, mu=0.0) -> x

performs a linear fit of `y` by the model `A*x` with `A` a linear interpolator.
The returned value `x` minimizes:

    sum(w.*(A*x - y).^2)

where `w` are given weights.  If `w` is not specified, all weights are assumed
to be equal to one; otherwise `w` must be an array of nonnegative values and of
same size as `y`.

Keywords `epsilon` and `mu` may be specified to regularize the solution and
minimize:

    sum(w.*(A*x - y).^2) + rho*(epsilon*norm(x)^2 + mu*norm(D*x)^2)

where `D` is a finite difference operator, `rho` is the maximum diagonal
element of `A'*diag(w)*A` and `norm` is the Euclidean norm.

"""
function fit(A::SparseInterpolator{T,S,N},
             y::AbstractArray{T,N},
             w::AbstractArray{T,N};
             epsilon::Real = RGL_EPS,
             mu::Real = RGL_MU) where {T,S,N}
    @assert size(y) == output_size(A)
    @assert size(w) == size(y)

    # Compute RHS vector A'*W*y with W = diag(w).
    rhs = A'*(w.*y)

    # Compute LHS matrix A'*W*A with W = diag(w).
    lhs = AtWA(A, w)

    # Regularize a bit.
    regularize!(lhs, epsilon, mu)

    # Solve the linear equations.
    cholfact!(lhs,:U,Val{true})\rhs
end

function fit(A::SparseInterpolator{T,S,N},
             y::AbstractArray{T,N};
             epsilon::Real = RGL_EPS,
             mu::Real = RGL_MU) where {T,S,N}
    @assert size(y) == output_size(A)
    @assert size(w) == size(y)

    # Compute RHS vector A'*y.
    rhs = A'*y

    # Compute LHS matrix A'*W*A with W = diag(w).
    lhs = AtA(A)

    # Regularize a bit.
    regularize!(lhs, epsilon, mu)

    # Solve the linear equations.
    cholfact!(lhs,:U,Val{true})\rhs
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

#------------------------------------------------------------------------------

# Parameter `T` is the floating-point type of the coefficients, parameter `S`
# is the size of the kernel (number of nodes to combine for a single
# interpolator) and parameter `D` is the dimension of interpolation.
struct SparseUnidimensionalInterpolator{T<:AbstractFloat,S,D} <: LinearMapping
    nrows::Int     # number of rows
    ncols::Int     # number of columns
    C::Vector{T}   # coefficients along the dimension of interpolation
    J::Vector{Int} # columns indices along the dimension of interpolation
end

(A::SparseUnidimensionalInterpolator)(x) = apply(A, x)

interp_dim(::SparseUnidimensionalInterpolator{T,S,D}) where {T,S,D} = D
Base.eltype(::SparseUnidimensionalInterpolator{T,S,D}) where {T,S,D} = T

coefficients(A::SparseUnidimensionalInterpolator) = A.C
columns(A::SparseUnidimensionalInterpolator) = A.J
Base.size(A::SparseUnidimensionalInterpolator) = (A.nrows, A.ncols)
Base.size(A::SparseUnidimensionalInterpolator, i::Integer) =
    (i == 1 ? A.nrows :
     i == 2 ? A.ncols : error("out of bounds dimension"))

"""
    SparseUnidimensionalInterpolator([T=eltype(ker),] ker, d, pos, grd)

yields a linear mapping which interpolates the `d`-th dimension of an array
with kernel `ker` at positions `pos` along the dimension of interpolation `d`
and assuming the input array has grid coordinates `grd` along the the `d`-th
dimension of interpolation.  Argument `pos` is a vector of positions, argument
`grd` may be a range or the length of the dimension of interpolation.  Optional
argument `T` is the floating-point type of the coefficients of the operator.

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
function SparseUnidimensionalInterpolator(::Type{T}, ker::Kernel,
                                          args...) where {T<:AbstractFloat}
    return SparseUnidimensionalInterpolator(T(ker), args...)
end

function SparseUnidimensionalInterpolator(ker::Kernel, d::Integer,
                                          pos::AbstractVector{<:Real},
                                          len::Integer)
    len ≥ 1 || throw(ArgumentError("invalid dimension length"))
    return SparseUnidimensionalInterpolator(ker, d, pos, 1:Int(len))
end

function SparseUnidimensionalInterpolator(ker::Kernel{T,S}, d::Integer,
                                          pos::AbstractVector{<:Real},
                                          grd::AbstractRange
                                          ) where {T<:AbstractFloat,S}
    d ≥ 1 || throw(ArgumentError("invalid dimension of interpolation"))
    nrows = length(pos)
    ncols = length(grd)
    c = (convert(T, first(grd)) + convert(T, last(grd)))/2
    q = 1/convert(T, step(grd))
    r = convert(T, 1 + length(grd))/2
    C, J = _sparsecoefs(CartesianIndices((nrows,)), ncols, ker,
                        i -> q*(convert(T, pos[i]) - c) + r)
    D = Int(d)
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
        Ifast = CartesianIndices(xdims[1:D-1])
        Islow = CartesianIndices(xdims[D+1:N])
        T = promote_type(Ta,Tx)
        alpha = convert(T, α)
        if β == 0
            _apply_direct!(T, Val{S}, C, J, alpha, x, y,
                           Ifast, nrows, Islow)
        else
            beta = convert(Ty, β)
            _apply_direct!(T, Val{S}, C, J, alpha, x, beta, y,
                           Ifast, nrows, Islow)
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
                        Ifast::CartesianIndices{Nfast},
                        len::Int,
                        Islow::CartesianIndices{Nslow}
                        ) where {T<:AbstractFloat,S,N,Nslow,Nfast}
    @assert N == Nslow + Nfast + 1
    @inbounds for islow in Islow
        for ifast in Ifast
            k0 = 0
            for i in 1:len
                sum = zero(T)
                @simd for s in 1:S
                    k = k0 + s
                    sum += C[k]*x[ifast,J[k],islow]
                end
                y[ifast,i,islow] = α*sum
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
                        Ifast::CartesianIndices{Nfast},
                        len::Int,
                        Islow::CartesianIndices{Nslow}
                        ) where {T<:AbstractFloat,S,N,Nslow,Nfast}
    @assert N == Nslow + Nfast + 1
    @inbounds for islow in Islow
        for ifast in Ifast
            k0 = 0
            for i in 1:len
                sum = zero(T)
                @simd for s in 1:S
                    k = k0 + s
                    sum += C[k]*x[ifast,J[k],islow]
                end
                y[ifast,i,islow] = α*sum + β*y[ifast,i,islow]
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
                         Ifast::CartesianIndices{Nfast},
                         len::Int,
                         Islow::CartesianIndices{Nslow}
                         ) where {S,N,Nslow,Nfast}
    @assert N == Nslow + Nfast + 1
    @inbounds for islow in Islow
        for ifast in Ifast
            k0 = 0
            for i in 1:len
                c = α*x[ifast,i,islow]
                @simd for s in 1:S
                    k = k0 + s
                    y[ifast,J[k],islow] += C[k]*c
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

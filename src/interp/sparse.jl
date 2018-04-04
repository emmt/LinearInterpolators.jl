#
# interp/sparse.jl --
#
# Implement sparse linear interpolator.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# Copyright (C) 2017-2018, Éric Thiébaut.
# This file is part of TiPi.  All rights reserved.
#

struct SparseInterpolator{T<:AbstractFloat,S,N} <: LinearMapping
    C::Vector{T}
    J::Vector{Int}
    nrows::Int
    ncols::Int
    dims::NTuple{N,Int} # dimensions of result
    function (::Type{SparseInterpolator{T,S,N}})(C::Vector{T},
                                                 J::Vector{Int},
                                                 dims::NTuple{N,Int},
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

(A::SparseInterpolator{T,S,N})(x::AbstractVector{T}) where {T,S,N} =
    apply(A, x)

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
    I = Array{Int}(nvals)
    K = 1:S
    for i in 1:nrows
        for k in K
            @inbounds I[k] = i
        end
        K += S
    end
    return I
end

# Convert to a sparse matrix.
Base.sparse(A::SparseInterpolator) =
    sparse(rows(A), columns(A), coefficients(A), A.nrows, A.ncols)


"""
# Sparse linear interpolator

A sparse linear interpolator is created by:

    op = SparseInterpolator(ker, pos, grd)

which yields a linear interpolator suitable for interpolating with the kernel
`ker` a function sampled on the grid `grd` at positions `pos`.

Then `y = apply(op, x)` or `y = op(x)` yields the interpolated values for
interpolation weights `x`.  The shape of `y` is the same as that of `pos`.

Formally, this amounts to computing:

    y[i] = sum_j ker((pos[i] - grd[j])/step(grd))*x[j]

with `step(grd)` the (constant) step size between the nodes of the grid `grd`
and `grd[j]` the `j`-th position of the grid.

"""
function SparseInterpolator(ker::Kernels.Kernel{T,S,B}, pos::AbstractArray,
                            grd::Range) where {T<:AbstractFloat,S,B}

    # Parameters to convert the interpolated position into a frational grid
    # index.
    delta = T(step(grd))
    alpha = one(T)/delta
    beta = T(first(grd)) - delta
    SparseInterpolator(ker, (i) -> (T(pos[i]) - beta)*alpha,
                       CartesianRange(indices(pos)), length(grd))
end

function SparseInterpolator(ker::Kernel{T,S,B}, pos::AbstractArray,
                            len::Integer) where {T<:AbstractFloat,S,B}
    SparseInterpolator(ker, (i) -> T(pos[i]), CartesianRange(indices(pos)),
                       Int(len))
end

function SparseInterpolator(ker::Kernels.Kernel{T,S,B}, pos::Function,
                            R::CartesianRange{CartesianIndex{N}}, ncols::Int
                            ) where {T<:AbstractFloat,S,B,N}
    C, J = computecoefs(R, ncols, ker, pos)
    return SparseInterpolator{T,S,N}(C, J, size(R), ncols)
end

function computecoefs(R::CartesianRange{CartesianIndex{N}}, ncols::Int,
                      ker::Kernel{T,1,B}, pos::Function) where {T,B,N}
    lim = limits(ker, ncols)
    nvals = length(R)
    C = Array{T}(nvals)
    J = Array{Int}(nvals)
    k = 0
    @inbounds for i in R
        x = pos(i) :: T
        j1, w1 = getcoefs(ker, lim, x)
        k += 1
        J[k] = j1
        C[k] = w1
    end
    return C, J
end

function computecoefs(R::CartesianRange{CartesianIndex{N}}, ncols::Int,
                      ker::Kernel{T,2,B}, pos::Function) where {T,B,N}
    lim = limits(ker, ncols)
    nvals = 2*length(R)
    C = Array{T}(nvals)
    J = Array{Int}(nvals)
    k = 0
    @inbounds for i in R
        x = pos(i) :: T
        j1, j2, w1, w2 = getcoefs(ker, lim, x)
        J[k+1] = j1
        J[k+2] = j2
        C[k+1] = w1
        C[k+2] = w2
        k += 2
    end
    return C, J
end

function computecoefs(R::CartesianRange{CartesianIndex{N}}, ncols::Int,
                      ker::Kernel{T,3,B}, pos::Function) where {T,B,N}
    lim = limits(ker, ncols)
    nvals = 3*length(R)
    C = Array{T}(nvals)
    J = Array{Int}(nvals)
    k = 0
    @inbounds for i in R
        x = pos(i) :: T
        j1, j2, j3, w1, w2, w3 = getcoefs(ker, lim, x)
        J[k+1] = j1
        J[k+2] = j2
        J[k+3] = j3
        C[k+1] = w1
        C[k+2] = w2
        C[k+3] = w3
        k += 3
    end
    return C, J
end

function computecoefs(R::CartesianRange{CartesianIndex{N}}, ncols::Int,
                      ker::Kernel{T,4,B}, pos::Function) where {T,B,N}
    lim = limits(ker, ncols)
    nvals = 4*length(R)
    C = Array{T}(nvals)
    J = Array{Int}(nvals)
    k = 0
    @inbounds for i in R
        x = pos(i) :: T
        j1, j2, j3, j4, w1, w2, w3, w4 = getcoefs(ker, lim, x)
        J[k+1] = j1
        J[k+2] = j2
        J[k+3] = j3
        J[k+4] = j4
        C[k+1] = w1
        C[k+2] = w2
        C[k+3] = w3
        C[k+4] = w4
        k += 4
    end
    return C, J
end

function _check(A::SparseInterpolator{T,S,N},
                out::AbstractArray{T,N},
                inp::AbstractVector{T}) where {T,S,N}
    nvals = S*A.nrows # number of non-zero coefficients
    J, ncols = A.J, A.ncols
    if length(A.C) != nvals
        error("corrupted sparse interpolator (bad number of coefficients)")
    end
    if length(J) != nvals
        error("corrupted sparse interpolator (bad number of indices)")
    end
    if length(inp) != ncols
        error("bad vector length (expecting $(A.ncol), got $(length(inp)))")
    end
    if size(out) != A.dims
        error("bad array size (expecting $(A.dims), got $(size(out)))")
    end
    if length(out) == A.nrows
        error("corrupted sparse interpolator (bad number of \"rows\")")
    end
    @inbounds for k in 1:nvals
        if !(1 ≤ J[k] ≤ ncols)
            error("corrupted sparse interpolator (out of bound indices)")
        end
    end
end

function vcreate(::Type{Direct}, A::SparseInterpolator{T,S,N},
                 x::AbstractVector{T}) where {T,S,N}
    return Array{T}(output_size(A))
end

function vcreate(::Type{Adjoint}, A::SparseInterpolator{T,S,N},
                 x::AbstractArray{T,N}) where {T,S,N}
    return Array{T}(input_size(A))
end

function apply!(α::Scalar, ::Type{Direct}, A::SparseInterpolator{T,S,N},
                x::AbstractVector{T},
                β::Scalar, y::AbstractArray{T,N}) where {T,S,N}
    _check(A, y, x)
    if α == zero(α)
        vscale!(y, β)
    else
        const alpha = convert(T, α)
        const beta = convert(T, β)
        nrows, ncols = A.nrows, A.ncols
        C, J, K = A.C, A.J, 1:S
        @inbounds for i in 1:nrows
            sum = zero(T)
            @simd for k in K
                j = J[k]
                sum += C[k]*x[j]
            end
            if beta == zero(beta)
                y[i] = alpha*sum
            else
                y[i] = alpha*sum + beta*y[i]
            end
            K += S
        end
    end
    return y
end

function apply!(α::Scalar, ::Type{Adjoint}, A::SparseInterpolator{T,S,N},
                x::AbstractArray{T,N},
                β::Scalar, y::AbstractVector{T}) where {T,S,N}
    _check(A, x, y)
    vscale!(y, β)
    if α != zero(α)
        alpha = convert(T, α)
        nrows, ncols = A.nrows, A.ncols
        C, J, K = A.C, A.J, 1:S
        @inbounds for i in 1:nrows
            c = alpha*x[i]
            @simd for k in K
                j = J[k]
                y[j] += C[k]*c
            end
            K += S
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
    AtWA!(Array{T}(ncols, ncols), A, w)
end

"""

`AtA(A)` yields the matrix `A'*A` from a sparse linear operator `A`.

"""
function AtA(A::SparseInterpolator{T,S,N}) where {T,S,N}
    ncols = A.ncols
    AtA!(Array{T}(ncols, ncols), A)
end

# Build the `A'*A` matrix from a sparse linear operator `A`.
function AtA!(dst::AbstractArray{T,2},
              A::SparseInterpolator{T,S,N}) where {T,S,N}
    nrows, ncols = A.nrows, A.ncols
    @assert size(dst) == (ncols, ncols)
    fill!(dst, zero(T))
    C, J, K = A.C, A.J, 1:S
    @assert length(J) == length(C)
    @inbounds for i in 1:nrows
        for k in K
            1 ≤ J[k] ≤ ncols || error("corrupted interpolator table")
        end
        for k1 in K
            j1, c1 = J[k1], C[k1]
            @simd for k2 in K
                j2, c2 = J[k2], C[k2]
                dst[j1,j2] += c1*c2
            end
        end
        K += S
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
    C, J, K = A.C, A.J, 1:S
    @assert length(J) == length(C)
    @inbounds for i in 1:nrows
        for k in K
            1 ≤ J[k] ≤ ncols || error("corrupted interpolator table")
        end
        w = wgt[i]
        for k1 in K
            j1 = J[k1],
            wc1 = w*C[k1]
            @simd for k2 in K
                j2 = J[k2]
                dst[j1,j2] += C[k2]*wc1
            end
        end
        K += S
    end
    return dst
end

# Default regularization levels.
const RGL_EPS = 1e-9
const RGL_MU = 0.0

"""
```julia
fit(A, y[, w][; epsilon=1e-9, mu=0.0]) -> x
```

performs a linear fit of `y` by the model `A*x` with `A` a linear interpolator.
The returned value `x` minimizes:

    sum(w.*(A*x - y).^2)

where `w` are some weights.  If `w` is not specified, all weights are assumed
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
maximum diagonal element of `A`.  The in-place version:

    regularize!(A, ϵ, μ) -> A

stores the regularized matrix in `A` (and returns it).

"""
regularize(A::AbstractArray{T,2}, args...) where {T<:AbstractFloat} =
    regularize!(copy!(Array{T}(size(A)), A), args...)

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
    const n = size(A,1)
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

@doc @doc(regularize) regularize!

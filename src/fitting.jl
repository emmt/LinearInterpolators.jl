module Fitting

export fit, solve, solve!

using ..LinearInterpolators: argument_error, dimension_mismatch

# FIXME: This is a hack to just check that code compiles.
abstract type SparseInterpolator{T,L,M,N} end

"""
    AtWA(A,w)

yields the matrix `A'*W*A` from linear operator `A` and weights `W = diag(w)`.

"""
function AtWA(A::SparseInterpolator{T1,L,M,1},
              w::AbstractArray{T2,M}) where {T1,T2,L,M}
    T = promote_type(T1,T2)
    ncols = prod(cols(A))
    return AtWA!(Array{T}(undef, ncols, ncols), A, w)
end

"""
    AtA(A)

yields the matrix `A'*A` from a sparse linear operator `A`.

"""
function AtWA(A::SparseInterpolator{T,L,M,1}) where {T,L,M}
    ncols = prod(cols(A))
    return AtA!(Array{T}(undef, ncols, ncols), A)
end

# Build the `A'*A` matrix from a sparse linear operator `A`.
function AtA!(dst::AbstractMatrix,
              A::SparseInterpolator{T,L,M,1}) where {T,L,M}
    inds = Base.OneTo(prod(cols(A)))
    axes(dst) == (inds, inds) || dimension_mismatch(
        "invalid destination indices")
    A.wgt::Tuple{Array{<:Tuple{Vararg{T}},M}}
    A.ind::Tuple{Array{<:Tuple{Vararg{Int},M}}}
    C = A.wgt[1]
    J = A.ind[1]
    axes(J) == axes(C) || dimension_mismatch(
        "corrupted interpolator")
    return unsafe_AtA!(dst, C, J)
end

# Build the `A'*W*A` matrix from a sparse linear operator `A` and weights `W`.
function AtWA!(dst::AbstractMatrix,
               A::SparseInterpolator{T,L,M,1},
               W::AbstractArray{T′,M}) where {T,T′,L,M}
    inds = Base.OneTo(prod(cols(A)))
    axes(dst) == (inds, inds) || dimension_mismatch(
        "invalid destination indices")
    A.wgt::Tuple{Array{<:Tuple{Vararg{T}},M}}
    A.ind::Tuple{Array{<:Tuple{Vararg{Int},M}}}
    C = A.wgt[1]
    J = A.ind[1]
    axes(J) == axes(C) || dimension_mismatch(
        "corrupted interpolator")
    axes(W) == axes(C) || dimension_mismatch(
        "invalid weight indices")
    return unsafe_AtA!(dst, C, I, W)
end

function unsafe_AtA!(dst::AbstractMatrix,
                     C::Array{NTuple{S,T},M},
                     J::Array{NTuple{S,Int},M}) where {T,S,M}
    fill!(dst, zero(eltype(dst)))
    n = minimum(size(dst))
    @inbounds for i in eachindex(C,J)
        c = C[i]
        j = J[i]
        flag = true
        for s in 1:S # FIXME: unroll
            flag &= (1 ≤ j[s])&(j[s] ≤ n)
        end
        flag || error("corrupted interpolator table")
        for s1 in 1:S # FIXME: unroll
            j1, c1 = j[s1], c[s1]
            @simd for s2 in 1:S # FIXME: unroll
                j2, c2 = j[s2], c[s2]
                dst[j1,j2] += c1*c2
            end
        end
    end
    return dst
end
# Expr(:call, :(&), :((1 ≤ $(j[$s]))&($(j[$s]) ≤ n)), recurse())

function unsafe_AtWA!(dst::AbstractMatrix,
                      C::Array{NTuple{S,T},M},
                      J::Array{NTuple{S,Int},M},
                      W::AbstractArray{T′,M}) where {T,T′,S,M}
    fill!(dst, zero(eltype(dst)))
    n = minimum(size(dst))
    @inbounds for i in eachindex(C,J,W)
        c = C[i]
        j = J[i]
        w = W[i]
        if w != zero(w)
            flag = true
            for s in 1:S # FIXME: unroll
                flag &= (1 ≤ j[s])&(j[s] ≤ n)
            end
            flag || error("corrupted interpolator table")
            for s1 in 1:S # FIXME: unroll
                j1, w_c1 = j[s1], w*c[s1]
                @simd for s2 in 1:S # FIXME: unroll
                    j2, c2 = j[s2], c[s2]
                    dst[j1,j2] += w_c1*c2
                end
            end
        end
    end
    return dst
end

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
function fit(A::SparseInterpolator{Ta,L,M,1},
             y::AbstractArray{Ty,M},
             w::AbstractArray{Tw,M};
             kwds...) where {Ta,Ty,Tw,L,M}
    axes(y) == output_axes(A) || dimension_mismatch(
        "invalid indices for data array")
    axes(w) == output_axes(A) || dimension_mismatch(
        "invalid indices for array of weights")

    # Compute RHS vector A'*W*y with W = diag(w).
    rhs = A'*(w.*y)

    # Compute LHS matrix A'*W*A with W = diag(w).
    lhs = AtWA(A, w)

    # Solve the linear equations.
    return solve!(Val(:linfit), lhs, rhs; kwds...)
end

function fit(A::SparseInterpolator{Ta,L,M,1},
             y::AbstractArray{Ty,M};
             kwds...) where {Ta,Ty,Tw,L,M}
    axes(y) == output_axes(A) || dimension_mismatch(
        "invalid indices for data array")

    # Compute RHS vector A'*y.
    rhs = A'*y

    # Compute LHS matrix A'*W*A with W = diag(w).
    lhs = AtA(A)

    # Solve the linear equations.
    return solve!(Val(:linfit), lhs, rhs; kwds...)
end

# Default regularization levels.
const RGL_EPS = 1e-9
const RGL_MU = 0.0

"""
    solve(prob, args...; kwds...)

yields the solution of problem `prob` for arguments `args...` and keywords
`kwds...`.  Inputs are not overwritten.

"""
function solve(prob::Val{:linfit}, A::AbstractMatrix{Ta},
               b::AbstractVector{Tb}; kwds...) where {Ta,Tb}
    T = float(promote_type(Ta, Tb))
    return solve!(prob,
                  copyto!(similar(A, T), A),
                  copyto!(similar(b, T), b); kwds...)
end

"""
    solve!(prob, args...; kwds...)

yields the solution of problem `prob` for arguments `args...` and keywords
`kwds...`.  Inputs may be overwritten.

"""
function solve!(prob::Val{:linfit}, A::AbstractMatrix{Ta},
                b::AbstractVector{Tb}; kwds...) where {Ta,Tb}
    T = float(promote_type(Ta, Tb))
    return solve!(prob,
                  convert(AbstractMatrix{T}, A),
                  convert(AbstractVector{T}, b); kwds...)
end

function solve!(prob::Val{:linfit}, A::AbstractMatrix{T},
                b::AbstractVector{T};
                epsilon::Real = RGL_EPS,
                mu::Real = RGL_MU) where {T}
    # Regularize a bit.
    regularize!(A, epsilon, mu)

    # Solve the linear equations.
    return cholesky!(A)\b # FIXME: check this
end

"""
    regularize(A, ϵ, μ) -> R

regularizes the symmetric matrix `A` to produce the matrix:

    R = A + ρ*(ϵ*I + μ*D'*D)

where `I` is the identity, `D` is a finite difference operator and `ρ` is the
maximum diagonal element of `A`.

"""
regularize(A::AbstractArray{T,2}, args...) where {T<:AbstractFloat} =
    regularize!(copyto!(similar(A, T), A), args...)

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
    eps ≥ zero(eps) || argument_error("`eps` must be nonnegative")
    mu ≥ zero(mu) || argument_error("`mu` must be nonnegative")
    m, n = size(A)
    m == n || argument_error("matrix `A` must be square")
    Base.has_offset_axes(A) && argument_error(
        "matrix `A` has non-standard indices")
    if eps > zero(eps) || mu > zero(mu)
        rho = A[1,1]
        for j in 2:n
            d = A[j,j]
            rho = max(rho, d)
        end
        rho > zero(rho) || error(
            "diagonal entries of matrix `A` are all negative")
    end
    if eps > zero(eps)
        q = eps*rho
        for j in 1:n
            A[j,j] += q
        end
    end
    if mu > zero(mu)
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

end # module Fitting

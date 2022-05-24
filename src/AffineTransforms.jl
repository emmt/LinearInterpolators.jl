"""
    AffineTransforms

implements affine transforms of small size whose coefficients are stored as a
tuple.

"""
module AffineTransforms

export AffineTransform, offset

using Base: @propagate_inbounds
using StaticArrays
import TwoDimensional

"""
    AffineTransform{M,N}(coefs)

yields an affine transform which maps `N`-tuples to `M`-tuples.  The arguments
specify the `M*(N+1)` coefficients of the affine transform in *row-major*
order.  For example:

    R = AffineTransform{2,3}(c10, c11, c12, c13,
                             c20, c21, c22, c23)

yields an affine tranform which can be applied to a 3-tuple to produce a
2-tuple:

    R((x1,x2,x3)) -> (c10 + c11*x1 + c12*x2 c13*x3,
                      c20 + c21*x1 + c22*x2 c23*x3)

The affine transform coefficients may be specifed by a matrix `A` representing
the linear part of the transform and a vector `b` representing the offset of
the transform:

    R = AffineTransform(A, b)  # A and b can be in any order

Unless `A` and `b` are static arrays (from the `StaticArrays` package) type
parameters `M` and `N` should be specified to avoid type instability.  Methods
`Matrix(R)` and `SMatrix(R)` yield the matrix `A`.  Similarly, methods
`Vector(R)` `SVector(R)` yield the vector `b`.

Calling `Tuple(R)` yields a tuple of the coefficients of the affine transform
`R` in row-major order:

    Tuple(R) -> (c10, c11, c12, c13, c20, c21, c22, c23)

The syntax `R[k]` yields the `k`-th coefficient (again in row-major order),
whild `R[i,j]` yields the coefficient at row index `i ∈ 1:M` and column index
`j ∈ 0:N` (column number `j=0` corresponds to the offset vector `b`):

    ∀ i ∈ 1:M, ∀ j ∈ 1:N, R[i,j] -> Matrix(R)[i,j]
    ∀ i ∈ 1:M,            R[i,0] -> Vector(R)[i]

The constructor may also be called to convert other affine transforms or the
type `T` of the coefficients.  This however requires to know type parameters
`M` and `N`, the `with_eltype` method can be used to overcome this:

    with_eltype(T, R)

is equivalent to:

    AffineTransform{M,N,T}(R)

"""
struct AffineTransform{M,N,T,L}
    coefs::NTuple{L,T}
    function AffineTransform{M,N,T}(coefs::NTuple{L}) where {M,N,T,L}
        M::Int
        N::Int
        L == M*(N + 1) || error("bad number of coefficients")
        return new{M,N,T,L}(coefs)
    end
end

# Build an affine transform from a list of coefficients.
AffineTransform{M,N}(coefs::Vararg) where {M,N} = AffineTransform{M,N}(coefs)
AffineTransform{M,N,T}(coefs::Vararg) where {M,N,T} =
    AffineTransform{M,N,T}(coefs)
function AffineTransform{M,N}(coefs::Tuple) where {M,N}
    T = promote_type(map(typeof, coefs)...)
    return AffineTransform{M,N,T}(coefs)
end

AffineTransform{M,N,T}(R::AffineTransform{M,N,T}) where {M,N,T} = R
AffineTransform{M,N,T}(R::AffineTransform{M,N,T′}) where {M,N,T,T′} =
    AffineTransform{M,N,T}(Tuple(R))

AffineTransform{2,2,T}(R::TwoDimensional.AffineTransform2D) where {M,N,T} =
    AffineTransform{2,2,T}(R.x, R.xx, R.xy,
                           R.y, R.yx, R.yy)
AffineTransform{2,2}(R::TwoDimensional.AffineTransform2D) = AffineTransform(R)
AffineTransform(R::TwoDimensional.AffineTransform2D{T}) where {T} =
    AffineTransform{2,2,T}(R)

TwoDimensional.AffineTransform2D(R::AffineTransform{2,2,T}) where {T} =
    TwoDimensional.AffineTransform2D{T}(R)
TwoDimensional.AffineTransform2D{T}(R::AffineTransform{2,2}) where {T} =
    TwoDimensional.AffineTransform2D{T}(R[2], R[3], R[1],
                                        R[5], R[6], R[4])

Base.convert(::Type{T}, x::T) where {T<:AffineTransform} = x
function Base.convert(T::Type{<:AffineTransform},
                      x::Union{AffineTransform,
                               TwoDimensional.AffineTransform2D})
    return T(x)
end

function Base.convert(T::Type{<:TwoDimensional.AffineTransform2D},
                      x::AffineTransform{2,2})
    return T(x)
end

Base.similar(R::AffineTransform{M,N}, ::Type{T}) where {M,N,T} =
    AffineTransform{M,N,T}(R)

Base.eltype(R::AffineTransform) = eltype(typeof(R))
Base.eltype(::Type{<:AffineTransform{M,N,T}}) where {M,N,T} = T

Base.ndims(R::AffineTransform) = ndims(typeof(R))
Base.ndims(::Type{<:AffineTransform}) = 2

Base.length(R::AffineTransform) = length(typeof(R))
Base.length(::Type{<:AffineTransform{M,N,T,L}}) where {M,N,T,L} = L

Base.axes(R::AffineTransform) = axes(typeof(R))
Base.axes(::Type{<:AffineTransform{M,N}}) where {M,N} = (Base.OneTo(M), 0:N)

Base.size(R::AffineTransform) = size(typeof(R))
Base.size(::Type{<:AffineTransform{M,N}}) where {M,N} = (M, N+1)

Base.getindex(R::AffineTransform, k::Integer) = getindex(R, Int(k))
function Base.getindex(R::AffineTransform{M,N,T,L}, k::Int) where {M,N,T,L}
    @boundscheck ((1 ≤ k)&(k ≤ L)) || error("out of bound index")
    return @inbounds storage(R)[k]
end

Base.getindex(R::AffineTransform, i::Integer, j::Integer) =
    getindex(R, Int(i), Int(j))
function Base.getindex(R::AffineTransform{M,N}, i::Int, j::Int) where {M,N}
    @boundscheck ((1 ≤ i)&(i ≤ M)&(0 ≤ j)&(j ≤ N)) || error(
        "out of bound indices")
    k = storage_index(R, i, j)
    return @inbounds storage(R)[k]
end

(R::AffineTransform{M,N})(x::Vararg{Any,N}) where {M,N} = R(x)
(R::AffineTransform{M,N})(x::SVector{N}) where {M,N} = SVector{M}(R(x.data))
(R::AffineTransform{M,N})(x::CartesianIndex{N}) where {M,N} = R(x.I)

@generated function (R::AffineTransform{M,N,T,L})(x::NTuple{N}) where {M,N,T,L}
    L == M*(N + 1) || error("bad number of coefficients")
    return generate_apply(M, N)
end

# Generate code to apply an affine transform.
function generate_apply(M::Int, N::Int)
    code = Expr(:tuple)
    k = 0
    for i = 1:M
        k += 1
        local ex = Expr(:call, :(+), :(R.coefs[$k]))
        for j = 1:N
            k += 1
            push!(ex.args, :(R.coefs[$k]*x[$j]))
        end
        push!(code.args, ex)
    end
    return code
end

"""
    storage(R)

yields the tuple of coefficients of the affine transform `R` in row-major order.

"""
storage(R::AffineTransform) = getfield(R, :coefs)
Base.Tuple(R::AffineTransform) = storage(R)

"""
    storage_index(R, i, j) -> k

yields the linear index `k ∈ 1:M*(N+1)` of the coefficient of the affine
transform `R::AffineTransform{M,N}` at row `i ∈ 1:M` and column `j ∈ 0:N`.

Argument `R` may be the type of the affine transform.

"""
storage_index(R::AffineTransform, i::Integer, j::Integer) =
    storage_index(typeof(R), i, j)

storage_index(T::Type{<:AffineTransform}, i::Integer, j::Integer) =
    storage_index(T, Int(i), Int(j))

# indices are stored in row-major order for i ∈ 1:M, j ∈ 0:N
storage_index(::Type{<:AffineTransform{M,N}}, i::Int, j::Int) where {M,N} =
    (i - 1)*(N+1) + j + 1

"""
    offset(R) -> b

yields a tuple with the offset implemented by the affine transform `R`.  This
is the same as applying `R` to a tuple of zeros.

"""
offset(R::AffineTransform{M,N}) where {M,N} = ntuple(i -> R[i,0], Val(M))

# Check for equality.
Base.:(==)(A::AffineTransform, B::AffineTransform) = false
Base.:(==)(A::AffineTransform{M,N}, B::AffineTransform{M,N}) where {M,N} =
    storage(A) == storage(B)

# Compose affine transforms.
function Base.:(*)(R1::AffineTransform{N1,N2},
                   R2::AffineTransform{N2,N3}) where {N1,N2,N3}
    A1, b1 = SMatrix(R1), SVector(R1)
    A2, b2 = SMatrix(R2), SVector(R2)
    return AffineTransform(A1*A2, A1*b2 + b1)
end

# Apply and affine transform to a vector.
function Base.:(*)(R::AffineTransform{M,N},
                   x::Union{NTuple{N},
                            SVector{N},
                            CartesianIndex{N}}) where {M,N}
    return R(x)
end

# Invert affine transforms.
function Base.inv(R::AffineTransform{N,N}) where {N}
    Q = inv(SMatrix(R))
    b = SVector(R)
    return AffineTransform(Q, -Q*b)
end

# Left division by an affine transform.
Base.:(\)(R::AffineTransform{N,N}, x::NTuple{N}) where {N} =
    (R\SVector(x)).data
Base.:(\)(R::AffineTransform{N,N}, x::SVector{N}) where {N} =
    SMatrix(R)\(x - SVector(R))
Base.:(\)(R1::AffineTransform{N,N}, R2::AffineTransform{N}) where {N} =
    inv(R1)*R2

# Right division by an affine transform.
Base.:(/)(R1::AffineTransform{N,N}, R2::AffineTransform{N}) where {N} =
    R1*inv(R2)

StaticArrays.SMatrix(R::AffineTransform{M,N,T}) where {M,N,T} =
    SMatrix{M,N,T}(R)
StaticArrays.SMatrix{M,N}(R::AffineTransform{M,N,T}) where {M,N,T} =
    SMatrix{M,N,T}(R)
StaticArrays.SMatrix{M,N,T}(R::AffineTransform{M,N}) where {M,N,T} =
    SMatrix{M,N,T}(ntuple(k -> ((j,i) = divrem(k-1, M);
                                @inbounds(R[i+1,j+1])), Val(M*N)))

function Base.Matrix(R::AffineTransform{M,N,T}) where {M,N,T}
    A = Matrix{T}(undef, M, N)
    @inbounds for j ∈ 1:N, i ∈ 1:M
        A[i,j] = R[i,j]
    end
    return A
end

StaticArrays.SVector(R::AffineTransform{M,N,T}) where {M,N,T} =
    SVector{M,T}(R)
StaticArrays.SVector{M}(R::AffineTransform{M,N,T}) where {M,N,T} =
    SVector{M,T}(R)
StaticArrays.SVector{M,T}(R::AffineTransform{M,N}) where {M,N,T} =
    SVector{M,T}(ntuple(i -> @inbounds(R[i,0]), Val(M)))

function Base.Vector(R::AffineTransform{M,N,T}) where {M,N,T}
    b = Vector{T}(undef, M)
    @inbounds for i ∈ 1:M
        b[i] = R[i,0]
    end
    return b
end

# Put arguments in order.
AffineTransform(b::AbstractVector, A::AbstractMatrix) = AffineTransform(A, b)
AffineTransform{M,N}(b::AbstractVector, A::AbstractMatrix) where {M,N} =
    AffineTransform{M,N}(A, b)
AffineTransform{M,N,T}(b::AbstractVector, A::AbstractMatrix) where {M,N,T} =
    AffineTransform{M,N,T}(A, b)

# Provide dimensions.
AffineTransform(A::SMatrix{M,N}, b::SVector{M}) where {M,N} =
    AffineTransform{M,N}(A, b)
function AffineTransform(A::AbstractMatrix, b::AbstractVector)
    M, N = size(A)
    length(b) == M || throw(DimensionMismatch(
        "matrix and vector of affine transform have incompatible sizes"))
    return AffineTransform{M,N}(A, b)
end

# Provide element type.
function AffineTransform{M,N}(A::AbstractMatrix, b::AbstractVector) where {M,N}
    T = promote_type(eltype(A), eltype(b))
    return AffineTransform{M,N,T}(A, b)
end

AffineTransform{M,N,T}(A::SMatrix{M,N}, b::SVector{M}) where {M,N,T} =
    unsafe_build(AffineTransform{M,N,T}, A, b)

function AffineTransform{M,N,T}(A::AbstractMatrix,
                                b::AbstractVector) where {M,N,T}
    axes(A) == (Base.OneTo(M), Base.OneTo(N)) || throw(DimensionMismatch(
        "matrix of affine transform has incompatible axes"))
    axes(b) == (Base.OneTo(M),) || throw(DimensionMismatch(
        "vector of affine transform has incompatible axes"))
    return unsafe_build(AffineTransform{M,N,T}, A, b)
end

@inline function unsafe_build(::Type{<:AffineTransform{M,N,T}},
                              A::AbstractMatrix,
                              b::AbstractVector)  where {M,N,T}
    # k = (i - 1)*(N+1) + j + 1  for i ∈ 1:M, j ∈ 0:N
    # ==> (i - 1, j) = divrem(k - 1, N+1)
    return AffineTransform{M,N,T}(
        ntuple(k -> ((i′, j) = divrem(k-1, N+1); i = i′ + 1;
                     @inbounds(j > 0 ? A[i,j] : b[i])), Val(M*(N+1))))
end

end # module

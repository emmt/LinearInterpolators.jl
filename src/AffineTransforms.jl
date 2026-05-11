"""

Module `AffineTransforms` implements affine transforms of small size whose coefficients are
stored as a tuple.

"""
module AffineTransforms

export AffineTransform, offset

using Base: @propagate_inbounds
using StaticArrays
import TwoDimensional

"""
    AffineTransform{M,N}(coefs)

Return an affine transform which maps `N`-tuples to `M`-tuples. The arguments specify the
`M*(N+1)` coefficients of the affine transform in *row-major* order (for efficiency when
applying the transform).

For example:

    R = AffineTransform{2,3}(c10, c11, c12, c13,
                             c20, c21, c22, c23)

yields an affine transform which can be applied to a 3-tuple to produce a 2-tuple:

    R((x1,x2,x3)) -> (c10 + c11*x1 + c12*x2 c13*x3,
                      c20 + c21*x1 + c22*x2 c23*x3)

The affine transform coefficients may be specified by a matrix `A` representing the linear
part of the transform and a vector `b` representing the offset of the transform:

    R = AffineTransform(A, b)  # A and b can be in any order

Unless `A` and `b` are static arrays (from the `StaticArrays` package) type parameters `M`
and `N` should be specified to avoid type instability. Methods `Matrix(R)` and `SMatrix(R)`
yield the matrix `A`. Similarly, methods `Vector(R)` and `SVector(R)` yield the vector `b`.

Calling `Tuple(R)` yields a tuple of the coefficients of the affine transform `R` in
row-major order:

    Tuple(R) -> (c10, c11, c12, c13, c20, c21, c22, c23)

The syntax `R[k]` yields the `k`-th coefficient (again in row-major order), while `R[i,j]`
yields the coefficient at row index `i ∈ 1:M` and column index `j ∈ 0:N` (column number
`j=0` corresponds to the offset vector `b`):

    ∀ i ∈ 1:M, ∀ j ∈ 1:N, R[i,j] -> Matrix(R)[i,j]
    ∀ i ∈ 1:M,            R[i,0] -> Vector(R)[i]

The constructor may also be called to convert other affine transforms or the type `T` of the
coefficients. This however requires to know type parameters `M` and `N`, the
`adapt_precision` method can be used to overcome this:

    adapt_precision(T, R)

is equivalent to:

    AffineTransform{M,N,T}(R)

"""
struct AffineTransform{M,N,T,L}
    coefs::NTuple{L,T}
    function AffineTransform{M,N,T}(coefs::NTuple{L}) where {M,N,T,L}
        M::Int
        N::Int
        L == M*(N + 1) || throw_bad_number_of_coefficients(L, M, N)
        return new{M,N,T,L}(coefs)
    end
end

@noinline throw_bad_number_of_coefficients(L::Integer, M::Integer, N::Integer) =
    throw(DimensionMismatch(
        "`AffineTransform{$M,$N}` has $(M*(N + 1)) coefficients, got $L coefficient(s)"))

# Build an affine transform from a list of coefficients, all with the same type `T`.
AffineTransform{M,N}(coefs::T...) where {M,N,T} = AffineTransform{M,N,T}(coefs)
AffineTransform{M,N}(coefs::NTuple{L,T}) where {L,M,N,T} = AffineTransform{M,N,T}(coefs)

# Build an affine transform from a list of coefficients, with mixed types.
AffineTransform{M,N}(coefs...) where {M,N} = AffineTransform{M,N}(promote(coefs...))
AffineTransform{M,N}(coefs::Tuple) where {M,N} = AffineTransform{M,N}(promote(coefs...))

# Type `T` of stored coefficients is specified.
AffineTransform{M,N,T}(coefs...) where {M,N,T} = AffineTransform{M,N,T}(coefs)

AffineTransform{M,N,T}(R::AffineTransform{M,N,T}) where {M,N,T} = R
AffineTransform{M,N,T}(R::AffineTransform{M,N}) where {M,N,T} =
    AffineTransform{M,N,T}(Tuple(R))

# Conversion from `TwoDimensional.AffineTransform2D`.
AffineTransform{2,2,T}(R::TwoDimensional.AffineTransform2D) where {T} =
    AffineTransform{2,2,T}(R.x, R.xx, R.xy,
                           R.y, R.yx, R.yy)
AffineTransform{2,2}(R::TwoDimensional.AffineTransform2D) = AffineTransform(R)
AffineTransform(R::TwoDimensional.AffineTransform2D{T}) where {T} =
    AffineTransform{2,2,T}(R)

# Conversion to `TwoDimensional.AffineTransform2D`.
TwoDimensional.AffineTransform2D(R::AffineTransform{2,2,T}) where {T} =
    TwoDimensional.AffineTransform2D{T}(R)
TwoDimensional.AffineTransform2D{T}(R::AffineTransform{2,2}) where {T} =
    TwoDimensional.AffineTransform2D{T}(R[2], R[3], R[1],
                                        R[5], R[6], R[4])

Base.convert(::Type{T}, x::T) where {T<:AffineTransform} = x
function Base.convert(::Type{T},
                      x::Union{AffineTransform,
                               TwoDimensional.AffineTransform2D}) where {T<:AffineTransform}
    return T(x)::T
end

function Base.convert(::Type{T},
                      x::AffineTransform{2,2}) where {T<:TwoDimensional.AffineTransform2D}
    return T(x)::T
end

# FIXME
Base.similar(R::AffineTransform{M,N}, ::Type{T}) where {M,N,T} = AffineTransform{M,N,T}(R)

# Abstract array API for affine transforms.

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

Base.getindex(R::AffineTransform, k::Integer) = getindex(R, Int(k)::Int)
@inline function Base.getindex(R::AffineTransform, k::Int)
    @boundscheck checkbounds(R, k)
    return @inbounds R.coefs[k]
end

Base.getindex(R::AffineTransform, i::Integer, j::Integer) =
    getindex(R, Int(i)::Int, Int(j)::Int)
@inline function Base.getindex(R::AffineTransform{M,N}, i::Int, j::Int) where {M,N}
    @boundscheck checkbounds(R, i, j)
    k = storage_index(R, i, j)
    return @inbounds R.coefs[k]
end

Base.checkbounds(R::AffineTransform, k::Integer) =
    checkbounds(Bool, R, k) || throw(BoundsError(R, k))
Base.checkbounds(R::AffineTransform, i::Integer, j::Integer) =
    checkbounds(Bool, R, i, j) || throw(BoundsError(R, (i, j)))

Base.checkbounds(::Type{Bool}, R::AffineTransform{M,N,T,L}, k::Integer) where {M,N,T,L} =
    ((1 ≤ k)&(k ≤ L))
Base.checkbounds(::Type{Bool}, R::AffineTransform{M,N}, i::Integer, j::Integer) where {M,N} =
    ((1 ≤ i)&(i ≤ M)&(0 ≤ j)&(j ≤ N))

# Apply the affine transform.

(R::AffineTransform{M,N})(x::Vararg{Any,N}) where {M,N} = R(x)
(R::AffineTransform{M,N})(x::SVector{N}) where {M,N} = SVector{M}(R(x.data))
(R::AffineTransform{M,N})(x::CartesianIndex{N}) where {M,N} = R(x.I)
@generated function (R::AffineTransform{M,N,T,L})(x::NTuple{N}) where {M,N,T,L}
    L == M*(N + 1) || error("bad number of coefficients")
    quote
        $(Expr(:meta, :inline))
        C = R.coefs
        return @inbounds $(encode_affine_op(:C, M, N, :x))
    end
end

# Generate code to apply an affine transform.
function encode_affine_op(C::Union{Symbol,Expr}, M::Int, N::Int, x::Symbol)
    code = Expr(:tuple)
    k = 0
    for i = 1:M
        k += 1
        local ex = Expr(:call, :(+), :($C[$k]))
        for j = 1:N
            k += 1
            push!(ex.args, :($C[$k]*$x[$j]))
        end
        push!(code.args, ex)
    end
    return code
end

"""
    storage(R)

Return the tuple of coefficients of the affine transform `R` in row-major order.

"""
storage(R::AffineTransform) = getfield(R, :coefs)
Base.Tuple(R::AffineTransform) = storage(R)

"""
    AffineTransform.storage_index(R, i, j) -> k
    AffineTransform.storage_index(typeof(R), i, j) -> k

Return the linear index `k ∈ 1:M*(N+1)` of the coefficient of the affine transform
`R::AffineTransform{M,N}` at row `i ∈ 1:M` and column `j ∈ 0:N`.

"""
storage_index(R::AffineTransform, i::Integer, j::Integer) = storage_index(typeof(R), i, j)
storage_index(T::Type{<:AffineTransform}, i::Integer, j::Integer) =
    storage_index(T, Int(i)::Int, Int(j)::Int)

# For efficiency, coefficients are stored in row-major order for i ∈ 1:M, j ∈ 0:N
storage_index(::Type{<:AffineTransform{M,N}}, i::Int, j::Int) where {M,N} =
    (N + 1)*i + j - N # FIXME check

"""
    offset(R) -> b

Return a tuple with the offset implemented by the affine transform `R`. This is the same as
applying `R` to a tuple of zeros.

"""
offset(R::AffineTransform{M,N}) where {M,N} = ntuple(i -> R[i,0], Val(M))

# Extract the linear part of an affine transform.
StaticArrays.SMatrix(R::AffineTransform{M,N,T}) where {M,N,T} = SMatrix{M,N,T}(R)
StaticArrays.SMatrix{M,N}(R::AffineTransform{M,N,T}) where {M,N,T} = SMatrix{M,N,T}(R)
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

# Extract the offset part of an affine transform in the form of a vector.
StaticArrays.SVector(R::AffineTransform{M,N,T}) where {M,N,T} = SVector{M,T}(R)
StaticArrays.SVector{M}(R::AffineTransform{M,N,T}) where {M,N,T} = SVector{M,T}(R)
StaticArrays.SVector{M,T}(R::AffineTransform{M,N}) where {M,N,T} =
    SVector{M,T}(ntuple(i -> @inbounds(R[i,0]), Val(M)))

function Base.Vector(R::AffineTransform{M,N,T}) where {M,N,T}
    b = Vector{T}(undef, M)
    @inbounds for i ∈ 1:M
        b[i] = R[i,0]
    end
    return b
end

# Check for equality.
for eq in (:(==), :isequal)
    @eval begin
        Base.$eq(A::AffineTransform, B::AffineTransform) = false
        Base.$eq(A::AffineTransform{M,N}, B::AffineTransform{M,N}) where {M,N} =
            $eq(storage(A), storage(B))
    end
end

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
Base.:(\)(R::AffineTransform{N,N}, x::NTuple{N}) where {N} = (R\SVector(x)).data
Base.:(\)(R::AffineTransform{N,N}, x::SVector{N}) where {N} = SMatrix(R)\(x - SVector(R))
Base.:(\)(R1::AffineTransform{N,N}, R2::AffineTransform{N}) where {N} = inv(R1)*R2

# Right division by an affine transform.
Base.:(/)(R1::AffineTransform{N,N}, R2::AffineTransform{N}) where {N} = R1*inv(R2)

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

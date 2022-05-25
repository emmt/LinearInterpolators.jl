"""
    compute_indices(bnd, off, rng, num::Val{N}) -> (j_1, j_2, ..., j_N)

yields the `N`-tuple of interpolation indices for boundary conditions `bnd` at
offset `off` and along a dimension where indices take values on range `rng`.
For instance, if `bnd` implements *flat* boundary conditions, then the `k`-th
returned index is computed as if given by:

    j_k = clamp(Int(off) + k, first(rng), last(rng))

It is assumed that the range `rng` is not empty.

"""
function compute_indices(bnd::BoundaryConditions,
                         off::Real,
                         len::Union{Integer,AbstractUnitRange{<:Integer}},
                         num::Val)
    return compute_indices(bnd, off, to_unit_range(len), num)
end

@generated function compute_indices(bnd::B,
                                    off::Real,
                                    rng::AbstractUnitRange{Int},
                                    num::Val{N}) where {N,B<:BoundaryConditions}
    return compute_indices(Expr, B, N)
end

"""
    compute_indices(::Type{Expr}, B::Type{<:BoundaryConditions}, N::Int)

yields the code of `compute_indices` method for boundary conditions of type `B`
and kernels of size `N`.

"""
compute_indices(::Type{Expr}, ::Type{Flat}, N::Int) = quote
    # We must avoid taking Int(off) if overflows may occur.  Integers up to
    # 1<<25 = 33_554_432 can be exactly represented by IEEE single precision
    # floating-point values (24-bit mantissa).
    $(expr_inline())
    i_first = first(rng)
    i_last = last(rng)
    if (i_first - 1 ≤ off)&(off ≤ i_last - N)
        # All indices in bounds, no needs to clamp.
        i_off = Int(off)
        return $(expr_tuple(i -> :(i_off + $i), N))
    elseif off ≤ i_first - N
        # All indices below lower bound.
        return $(expr_tuple(i -> :i_first, N))
    elseif off ≥ i_last - 1
        # All indices above upper bound.
        return $(expr_tuple(i -> :i_last, N))
    else
        # Some indices in bounds but not all, we must clamp and it is
        # probably safe to truncate to Int.
        i_off = Int(off)
        return $(expr_tuple(i -> :(clamp(i_off + $i, i_first, i_last)), N))
    end
end

expr_inline() = Expr(:meta, :inline)
expr_tuple(f::Function, N::Integer) = expr_tuple(f, Val(Int(N)))
expr_tuple(f::Function, N::Val) = Expr(:tuple, ntuple(f, N)...)

"""
    to_ntuple(::Type{N,T}, x)

yields a `N`-tuple of objects of type `T` built from `x` which can be a
`N`-tuple of objects of type `T` (which is returned) or a single object of type
`T` (which is repeated `N` times).

"""
to_ntuple(::Type{NTuple{N,T}}, x::NTuple{N,T}) where {T,N} = x
to_ntuple(::Type{NTuple{N,T}}, x::T) where {T,N} = ntuple(i -> x, Val(N))

"""
    to_unit_range(x)

yields an `Int` valued unit range.  Argument can be a dimension length or an
integer valued unit range.

"""
to_unit_range(len::Integer) = Base.OneTo{Int}(len)
to_unit_range(rng::AbstractUnitRange{Int}) = rng
to_unit_range(rng::AbstractUnitRange{<:Integer}) =
    convert(AbstractUnitRange{Int}, rng)

# FIXME: The `promote_eltype` function should be in `ArrayTools` package.
"""
    promote_eltype(x...)

yields the promoted element type of its arguments.  Arguments `x` may be
anything implementing the `eltype` method.

"""
promote_eltype(x) = eltype(x)
promote_eltype(x...) = promote_type(map(eltype, x)...)
promote_eltype() = UndefinedType

Base.promote_type(T::Type, ::Type{UndefinedType}) = T
Base.promote_type(::Type{UndefinedType}, T::Type) = T
Base.promote_type(::Type{UndefinedType}, ::Type{UndefinedType}) = UndefinedType

# FIXME: The `with_eltype` function should be in `ArrayTools` package.
"""
    with_eltype(T, A)

yields an object with the same contents as `A` but with element type `T`.
If `eltype(A) === T`, `A` itself is returned.

If `x` is not a type and `eltype(x)` is implemented, then `with_eltype(x,A)` is
equivalent to `with_eltype(eltype(x),A)`.

"""
with_eltype(x, A) = with_eltype(eltype(x), A)
with_eltype(T::Type, A) = error(
    "`with_eltype($T, ::$(typeof(A)))` is not implemented")

with_eltype(::Type{T}, A::AbstractArray{T}) where {T} = A
with_eltype(::Type{T}, A::AbstractArray) where {T} =
    convert(AbstractArray{T}, A)

with_eltype(::Type{T}, A::AbstractInterpolator{T}) where {T} = A
with_eltype(::Type{T}, A::AbstractInterpolator) where {T} =
    convert(AbstractInterpolator{T}, A)

with_eltype(::Type{T}, A::AffineTransform{M,N,T}) where {M,N,T} = A
with_eltype(::Type{T}, A::AffineTransform{M,N}) where {M,N,T} =
    convert(AffineTransform{M,N,T}, A)

with_eltype(::Type{T}, ker::Kernel{T}) where {T} = ker
with_eltype(::Type{T}, ker::Kernel) where {T} = convert(Kernel{T}, ker)

function with_eltype(::Type{NTuple{N,T}},
                     A::AbstractArray{NTuple{N,T},M}) where {T,M,N}
    return A
end
function with_eltype(::Type{NTuple{N,T}},
                     A::AbstractArray{NTuple{N,T′},M}) where {T,T′,M,N}
    B = similar(A, NTuple{N,T})
    @inbounds @simd for i in eachindex(A, B)
        B[i] = A[i]
    end
    return B
end

function with_eltype(::Type{Tuple{Vararg{T}}},
                     A::AbstractArray{NTuple{N,T},M}) where {T,M,N}
    return A
end
function with_eltype(::Type{Tuple{Vararg{T}}},
                     A::AbstractArray{NTuple{N,T′},M}) where {T,T′,M,N}
    B = similar(A, NTuple{N,T})
    @inbounds @simd for i in eachindex(A, B)
        B[i] = A[i]
    end
    return B
end

Base.eltype(A::AbstractInterpolator) = eltype(typeof(A))
Base.eltype(::Type{<:AbstractInterpolator{T}}) where {T} = T

Base.convert(::Type{T}, A::T) where {T<:AbstractInterpolator} = A
Base.convert(::Type{T}, A::AbstractInterpolator) where {T<:AbstractInterpolator} = T(A)

"""
    check_axes(A, dims)

yields whether the axes of array `A` are correct, that is one-based and with
lengths equal to the size `dims`.

"""
check_axes(A::AbstractArray, dims::Dims) = false
check_axes(A::AbstractArray{T,N}, dims::Dims{N}) where {T,N} =
    axes(A) == map(Base.OneTo, dims)

"""
    check_indices(I, R)

yields whether the indices in array `I` are all in the range `R`.

"""
function check_indices(I::AbstractArray{<:Integer},
                       R::AbstractUnitRange{<:Integer})
    flag = true
    i_first = Int(first(R))
    i_last  = Int(last(R))
    @inbounds for k in eachindex(I)
        ik = Int(I[k])
        flag &= (i_first ≤ ik)&(ik ≤ i_last)
    end
    return flag
end

function check_indices(I::AbstractArray{<:NTuple{M,Integer}},
                       R::AbstractUnitRange{<:Integer}) where {M}
    flag = true
    i_first = Int(first(R))
    i_last  = Int(last(R))
    @inbounds for k in eachindex(I)
        Ik = I[k]
        for m in 1:M
            Ikm = Int(Ik[m])
            flag &= (i_first ≤ Ikm)&(Ikm ≤ i_last)
        end
    end
    return flag
end

"""
    nth(n)

yields the string `"\$n\$(ordinal_suffix(n))"`.

"""
nth(n::Integer) = string(n)*ordinal_suffix(n)
# NOTE: `string(n)*ordinal_suffix(n)` is about 3 times faster (76.5ns) than
#       ``string(n,ordinal_suffix(n))` or `"$n$(ordinal_suffix(n))"` which are
#       equally slow (208ns).

"""
    ordinal_suffix(n)

yields the ordinal suffix `"-st"`, `"-nd"`, `"-rd"`, or `"-nt"` corresponding
to the value of the integer `n`.

"""
function ordinal_suffix(n::Integer)
    d = abs(n)%10
    return (d == 1 ? "-st" :
            d == 2 ? "-nd" :
            d == 3 ? "-rd" : "-th")
end

argument_error(msg::String) = throw(ArgumentError(msg))
@noinline argument_error(args...) = argument_error(string(args...))

dimension_mismatch(msg::String) = throw(DimensionMismatch(msg))
@noinline dimension_mismatch(args...) = dimension_mismatch(string(args...))

#
# methods.jl -
#
# Implement boundary conditions and other common methods.
#

const Axis = Union{Integer,AbstractUnitRange{<:Integer}}
to_axis(len::Integer) = Base.OneTo{Int}(len)
to_axis(r::Base.OneTo{Int}) = r
to_axis(r::Base.OneTo{<:Integer}) = to_axis(length(r))
to_axis(r::AbstractUnitRange{Int}) = r
to_axis(r::AbstractUnitRange{<:Integer}) = Int(first(r)):Int(last(r))

eltype(A::AbstractInterpolator) = eltype(typeof(A))
eltype(::Type{<:AbstractInterpolator{T}}) where {T} = T

eltype(B::Boundaries) = eltype(typeof(B))
eltype(::Type{<:Boundaries{T}}) where {T} = T

first(B::Boundaries) = first(axis(B))
last(B::Boundaries) = last(axis(B))
clamp(i, B::Boundaries) = clamp(i, first(B), last(B))

"""
    axis(B::Boundaries)

yields the range of indices of the interpolated dimension for the boundary
conditions `B`.

"""
axis(B::Flat) = B.axis
axis(B::SafeFlat) = B.axis

# Constructors.
for B in (:Flat, :SafeFlat)
    @eval begin
        $B(::Kernel{T,S}, axis::Axis) where {T<:AbstractFloat,S} = $B{T,S}(axis)
        $B{T}(::Kernel{T,S}, axis::Axis) where {T<:AbstractFloat,S} = $B{T,S}(axis)
        $B{T,S}(::Kernel{T,S}, axis::Axis) where {T<:AbstractFloat,S} = $B{T,S}(axis)
        $B{T,S}(axis::Axis) where {T<:AbstractFloat,S} = $B{T,S}(to_axis(axis))
        $B{T,S}(axis::R) where {T<:AbstractFloat,S,R<:AbstractUnitRange{Int}} = $B{T,S,R}(axis)
    end
end

"""
    check_axes(A, inds) -> bool

yields whether array `A` has the same indices as specified by `inds`, a tuple
of index ranges or of integers.  If `inds` is a tuple of integers, they are
interpreted as the lengths of 1-based unit ranges.

"""
check_axes(A::AbstractArray, inds) = false
check_axes(A::AbstractArray{<:Any,N}, dims::Dims{N}) where {N} =
    check_axes(A, map(to_axis, dims))

@inline function check_axes(A::AbstractArray{<:Any,N},
                            inds::NTuple{N,AbstractUnitRange{<:Integer}}) where {N}
    flag = true
    for i in 1:N
        flag &= axes(A,i) == inds[i]
    end
    return flag
end

"""
    check_size(dims) -> nelem

yields the product of the elements of `dims` throwing an exception if any of
these is not a valid dimension length.

"""
function check_size(dims::NTuple{N,Integer}) where {N}
    nelem = 1
    for i in 1:N
        (d = Int(dims[i])) ≥ 0 || error("invalid dimension")
        nelem *= d
    end
    return nelem
end

# FIXME: move this in InterpolationKernel?
"""
    with_eltype(T, x)

yields an analog of object `x` whose elements have type `T`.

"""
with_eltype(::Type{T}, ker::Kernel{T}) where {T<:AbstractFloat} = ker
with_eltype(::Type{T}, ker::Kernel) where {T<:AbstractFloat} = Kernel{T}(ker)
with_eltype(::Type{T}, A::AbstractArray{T}) where {T} = begin
    assert_concrete(T)
    return A
end
with_eltype(::Type{T}, A::AbstractArray) where {T} = begin
    assert_concrete(T)
    return convert(AbstractArray{T}, A)
end
with_eltype(::Type{T}, r::Base.OneTo{T}) where {T} = r
with_eltype(::Type{T}, r::Base.OneTo) where {T<:Integer} =
    Base.OneTo{T}(length(r))
with_eltype(::Type{T}, r::AbstractUnitRange{T}) where {T} = r
with_eltype(::Type{T}, r::AbstractUnitRange) where {T} =
    convert(T, first(r)):convert(T, last(r))
with_eltype(::Type{T}, r::StepRange{T,T}) where {T} = r
with_eltype(::Type{T}, r::StepRange) where {T} =
    convert(T, first(r)):convert(T, step(r)):convert(T, last(r))

assert_concrete(::Type{T}) where {T} =
    isconcretetype(T) || bad_argument("type ", T, " is not concrete")

"""
    sum_of_terms(ex)

yields the expression that is the sum of the terms in the tuple/array of
expressions/symbols `ex`.

"""
sum_of_terms(ex) =
    (length(ex) > 1 ? Expr(:call, :+, ex...) :
     length(ex) == 1 ? ex[1] : :nothing)

"""
    promote_kernel(ker, A...)

yields kernel `ker` promoted to a suitable floating-point type for
interpolating with arrays `A...`.

A single type argument `T` may also be provided:

    promote_kernel(ker, T)

"""
promote_kernel(ker::Kernel) = ker
promote_kernel(ker::Kernel, A::AbstractArray) = promote_kernel(ker, eltype(A))
@inline promote_kernel(ker::Kernel, A::AbstractArray...) =
    promote_kernel(ker, promote_type(map(eltype, A)...))

promote_kernel(ker::Kernel{T}, ::Type{T}) where {T} = ker
promote_kernel(ker::Kernel,    ::Type{T}) where {T<:Integer} = ker
promote_kernel(ker::Kernel,    ::Type{T}) where {T<:Real} =
    convert(Kernel{promote_type(T, eltype(ker))}, ker)
promote_kernel(ker::Kernel,    ::Type{T}) where {T<:Complex} =
    promote_kernel(ker, real(T))

"""
    compute_indices(B, off) -> inds

yields a `S`-tuple of indices `(off+1,...,off+S)` taking into account the
boundary conditions implemented by `B`.

""" compute_indices

#
# We want to apply "flat" boundary conditions to the indices `t+1`, `t+2`, ...,
# `t+S` to interpolate an array along axis `jmin:jmax` with a kernel of size
# `S`.  The different possibilities for the indices in the interpolated
# dimension are:
#
# - if `t + S ≤ jmin` (that is, `t ≤ t1` with `t1 = jmin - S`), then all
#   indices in the interpolated array are `jmin`;
#
# - if `t + 1 ≥ jmin` and `t + S ≤ jmax` (that is, `t2 ≤ t ≤ t3` with `t2 =
#   jmin - 1` and `t3 = jmax - S`), then the indices in the interpolated array
#   are `(t+1,t+2,...,t+S)`;
#
# - if `t + 1 ≥ jmax` (that is, `t ≥ t4` with `t4 = jmax - 1`), then all
#   indices in the interpolated array are `jmax`;
#
# - otherwise the indices in the interpolated array are
#   `(clamp(t+1,jmin,jmax),...,clamp(t+S,jmin,jmax))`.
#
# The last case costs the most and the second case is likely to occurs the most
# often; so the logic is implemented for the `SafeFlat` boundary conditions as
# follows:
#
#     if t ≤ t3
#         if t ≥ t2
#             j = Int(t)
#             return (j+1,j+2,...,j+S)
#         elseif t ≤ t1
#             return (jmin,jmin,...)
#         end
#     elseif t ≥ t4
#         return (jmax,jmax,...)
#     end
#     j = Int(t)
#     return (clamp(j+1,jmin,jmax),...,clamp(j+S,jmin,jmax))
#
# The offset `t` returned by `compute_offset_and_weights` is a floating-point
# value equal to its integer part (it is not yet converted to an integer to
# avoid integer overflows).  These overflows are avoided by the `SaveFlat`
# boundary conditions and are ignored by the `Flat` boundary conditions.  As a
# consequence, the `SaveFlat` boundary conditions are safer but slower than the
# `Flat` ones.
#
# Ignoring integer overflows, the logic implemented for the `Flat` boundary
# conditions is:
#
#     j = Int(t) # assume no integer overflows
#     if j2 ≤ j ≤ j3
#         return (j+1,j+2,...,j+S)
#     else
#         return (clamp(j+1,jmin,jmax),...,clamp(j+S,jmin,jmax))
#     end
#
@generated function compute_indices(B::Flat{T,S}, off::T) where {T,S}
    inner_indices = Expr(:tuple, [:(j + $s) for s in 1:S]...)
    clamped_indices = Expr(:tuple, [:(clamp(j + $s, B)) for s in 1:S]...)
    quote
        $(Expr(:meta, :inline))
        j = Int(off) # assume no integer overflows
        (B.off_2 ≤ j ≤ B.off_3) ? $inner_indices : $clamped_indices
    end
end
#
@generated function compute_indices(B::SafeFlat{T,S}, off::T) where {T,S}
    same_indices = Expr(:tuple, [:j for s in 1:S]...)
    inner_indices = Expr(:tuple, [:(j + $s) for s in 1:S]...)
    clamped_indices = Expr(:tuple, [:(clamp(j + $s, B)) for s in 1:S]...)
    quote
        $(Expr(:meta, :inline))
        if off ≤ B.off_3
            if off ≥ B.off_2
                j = Int(off)
                return $inner_indices
            elseif off ≤ B.off_1
                j = first(B.axis)
                return $same_indices
            end
        elseif off ≥ B.off_4
            j = last(B.axis)
            return $same_indices
        end
        j = Int(off)
        return $clamped_indices
    end
end

@noinline function bad_type_parameter(sym::Symbol, val, T::Type)
    if isa(val, T)
        bad_argument("bad value $val for type parameter $sym")
    else
        bad_argument("bad type $(typeof(val)) for type parameter $sym")
    end
end

bad_argument(mesg::AbstractString) = throw(ArgumentError(mesg))
@noinline bad_argument(args...) = bad_argument(string(args...))

dimension_mismatch(mesg::AbstractString) = throw(DimensionMismatch(mesg))
@noinline dimension_mismatch(args...) = dimension_mismatch(string(args...))

#
# boundaries --
#
# Implement boundary conditions.
#
#------------------------------------------------------------------------------
#
# This file is part of the LinearInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2016-2021, Éric Thiébaut.
#

eltype(B::Limits) = eltype(typeof(B))
eltype(::Type{<:Limits{T}}) where {T} = T
length(B::Limits) = B.len
size(B::Limits) = (B.len,)
size(B::Limits, i::Integer) =
    (i == 1 ? B.len : i > 1 ? 1 : throw(BoundsError()))
first(B::Limits) = 1
last(B::Limits) = B.len
clamp(i, B::Limits) = clamp(i, first(B), last(B))

"""
    limits(ker, len) -> lim::Limits

yields and instance of `Limits` combining the boundary conditions of kernel
`ker` with the lengh `len` of the dimension of interpolation in the
interpolated array.

All interpolation limits inherit from the abstract type `Limits{T}` where `T`
is the floating-point type.  Interpolation limits are the combination of an
extrapolation method and the length of the dimension to interpolate.

""" limits

limits(::Kernel{T,S,Flat}, len::Integer) where {T,S} =
    FlatLimits{T}(len)

limits(::Kernel{T,S,SafeFlat}, len::Integer) where {T,S} =
    SafeFlatLimits{T}(prevfloat(T(2       - S/2)),
                      nextfloat(T(len - 1 + S/2)), len)

boundaries(::FlatLimits) = Flat
boundaries(::SafeFlatLimits) = SafeFlat

inferior(B::SafeFlatLimits) = B.inf
superior(B::SafeFlatLimits) = B.sup

"""
    getcoefs(ker, lim, x) -> j1, j2, ..., w1, w2, ...

yields the indexes of the neighbors and the corresponding interpolation weights
for interpolating at position `x` by kernel `ker` with the limits implemented
by `lim`.

If `x` is not a scalar of the same floating-point type, say `T`, as the kernel
`ker`, it is converted to `T` by calling
[`LinearInterpolators.convert_coordinate(T,x)`](@ref).  This latter method may
be extended for non-standard numerical types of `x`.

""" getcoefs

# The following is ugly (too specific) but necessary to avoid ambiguities.
for (B,L) in ((:Flat,     :FlatLimits),
              (:SafeFlat, :SafeFlatLimits),)
    @eval begin
        @inline function getcoefs(ker::Kernel{T,S,$B},
                                  lim::$L{T},
                                  x) where {T<:AbstractFloat,S}
            getcoefs(ker, lim, convert_coordinate(T, x)::T)
        end
    end
end

# Specialized code for S = 1 (i.e., take nearest neighbor).
@inline function getcoefs(ker::Kernel{T,1,Flat},
                          lim::FlatLimits{T},
                          x::T) where {T<:AbstractFloat}
    r = round(x)
    j1 = clamp(trunc(Int, r), lim)
    w1 = getweights(ker, x - r)
    return j1, w1
end

# For S > 1, code is automatically generated.
@generated function getcoefs(ker::Kernel{T,S,Flat},
                             lim::FlatLimits{T},
                             x::T) where {T<:AbstractFloat,S}

    c = ((S + 1) >> 1)
    J = [Symbol(:j_,i) for i in 1:S]
    setindices = ([:(  $(J[i]) = $(J[c]) - $(c - i)  ) for i in 1:c-1]...,
                  [:(  $(J[i]) = $(J[c]) + $(i - c)  ) for i in c+1:S]...)
    clampindices = [:( $(J[i]) = clamp($(J[i]), lim) ) for i in 1:S]
    if isodd(S)
        nearest = :(f = round(x))
    else
        nearest = :(f = floor(x))
    end

    quote
        $(Expr(:meta, :inline))
        $(nearest)
        $(J[c]) = trunc(Int, f)
        $(setindices...)
        if $(J[1]) < first(lim) || $(J[S]) > last(lim)
            $(clampindices...)
        end
        return ($(J...), getweights(ker, x - f)...)
    end
end

@generated function getcoefs(ker::Kernel{T,S,SafeFlat},
                             lim::SafeFlatLimits{T},
                             x::T) where {T<:AbstractFloat,S}

    J = [Symbol(:j_,i) for i in 1:S]
    W = [Symbol(:w_,i) for i in 1:S]
    sameindices = [:(  $(J[i]) = j                   ) for i in 1:S]
    beyondfirst = (:(  j = first(lim)                ),
                   sameindices...,
                   [:( $(W[i]) = zero(T)             ) for i in 1:S-1]...,
                   :(  $(W[S]) = one(T)              ))
    beyondlast = (:(   j = last(lim)                 ),
                  sameindices...,
                  :(   $(W[1]) = one(T)              ),
                  [:(  $(W[i]) = zero(T)             ) for i in 2:S]...)
    c = ((S + 1) >> 1)
    setindices = ([:(  $(J[i]) = $(J[c]) - $(c - i)  ) for i in 1:c-1]...,
                  [:(  $(J[i]) = $(J[c]) + $(i - c)  ) for i in c+1:S]...)
    clampindices = [:( $(J[i]) = clamp($(J[i]), lim) ) for i in 1:S]
    if isodd(S)
        nearest = :(f = round(x))
    else
        nearest = :(f = floor(x))
    end

    quote
        $(Expr(:meta, :inline))
        if x ≤ inferior(lim)
            $(beyondfirst...)
        elseif x ≥ superior(lim)
            $(beyondlast...)
        else
            $(nearest)
            $(J[c]) = trunc(Int, f)
            $(setindices...)
            if $(J[1]) < first(lim) || $(J[S]) > last(lim)
                $(clampindices...)
            end
            $(Expr(:tuple, W...)) = getweights(ker, x - f)
        end
        return ($(J...), $(W...))
    end
end

"""
    LinearInterpolators.convert_coordinate(T, c) -> x::T

yields interpolation coordinate `c` to floating-point type `T`.  The default
behavior is to yield `convert(T,c)` but this method may be extended to cope
with non-standard numeric types.

"""
convert_coordinate(T::Type{<:AbstractFloat}, c::Number) = convert(T, c)

#
# interp/safeflat.jl --
#
# Implement safe "Flat" boundary conditions.  The operations are "safe" in the
# sense that no `InexactError` get thrown even for very large extrapolation
# distances.
#
#------------------------------------------------------------------------------
#
# This file is part of the LinearInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2016-2018, Éric Thiébaut.
#

struct SafeFlatLimits{T<:AbstractFloat} <: Limits{T}
    inf::T   # bound for all neighbors before range
    sup::T   # bound for all neighbors after range
    len::Int # length of dimension in interpolated array
    function (::Type{SafeFlatLimits{T}})(inf, sup, len::Integer) where {T}
        @assert inf < sup
        @assert len ≥ 1
        new{T}(inf, sup, len)
    end
end

limits(::Kernel{T,S,SafeFlat}, len::Integer) where {T,S} =
    SafeFlatLimits{T}(prevfloat(T(2       - S/2)),
                      nextfloat(T(len - 1 + S/2)), len)

boundaries(::SafeFlatLimits) = SafeFlat

inferior(B::SafeFlatLimits) = B.inf
superior(B::SafeFlatLimits) = B.sup

@generated function getcoefs(ker::Kernel{T,S,SafeFlat},
                             lim::SafeFlatLimits{T}, x::T) where {S,T}

    J, W = make_varlist(:j, S), make_varlist(:w, S)
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

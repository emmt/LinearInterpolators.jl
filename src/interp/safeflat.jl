#
# interp/safeflat.jl -
#
# Implement safe "Flat" boundary conditions.  The operations are "safe" in the
# sense that no `InexactError` get thrown even for very large extrapolation
# distances.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2017-2018, Éric Thiébaut.
# This file is part of TiPi.  All rights reserved.
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

inferior(B::SafeFlatLimits{T}) where {T} = B.inf
superior(B::SafeFlatLimits{T}) where {T} = B.sup

@inline function getcoefs(ker::Kernel{T,1,SafeFlat},
                          lim::SafeFlatLimits{T}, x::T) where {T}
    if x ≤ inferior(lim)
        j1 = first(lim)
        w1 = one(T)
    elseif x ≥ superior(lim)
        j1 = last(lim)
        w1 = one(T)
    else
        r = round(x)
        j1 = clamp(trunc(Int, r), lim)
        w1 = getweights(ker, x - r)
    end
    return j1, w1
end

@inline function getcoefs(ker::Kernel{T,2,SafeFlat},
                          lim::SafeFlatLimits{T}, x::T) where {T}
    if x ≤ inferior(lim)
        j1 = j2 = first(lim)
        w1 = zero(T)
        w2 = one(T)
    elseif x ≥ superior(lim)
        j1 = j2 = last(lim)
        w1 = one(T)
        w2 = zero(T)
    else
        f = floor(x)
        j1 = trunc(Int, f)
        j2 = j1 + 1
        if j1 < first(lim) || j2 > last(lim)
            j1 = clamp(j1, lim)
            j2 = clamp(j2, lim)
        end
        w1, w2 = getweights(ker, x - f)
    end
    return j1, j2, w1, w2
end

@inline function getcoefs(ker::Kernel{T,3,SafeFlat},
                          lim::SafeFlatLimits{T}, x::T) where {T}
    if x ≤ inferior(lim)
        j1 = j2 = j3 = first(lim)
        w1 = w2 = zero(T)
        w3 = one(T)
    elseif x ≥ superior(lim)
        j1 = j2 = j3 = last(lim)
        w1 = one(T)
        w2 = w3 = zero(T)
    else
        r = round(x)
        j2 = trunc(Int, r)
        j1, j3 = j2 - 1, j2 + 1
        if j1 < first(lim) || j3 > last(lim)
            j1 = clamp(j1, lim)
            j2 = clamp(j2, lim)
            j3 = clamp(j3, lim)
        end
        w1, w2, w3 = getweights(ker, x - r)
    end
    return j1, j2, j3, w1, w2, w3
end

@inline function getcoefs(ker::Kernel{T,4,SafeFlat},
                          lim::SafeFlatLimits{T}, x::T) where {T}
    if x ≤ inferior(lim)
        j1 = j2 = j3 = j4 = first(lim)
        w1 = w2 = w3 = zero(T)
        w4 = one(T)
    elseif x ≥ superior(lim)
        j1 = j2 = j3 = j4 = last(lim)
        w1 = one(T)
        w2 = w3 = w4 = zero(T)
    else
        f = floor(x)
        j2 = trunc(Int, f)
        j1, j3, j4 = j2 - 1, j2 + 1, j2 + 2
        if j1 < first(lim) || j4 > last(lim)
            j1 = clamp(j1, lim)
            j2 = clamp(j2, lim)
            j3 = clamp(j3, lim)
            j4 = clamp(j4, lim)
        end
        w1, w2, w3, w4 = getweights(ker, x - f)
    end
    return j1, j2, j3, j4, w1, w2, w3, w4
end

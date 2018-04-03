#
# interp/flat.jl -
#
# Implement "Flat" boundary conditions.  These conditions implies that
# extrapolated positions yield the same value of the interpolated array at
# the nearest position.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2017-2018, Éric Thiébaut.
# This file is part of TiPi.  All rights reserved.
#

struct FlatLimits{T<:AbstractFloat} <: Limits{T}
    len::Int # length of dimension in interpolated array
    function (::Type{FlatLimits{T}})(len::Integer) where {T}
        @assert len ≥ 1
        new{T}(len)
    end
end

limits(::Kernel{T,S,Flat}, len::Integer) where {T,S} =
    FlatLimits{T}(len)

boundaries(::FlatLimits) = Flat

@inline function getcoefs(ker::Kernel{T,1,Flat},
                          lim::FlatLimits{T}, x::T) where {T}
    r = round(x)
    j1 = clamp(trunc(Int, r), lim)
    w1 = getweights(ker, x - r)
    return j1, w1
end

@inline function getcoefs(ker::Kernel{T,2,Flat},
                          lim::FlatLimits{T}, x::T) where {T}
    f = floor(x)
    j1 = trunc(Int, f)
    j2 = j1 + 1
    if j1 < first(lim) || j2 > last(lim)
        j1 = clamp(j1, lim)
        j2 = clamp(j2, lim)
    end
    w1, w2 = getweights(ker, x - f)
    return j1, j2, w1, w2
end

@inline function getcoefs(ker::Kernel{T,3,Flat},
                          lim::FlatLimits{T}, x::T) where {T}
    r = round(x)
    j2 = trunc(Int, r)
    j1, j3 = j2 - 1, j2 + 1
    if j1 < first(lim) || j3 > last(lim)
        j1 = clamp(j1, lim)
        j2 = clamp(j2, lim)
        j3 = clamp(j3, lim)
    end
    w1, w2, w3 = getweights(ker, x - r)
    return j1, j2, j3, w1, w2, w3
end

@inline function getcoefs(ker::Kernel{T,4,Flat},
                          lim::FlatLimits{T}, x::T) where {T}
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
    return j1, j2, j3, j4, w1, w2, w3, w4
end

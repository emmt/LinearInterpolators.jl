#
# interp/flat.jl --
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

# Specialized code for S = 1 (i.e., take nearest neighbor).
@inline function getcoefs(ker::Kernel{T,1,Flat},
                          lim::FlatLimits{T}, x::T) where {T}
    r = round(x)
    j1 = clamp(trunc(Int, r), lim)
    w1 = getweights(ker, x - r)
    return j1, w1
end

# For S > 1, code is automatically generated.
@generated function getcoefs(ker::Kernel{T,S,Flat},
                             lim::FlatLimits{T}, x::T) where {T,S}

    m = S >> 1
    J = make_varlist(:j, S)
    setindices = ([:(  $(J[i]) = $(J[m]) - $(m - i)  ) for i in 1:m-1]...,
                  [:(  $(J[i]) = $(J[m]) + $(i - m)  ) for i in m+1:S]...)
    clampindices = [:( $(J[i]) = clamp($(J[i]), lim) ) for i in 1:S]

    quote
        $(Expr(:meta, :inline))
        f = floor(x)
        $(J[m]) = trunc(Int, f)
        $(setindices...)
        if $(J[1]) < first(lim) || $(J[S]) > last(lim)
            $(clampindices...)
        end
        return ($(J...), getweights(ker, x - f)...)
    end
end

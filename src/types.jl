#
# types.jl -
#
# Definitions of common type in `LinearInterpolators`.
#
#------------------------------------------------------------------------------
#
# This file is part of the LinearInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2016-2021, Éric Thiébaut.
#

"""
    Limits{T<:AbstractFloat}
Abstract type for defining boundary conditions.
"""
abstract type Limits{T<:AbstractFloat} end # FIXME: add parameter S


"""

`FlatLimits` are boundary conditions that implies that extrapolated positions
yield the same value of the interpolated array at the nearest position.

"""
struct FlatLimits{T<:AbstractFloat} <: Limits{T}
    len::Int # length of dimension in interpolated array
    function FlatLimits{T}(len::Integer) where {T}
        @assert len ≥ 1
        new{T}(len)
    end
end

"""

`SafeFlatLimits` are boundary conditions that implies that extrapolated
positions yield the same value of the interpolated array at the nearest
position.  Compared to `FlatLimits`, the operations are "safe" in the sense
that no `InexactError` get thrown even for very large extrapolation distances.

"""
struct SafeFlatLimits{T<:AbstractFloat} <: Limits{T}
    inf::T   # bound for all neighbors before range
    sup::T   # bound for all neighbors after range
    len::Int # length of dimension in interpolated array
    function SafeFlatLimits{T}(inf, sup, len::Integer) where {T}
        @assert inf < sup
        @assert len ≥ 1
        new{T}(inf, sup, len)
    end
end

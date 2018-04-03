#
# interp/unidimensional.jl --
#
# Unidimensional interpolation (the result may be multi-dimensional).
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2017-2018, Éric Thiébaut.
# This file is part of TiPi.  All rights reserved.
#

#------------------------------------------------------------------------------
# Out-of place versions.

function apply_direct(ker::Kernel{T,S,B}, x::AbstractArray{T,N},
                      src::AbstractVector{T}) where {T,S,B,N}
    return apply_direct!(Array{T}(size(x)), ker, x, src)
end

function apply_adjoint(ker::Kernel{T,S,B}, x::AbstractArray{T,N},
                       src::AbstractArray{T,N}, len::Integer) where {T,S,B,N}
    return apply_adjoint!(Array{T}(len), ker, x, src; clr = true)
end

#------------------------------------------------------------------------------
# In-place direct operation.

function apply_direct!(dst::AbstractArray{T,N},
                       ker::Kernel{T,1,B},
                       x::AbstractArray{T,N},
                       src::AbstractVector{T}) where {T,B,N}
    @assert size(dst) == size(x)
    lim = limits(ker, length(src))
    @inbounds for i in eachindex(dst, x)
        j1, w1 = getcoefs(ker, lim, x[i])
        dst[i] = w1*src[j1]
    end
    return dst
end

function apply_direct!(dst::AbstractArray{T,N},
                       ker::Kernel{T,2,B},
                       x::AbstractArray{T,N},
                       src::AbstractVector{T}) where {T,B,N}
    @assert size(dst) == size(x)
    lim = limits(ker, length(src))
    @inbounds for i in eachindex(dst, x)
        j1, j2, w1, w2 = getcoefs(ker, lim, x[i])
        dst[i] = w1*src[j1] + w2*src[j2]
    end
    return dst
end

function apply_direct!(dst::AbstractArray{T,N},
                       ker::Kernel{T,3,B},
                       x::AbstractArray{T,N},
                       src::AbstractVector{T}) where {T,B,N}
    @assert size(dst) == size(x)
    lim = limits(ker, length(src))
    @inbounds for i in eachindex(dst, x)
        j1, j2, j3, w1, w2, w3 = getcoefs(ker, lim, x[i])
        dst[i] = w1*src[j1] + w2*src[j2] + w3*src[j3]
    end
    return dst
end

function apply_direct!(dst::AbstractArray{T,N},
                       ker::Kernel{T,4,B},
                       x::AbstractArray{T,N},
                       src::AbstractVector{T}) where {T,B,N}
    @assert size(dst) == size(x)
    # FIXME: return apply_direct!(dst, ker, (i) -> x[i], src)
    lim = limits(ker, length(src))
    @inbounds for i in eachindex(dst, x)
        j1, j2, j3, j4, w1, w2, w3, w4 = getcoefs(ker, lim, x[i])
        dst[i] = w1*src[j1] + w2*src[j2] + w3*src[j3] + w4*src[j4]
    end
    return dst
end

#------------------------------------------------------------------------------
# In-place adjoint operation.

function apply_adjoint!(dst::AbstractVector{T},
                        ker::Kernel{T,1,B},
                        x::AbstractArray{T,N},
                        src::AbstractArray{T,N};
                        clr::Bool = true) where {T,B,N}
    @assert size(src) == size(x)
    if clr
        fill!(dst, zero(T))
    end
    lim = limits(ker, length(dst))
    @inbounds for i in eachindex(src, x)
        j1, w1 = getcoefs(ker, lim, x[i])
        dst[j1] += w1*src[i]
    end
    return dst
end

function apply_adjoint!(dst::AbstractVector{T},
                        ker::Kernel{T,2,B},
                        x::AbstractArray{T,N},
                        src::AbstractArray{T,N};
                        clr::Bool = true) where {T,B,N}
    @assert size(src) == size(x)
    if clr
        fill!(dst, zero(T))
    end
    lim = limits(ker, length(dst))
    @inbounds for i in eachindex(src, x)
        j1, j2, w1, w2 = getcoefs(ker, lim, x[i])
        a = src[i]
        dst[j1] += w1*a
        dst[j2] += w2*a
    end
    return dst
end

function apply_adjoint!(dst::AbstractVector{T},
                        ker::Kernel{T,3,B},
                        x::AbstractArray{T,N},
                        src::AbstractArray{T,N};
                        clr::Bool = true) where {T,B,N}
    @assert size(src) == size(x)
    if clr
        fill!(dst, zero(T))
    end
    lim = limits(ker, length(dst))
    @inbounds for i in eachindex(src, x)
        j1, j2, j3, w1, w2, w3 = getcoefs(ker, lim, x[i])
        a = src[i]
        dst[j1] += w1*a
        dst[j2] += w2*a
        dst[j3] += w3*a
    end
    return dst
end

function apply_adjoint!(dst::AbstractVector{T},
                        ker::Kernel{T,4,B},
                        x::AbstractArray{T,N},
                        src::AbstractArray{T,N};
                        clr::Bool = true) where {T,B,N}
    @assert size(src) == size(x)
    # FIXME: return apply_adjoint!(dst, ker, (i) -> x[i], src; clr = clr)
    if clr
        fill!(dst, zero(T))
    end
    lim = limits(ker, length(dst))
    @inbounds for i in eachindex(src, x)
        j1, j2, j3, j4, w1, w2, w3, w4 = getcoefs(ker, lim, x[i])
        a = src[i]
        dst[j1] += w1*a
        dst[j2] += w2*a
        dst[j3] += w3*a
        dst[j4] += w4*a
    end
    return dst
end

#------------------------------------------------------------------------------
# In-place direct operation at positions given by a function.

function apply_direct!(dst::AbstractArray{T,N},
                       ker::Kernel{T,1,B},
                       f::Function,
                       src::AbstractVector{T}) where {T,B,N}
    lim = limits(ker, length(src))
    @inbounds for i in eachindex(dst)
        x = f(i) :: T
        j1, w1 = getcoefs(ker, lim, x)
        dst[i] = w1*src[j1]
    end
    return dst
end

function apply_direct!(dst::AbstractArray{T,N},
                       ker::Kernel{T,2,B},
                       f::Function,
                       src::AbstractVector{T}) where {T,B,N}
    lim = limits(ker, length(src))
    @inbounds for i in eachindex(dst)
        x = f(i) :: T
        j1, j2, w1, w2 = getcoefs(ker, lim, x)
        dst[i] = w1*src[j1] + w2*src[j2]
    end
    return dst
end

function apply_direct!(dst::AbstractArray{T,N},
                       ker::Kernel{T,3,B},
                       f::Function,
                       src::AbstractVector{T}) where {T,B,N}
    lim = limits(ker, length(src))
    @inbounds for i in eachindex(dst)
        x = f(i) :: T
        j1, j2, j3, w1, w2, w3 = getcoefs(ker, lim, x)
        dst[i] = w1*src[j1] + w2*src[j2] + w3*src[j3]
    end
    return dst
end

function apply_direct!(dst::AbstractArray{T,N},
                   ker::Kernel{T,4,B},
                              f::Function,
                   src::AbstractVector{T}) where {T,B,N}
    lim = limits(ker, length(src))
    @inbounds for i in eachindex(dst)
        x = f(i) :: T
        j1, j2, j3, j4, w1, w2, w3, w4 = getcoefs(ker, lim, x)
        dst[i] = w1*src[j1] + w2*src[j2] + w3*src[j3] + w4*src[j4]
    end
    return dst
end

#------------------------------------------------------------------------------
# In-place adjoint operation at positions given by a function.

function apply_adjoint!(dst::AbstractVector{T},
                        ker::Kernel{T,1,B},
                        f::Function,
                        src::AbstractArray{T,N};
                        clr::Bool = true) where {T,B,N}
    if clr
        fill!(dst, zero(T))
    end
    lim = limits(ker, length(dst))
    @inbounds for i in eachindex(src)
        x = f(i) :: T
        j1, w1 = getcoefs(ker, lim, x)
        dst[j1] += w1*src[i]
    end
    return dst
end

function apply_adjoint!(dst::AbstractVector{T},
                        ker::Kernel{T,2,B},
                        f::Function,
                        src::AbstractArray{T,N};
                        clr::Bool = true) where {T,B,N}
    if clr
        fill!(dst, zero(T))
    end
    lim = limits(ker, length(dst))
    @inbounds for i in eachindex(src)
        x = f(i) :: T
        j1, j2, w1, w2 = getcoefs(ker, lim, x)
        a = src[i]
        dst[j1] += w1*a
        dst[j2] += w2*a
    end
    return dst
end

function apply_adjoint!(dst::AbstractVector{T},
                        ker::Kernel{T,3,B},
                        f::Function,
                        src::AbstractArray{T,N};
                        clr::Bool = true) where {T,B,N}
    if clr
        fill!(dst, zero(T))
    end
    lim = limits(ker, length(dst))
    @inbounds for i in eachindex(src)
        x = f(i) :: T
        j1, j2, j3, w1, w2, w3 = getcoefs(ker, lim, x)
        a = src[i]
        dst[j1] += w1*a
        dst[j2] += w2*a
        dst[j3] += w3*a
    end
    return dst
end

function apply_adjoint!(dst::AbstractVector{T},
                        ker::Kernel{T,4,B},
                        f::Function,
                        src::AbstractArray{T,N};
                        clr::Bool = true) where {T,B,N}
    if clr
        fill!(dst, zero(T))
    end
    lim = limits(ker, length(dst))
    @inbounds for i in eachindex(src)
        x = f(i) :: T
        j1, j2, j3, j4, w1, w2, w3, w4 = getcoefs(ker, lim, x)
        a = src[i]
        dst[j1] += w1*a
        dst[j2] += w2*a
        dst[j3] += w3*a
        dst[j4] += w4*a
    end
    return dst
end

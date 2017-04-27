#
# interp/unidimensional.jl --
#
# Unidimensional interpolation (the result may be multi-dimensional).
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2017, Éric Thiébaut.
# This file is part of TiPi.  All rights reserved.
#

#------------------------------------------------------------------------------
# Out-of place versions.

function apply_direct{T,S,B,N}(ker::Kernel{T,S,B},
                               x::AbstractArray{T,N},
                               A::Vector{T})
    return apply_direct!(Array{T}(size(x)), ker, x, A)
end

function apply_adjoint{T,S,B,N}(ker::Kernel{T,S,B},
                                x::AbstractArray{T,N},
                                A::Array{T,N},
                                len::Integer)
    return apply_adjoint!(Array{T}(len), ker, x, A; clr = true)
end


#------------------------------------------------------------------------------
# In-place direct operation.

function apply_direct!{T,B,N}(dst::Array{T,N},
                              ker::Kernel{T,1,B},
                              x::AbstractArray{T,N},
                              A::Vector{T})
    @assert size(dst) == size(x)
    lim = limits(ker, length(A))
    @inbounds for i in eachindex(dst, x)
        j1, w1 = getcoefs(ker, lim, x[i])
        dst[i] = w1*A[j1]
    end
    return dst
end

function apply_direct!{T,B,N}(dst::Array{T,N},
                             ker::Kernel{T,2,B},
                             x::AbstractArray{T,N},
                             A::Vector{T})
    @assert size(dst) == size(x)
    lim = limits(ker, length(A))
    @inbounds for i in eachindex(dst, x)
        j1, j2, w1, w2 = getcoefs(ker, lim, x[i])
        dst[i] = w1*A[j1] + w2*A[j2]
    end
    return dst
end

function apply_direct!{T,B,N}(dst::Array{T,N},
                             ker::Kernel{T,3,B},
                             x::AbstractArray{T,N},
                             A::Vector{T})
    @assert size(dst) == size(x)
    lim = limits(ker, length(A))
    @inbounds for i in eachindex(dst, x)
        j1, j2, j3, w1, w2, w3 = getcoefs(ker, lim, x[i])
        dst[i] = w1*A[j1] + w2*A[j2] + w3*A[j3]
    end
    return dst
end

function apply_direct!{T,B,N}(dst::Array{T,N},
                              ker::Kernel{T,4,B},
                              x::AbstractArray{T,N},
                              A::Vector{T})
    @assert size(dst) == size(x)
    lim = limits(ker, length(A))
    @inbounds for i in eachindex(dst, x)
        j1, j2, j3, j4, w1, w2, w3, w4 = getcoefs(ker, lim, x[i])
        dst[i] = w1*A[j1] + w2*A[j2] + w3*A[j3] + w4*A[j4]
    end
    return dst
end


#------------------------------------------------------------------------------
# In-place adjoint operation.

function apply_adjoint!{T,B,N}(dst::Vector{T},
                               ker::Kernel{T,1,B},
                               x::AbstractArray{T,N},
                               A::Array{T,N};
                               clr::Bool = true)
    @assert size(A) == size(x)
    if clr
        fill!(dst, zero(T))
    end
    lim = limits(ker, length(dst))
    @inbounds for i in eachindex(A, x)
        j1, w1 = getcoefs(ker, lim, x[i])
        dst[j1] += w1*A[i]
    end
    return dst
end

function apply_adjoint!{T,B,N}(dst::Vector{T},
                               ker::Kernel{T,2,B},
                               x::AbstractArray{T,N},
                               A::Array{T,N};
                               clr::Bool = true)
    @assert size(A) == size(x)
    if clr
        fill!(dst, zero(T))
    end
    lim = limits(ker, length(dst))
    @inbounds for i in eachindex(A, x)
        j1, j2, w1, w2 = getcoefs(ker, lim, x[i])
        a = A[i]
        dst[j1] += w1*a
        dst[j2] += w2*a
    end
    return dst
end

function apply_adjoint!{T,B,N}(dst::Vector{T},
                               ker::Kernel{T,3,B},
                               x::AbstractArray{T,N},
                               A::Array{T,N};
                               clr::Bool = true)
    @assert size(A) == size(x)
    if clr
        fill!(dst, zero(T))
    end
    lim = limits(ker, length(dst))
    @inbounds for i in eachindex(A, x)
        j1, j2, j3, w1, w2, w3 = getcoefs(ker, lim, x[i])
        a = A[i]
        dst[j1] += w1*a
        dst[j2] += w2*a
        dst[j3] += w3*a
    end
    return dst
end

function apply_adjoint!{T,B,N}(dst::Vector{T},
                               ker::Kernel{T,4,B},
                               x::AbstractArray{T,N},
                               A::Array{T,N};
                               clr::Bool = true)
    @assert size(A) == size(x)
    if clr
        fill!(dst, zero(T))
    end
    lim = limits(ker, length(dst))
    @inbounds for i in eachindex(A, x)
        j1, j2, j3, j4, w1, w2, w3, w4 = getcoefs(ker, lim, x[i])
        a = A[i]
        dst[j1] += w1*a
        dst[j2] += w2*a
        dst[j3] += w3*a
        dst[j4] += w4*a
    end
    return dst
end

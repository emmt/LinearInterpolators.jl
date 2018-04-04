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
# Out-of place versions (coordinates cannot be a function).

function apply(ker::Kernel{T,S,B},
               x::AbstractArray{T,N},
               src::AbstractVector{T}) where {T,S,B,N}
    return apply(Direct, ker, x, src)
end

function apply(::Type{Direct}, ker::Kernel{T,S,B},
               x::AbstractArray{T,N},
               src::AbstractVector{T}) where {T,S,B,N}
    return apply!(Array{T}(size(x)), Direct, ker, x, src)
end

function apply(::Type{Adjoint}, ker::Kernel{T,S,B},
               x::AbstractArray{T,N},
               src::AbstractArray{T,N}, len::Integer) where {T,S,B,N}
    return apply!(Array{T}(len), Adjoint, ker, x, src)
end

#------------------------------------------------------------------------------
# In-place wrappers.

function apply!(dst::AbstractArray{T,N},
                ker::Kernel{T,S,B},
                x::Union{Function,AbstractArray{T,N}},
                src::AbstractVector{T}) where {T,S,B,N}
    apply!(one(Scalar), Direct, ker, x, src, zero(Scalar), dst)
end

function apply!(dst::AbstractArray{T,N},
                ::Type{Direct},
                ker::Kernel{T,S,B},
                x::Union{Function,AbstractArray{T,N}},
                src::AbstractVector{T}) where {T,S,B,N}
    apply!(one(Scalar), Direct, ker, x, src, zero(Scalar), dst)
end

function apply!(dst::AbstractVector{T},
                ::Type{Adjoint},
                ker::Kernel{T,S,B},
                x::Union{Function,AbstractArray{T,N}},
                src::AbstractArray{T,N}) where {T,S,B,N}
    apply!(one(Scalar), Adjoint, ker, x, src, zero(Scalar), dst)
end

function apply!(α::Real,
                ::Type{Direct},
                ker::Kernel{T,S,B},
                x::Union{Function,AbstractArray{T,N}},
                src::AbstractVector{T},
                β::Real,
                dst::AbstractArray{T,N}) where {T,S,B,N}
    apply!(convert(Scalar, α), Direct, ker, x, src, convert(Scalar, β), dst)
end

function apply!(α::Real,
                ::Type{Adjoint},
                ker::Kernel{T,S,B},
                x::Union{Function,AbstractArray{T,N}},
                src::AbstractArray{T,N},
                β::Real,
                dst::AbstractVector{T}) where {T,S,B,N}
    apply!(convert(Scalar, α), Adjoint, ker, x, src, convert(Scalar, β), dst)
end

# Fallback method to avoid infinite loop.
function apply!(α::Scalar,
                ::Type{Direct},
                ker::Kernel{T,S,B},
                x::Union{Function,AbstractArray{T,N}},
                src::AbstractVector{T},
                β::Scalar,
                dst::AbstractArray{T,N}) where {T,S,B,N}
    error("unimplemented operation")
end

function apply!(α::Scalar,
                ::Type{Adjoint},
                ker::Kernel{T,S,B},
                x::Union{Function,AbstractArray{T,N}},
                src::AbstractArray{T,N},
                β::Scalar,
                dst::AbstractVector{T}) where {T,S,B,N}
    error("unimplemented operation")
end

#------------------------------------------------------------------------------
# In-place direct operation.

function apply!(α::Scalar,
                ::Type{Direct},
                ker::Kernel{T,1,B},
                x::AbstractArray{T,N},
                src::AbstractVector{T},
                β::Scalar,
                dst::AbstractArray{T,N}) where {T,B,N}
    @assert size(dst) == size(x)
    lim = limits(ker, length(src))
    if α == 0
        vscale!(dst, β)
    elseif β == 0
        if α == 1
            @inbounds for i in eachindex(dst, x)
                j1, w1 = getcoefs(ker, lim, x[i])
                dst[i] = w1*src[j1]
            end
        else
            alpha = convert(T, α)
            @inbounds for i in eachindex(dst, x)
                j1, w1 = getcoefs(ker, lim, x[i])
                dst[i] = alpha*w1*src[j1]
            end
        end
    elseif β == 1
        if α == 1
            @inbounds for i in eachindex(dst, x)
                j1, w1 = getcoefs(ker, lim, x[i])
                dst[i] += w1*src[j1]
            end
        else
            alpha = convert(T, α)
            @inbounds for i in eachindex(dst, x)
                j1, w1 = getcoefs(ker, lim, x[i])
                dst[i] += alpha*w1*src[j1]
            end
        end
    else
        alpha = convert(T, α)
        beta = convert(T, β)
        @inbounds for i in eachindex(dst, x)
            j1, w1 = getcoefs(ker, lim, x[i])
            dst[i] = alpha*w1*src[j1] + beta*dst[i]
        end
    end
    return dst
end

function apply!(α::Scalar,
                ::Type{Direct},
                ker::Kernel{T,2,B},
                x::AbstractArray{T,N},
                src::AbstractVector{T},
                β::Scalar,
                dst::AbstractArray{T,N}) where {T,B,N}
    @assert size(dst) == size(x)
    lim = limits(ker, length(src))
    if α == 0
        vscale!(dst, β)
    elseif α == 1
        if β == 0
            @inbounds for i in eachindex(dst, x)
                j1, j2, w1, w2 = getcoefs(ker, lim, x[i])
                dst[i] = w1*src[j1] + w2*src[j2]
            end
        else
            beta = convert(T, β)
            @inbounds for i in eachindex(dst, x)
                j1, j2, w1, w2 = getcoefs(ker, lim, x[i])
                dst[i] = w1*src[j1] + w2*src[j2] + beta*dst[i]
            end
        end
    else
        alpha = convert(T, α)
        if β == 0
            @inbounds for i in eachindex(dst, x)
                j1, j2, w1, w2 = getcoefs(ker, lim, x[i])
                dst[i] = (w1*src[j1] + w2*src[j2])*alpha
            end
        else
            beta = convert(T, β)
            @inbounds for i in eachindex(dst, x)
                j1, j2, w1, w2 = getcoefs(ker, lim, x[i])
                dst[i] = (w1*src[j1] + w2*src[j2])*alpha + beta*dst[i]
            end
        end
    end
    return dst
end

function apply!(α::Scalar,
                ::Type{Direct},
                ker::Kernel{T,3,B},
                x::AbstractArray{T,N},
                src::AbstractVector{T},
                β::Scalar,
                dst::AbstractArray{T,N}) where {T,B,N}
    @assert size(dst) == size(x)
    lim = limits(ker, length(src))
    if α == 0
        vscale!(dst, β)
    elseif β == 0
        if α == 1
            @inbounds for i in eachindex(dst, x)
                j1, j2, j3, w1, w2, w3 = getcoefs(ker, lim, x[i])
                dst[i] = w1*src[j1] + w2*src[j2] + w3*src[j3]
            end
        else
            alpha = convert(T, α)
            @inbounds for i in eachindex(dst, x)
                j1, j2, j3, w1, w2, w3 = getcoefs(ker, lim, x[i])
                dst[i] = (w1*src[j1] + w2*src[j2] + w3*src[j3])*alpha
            end
        end
    else
        alpha = convert(T, α)
        beta = convert(T, β)
        @inbounds for i in eachindex(dst, x)
            j1, j2, j3, w1, w2, w3 = getcoefs(ker, lim, x[i])
            dst[i] = (w1*src[j1] + w2*src[j2] + w3*src[j3])*alpha + beta*dst[i]
        end
    end
    return dst
end

function apply!(α::Scalar,
                ::Type{Direct},
                ker::Kernel{T,4,B},
                x::AbstractArray{T,N},
                src::AbstractVector{T},
                β::Scalar,
                dst::AbstractArray{T,N}) where {T,B,N}
    @assert size(dst) == size(x)
    lim = limits(ker, length(src))
    if α == 0
        vscale!(dst, β)
    elseif β == 0
        if α == 1
            @inbounds for i in eachindex(dst, x)
                j1, j2, j3, j4, w1, w2, w3, w4 = getcoefs(ker, lim, x[i])
                dst[i] = w1*src[j1] + w2*src[j2] + w3*src[j3] + w4*src[j4]
            end
        else
            alpha = convert(T, α)
            @inbounds for i in eachindex(dst, x)
                j1, j2, j3, j4, w1, w2, w3, w4 = getcoefs(ker, lim, x[i])
                dst[i] = (w1*src[j1] + w2*src[j2] +
                          w3*src[j3] + w4*src[j4])*alpha
            end

        end
    else
        alpha = convert(T, α)
        beta = convert(T, β)
        @inbounds for i in eachindex(dst, x)
            j1, j2, j3, j4, w1, w2, w3, w4 = getcoefs(ker, lim, x[i])
            dst[i] = (w1*src[j1] + w2*src[j2] +
                      w3*src[j3] + w4*src[j4])*alpha + beta*dst[i]
        end
    end
    return dst
end

#------------------------------------------------------------------------------
# In-place adjoint operation.

function apply!(α::Scalar,
                ::Type{Adjoint},
                ker::Kernel{T,1,B},
                x::AbstractArray{T,N},
                src::AbstractArray{T,N},
                β::Scalar,
                dst::AbstractVector{T}) where {T,B,N}
    @assert size(src) == size(x)
    vscale!(dst, β)
    lim = limits(ker, length(dst))
    if α == 1
        @inbounds for i in eachindex(src, x)
            j1, w1 = getcoefs(ker, lim, x[i])
            dst[j1] += w1*src[i]
        end
    elseif α != 0
        alpha = convert(T, α)
        @inbounds for i in eachindex(src, x)
            j1, w1 = getcoefs(ker, lim, x[i])
            dst[j1] += alpha*w1*src[i]
        end
    end
    return dst
end

function apply!(α::Scalar,
                ::Type{Adjoint},
                ker::Kernel{T,2,B},
                x::AbstractArray{T,N},
                src::AbstractArray{T,N},
                β::Scalar,
                dst::AbstractVector{T}) where {T,B,N}
    @assert size(src) == size(x)
    vscale!(dst, β)
    lim = limits(ker, length(dst))
    if α == 1
        @inbounds for i in eachindex(src, x)
            j1, j2, w1, w2 = getcoefs(ker, lim, x[i])
            a = src[i]
            dst[j1] += w1*a
            dst[j2] += w2*a
        end
    elseif α != 0
        alpha = convert(T, α)
        @inbounds for i in eachindex(src, x)
            j1, j2, w1, w2 = getcoefs(ker, lim, x[i])
            a = alpha*src[i]
            dst[j1] += w1*a
            dst[j2] += w2*a
        end
    end
    return dst
end

function apply!(α::Scalar,
                ::Type{Adjoint},
                ker::Kernel{T,3,B},
                x::AbstractArray{T,N},
                src::AbstractArray{T,N},
                β::Scalar,
                dst::AbstractVector{T}) where {T,B,N}
    @assert size(src) == size(x)
    vscale!(dst, β)
    lim = limits(ker, length(dst))
    if α == 1
        @inbounds for i in eachindex(src, x)
            j1, j2, j3, w1, w2, w3 = getcoefs(ker, lim, x[i])
            a = src[i]
            dst[j1] += w1*a
            dst[j2] += w2*a
            dst[j3] += w3*a
        end
    elseif α != 0
        alpha = convert(T, α)
        @inbounds for i in eachindex(src, x)
            j1, j2, j3, w1, w2, w3 = getcoefs(ker, lim, x[i])
            a = alpha*src[i]
            dst[j1] += w1*a
            dst[j2] += w2*a
            dst[j3] += w3*a
        end
    end
    return dst
end

function apply!(α::Scalar,
                ::Type{Adjoint},
                ker::Kernel{T,4,B},
                x::AbstractArray{T,N},
                src::AbstractArray{T,N},
                β::Scalar,
                dst::AbstractVector{T}) where {T,B,N}
    @assert size(src) == size(x)
    vscale!(dst, β)
    lim = limits(ker, length(dst))
    if α == 1
        @inbounds for i in eachindex(src, x)
            j1, j2, j3, j4, w1, w2, w3, w4 = getcoefs(ker, lim, x[i])
            a = src[i]
            dst[j1] += w1*a
            dst[j2] += w2*a
            dst[j3] += w3*a
            dst[j4] += w4*a
        end
    elseif α != 0
        alpha = convert(T, α)
        @inbounds for i in eachindex(src, x)
            j1, j2, j3, j4, w1, w2, w3, w4 = getcoefs(ker, lim, x[i])
            a = alpha*src[i]
            dst[j1] += w1*a
            dst[j2] += w2*a
            dst[j3] += w3*a
            dst[j4] += w4*a
        end
    end
    return dst
end

#------------------------------------------------------------------------------
# In-place direct operation at positions given by a function.

function apply!(α::Scalar,
                ::Type{Direct},
                ker::Kernel{T,1,B},
                f::Function,
                src::AbstractVector{T},
                β::Scalar,
                dst::AbstractArray{T,N}) where {T,B,N}
    lim = limits(ker, length(src))
    if α == 0
        vscale!(dst, β)
    elseif β == 0
        if α == 1
            @inbounds for i in eachindex(dst)
                x = f(i) :: T
                j1, w1 = getcoefs(ker, lim, x)
                dst[i] = w1*src[j1]
            end
        else
            alpha = convert(T, α)
            @inbounds for i in eachindex(dst)
                x = f(i) :: T
                j1, w1 = getcoefs(ker, lim, x)
                dst[i] = alpha*w1*src[j1]
            end
        end
    else
        alpha = convert(T, α)
        beta = convert(T, β)
        @inbounds for i in eachindex(dst)
            x = f(i) :: T
            j1, w1 = getcoefs(ker, lim, x)
            dst[i] = alpha*w1*src[j1] + beta*dst[i]
        end
    end
    return dst
end

function apply!(α::Scalar,
                ::Type{Direct},
                ker::Kernel{T,2,B},
                f::Function,
                src::AbstractVector{T},
                β::Scalar,
                dst::AbstractArray{T,N}) where {T,B,N}
    lim = limits(ker, length(src))
    if α == 0
        vscale!(dst, β)
    elseif β == 0
        if α == 1
            @inbounds for i in eachindex(dst)
                x = f(i) :: T
                j1, j2, w1, w2 = getcoefs(ker, lim, x)
                dst[i] = w1*src[j1] + w2*src[j2]
            end
        else
            alpha = convert(T, α)
            @inbounds for i in eachindex(dst)
                x = f(i) :: T
                j1, j2, w1, w2 = getcoefs(ker, lim, x)
                dst[i] = (w1*src[j1] + w2*src[j2])*alpha
            end
        end
    else
        alpha = convert(T, α)
        beta = convert(T, β)
        @inbounds for i in eachindex(dst)
            x = f(i) :: T
            j1, j2, w1, w2 = getcoefs(ker, lim, x)
            dst[i] = (w1*src[j1] + w2*src[j2])*alpha + beta*dst[i]
        end
    end
    return dst
end

function apply!(α::Scalar,
                ::Type{Direct},
                ker::Kernel{T,3,B},
                f::Function,
                src::AbstractVector{T},
                β::Scalar,
                dst::AbstractArray{T,N}) where {T,B,N}
    lim = limits(ker, length(src))
    if α == 0
        vscale!(dst, β)
    elseif β == 0
        if α == 1
            @inbounds for i in eachindex(dst)
                x = f(i) :: T
                j1, j2, j3, w1, w2, w3 = getcoefs(ker, lim, x)
                dst[i] = w1*src[j1] + w2*src[j2] + w3*src[j3]
            end
        else
            alpha = convert(T, α)
            @inbounds for i in eachindex(dst)
                x = f(i) :: T
                j1, j2, j3, w1, w2, w3 = getcoefs(ker, lim, x)
                dst[i] = (w1*src[j1] + w2*src[j2] + w3*src[j3])*alpha
            end
        end
    else
        alpha = convert(T, α)
        beta = convert(T, β)
        @inbounds for i in eachindex(dst)
            x = f(i) :: T
            j1, j2, j3, w1, w2, w3 = getcoefs(ker, lim, x)
            dst[i] = (w1*src[j1] + w2*src[j2] + w3*src[j3])*alpha + beta*dst[i]
        end
    end
    return dst
end

function apply!(α::Scalar,
                ::Type{Direct},
                ker::Kernel{T,4,B},
                f::Function,
                src::AbstractVector{T},
                β::Scalar,
                dst::AbstractArray{T,N}) where {T,B,N}
    lim = limits(ker, length(src))
    if α == 0
        vscale!(dst, β)
    elseif β == 0
        if α == 1
            @inbounds for i in eachindex(dst)
                x = f(i) :: T
                j1, j2, j3, j4, w1, w2, w3, w4 = getcoefs(ker, lim, x)
                dst[i] = w1*src[j1] + w2*src[j2] + w3*src[j3] + w4*src[j4]
            end
        else
            alpha = convert(T, α)
            @inbounds for i in eachindex(dst)
                x = f(i) :: T
                j1, j2, j3, j4, w1, w2, w3, w4 = getcoefs(ker, lim, x)
                dst[i] = (w1*src[j1] + w2*src[j2] +
                          w3*src[j3] + w4*src[j4])*alpha
            end
        end
    else
        alpha = convert(T, α)
        beta = convert(T, β)
        @inbounds for i in eachindex(dst)
            x = f(i) :: T
            j1, j2, j3, j4, w1, w2, w3, w4 = getcoefs(ker, lim, x)
            dst[i] = (w1*src[j1] + w2*src[j2] +
                      w3*src[j3] + w4*src[j4])*alpha + beta*dst[i]
        end
    end
    return dst
end

#------------------------------------------------------------------------------
# In-place adjoint operation at positions given by a function.

function apply!(α::Scalar,
                ::Type{Adjoint},
                ker::Kernel{T,1,B},
                f::Function,
                src::AbstractArray{T,N},
                β::Scalar,
                dst::AbstractVector{T}) where {T,B,N}
    vscale!(dst, β)
    lim = limits(ker, length(dst))
    if α == 1
        @inbounds for i in eachindex(src)
            x = f(i) :: T
            j1, w1 = getcoefs(ker, lim, x)
            dst[j1] += w1*src[i]
        end
    elseif α != 0
        alpha = convert(T, α)
        @inbounds for i in eachindex(src)
            x = f(i) :: T
            j1, w1 = getcoefs(ker, lim, x)
            dst[j1] += alpha*w1*src[i]
        end
    end
    return dst
end

function apply!(α::Scalar,
                ::Type{Adjoint},
                ker::Kernel{T,2,B},
                f::Function,
                src::AbstractArray{T,N},
                β::Scalar,
                dst::AbstractVector{T}) where {T,B,N}
    vscale!(dst, β)
    lim = limits(ker, length(dst))
    if α == 1
        @inbounds for i in eachindex(src)
            x = f(i) :: T
            j1, j2, w1, w2 = getcoefs(ker, lim, x)
            a = src[i]
            dst[j1] += w1*a
            dst[j2] += w2*a
        end
    elseif α != 0
        alpha = convert(T, α)
        @inbounds for i in eachindex(src)
            x = f(i) :: T
            j1, j2, w1, w2 = getcoefs(ker, lim, x)
            a = alpha*src[i]
            dst[j1] += w1*a
            dst[j2] += w2*a
        end
    end
    return dst
end

function apply!(α::Scalar,
                ::Type{Adjoint},
                ker::Kernel{T,3,B},
                f::Function,
                src::AbstractArray{T,N},
                β::Scalar,
                dst::AbstractVector{T}) where {T,B,N}
    vscale!(dst, β)
    lim = limits(ker, length(dst))
    if α == 1
        @inbounds for i in eachindex(src)
            x = f(i) :: T
            j1, j2, j3, w1, w2, w3 = getcoefs(ker, lim, x)
            a = src[i]
            dst[j1] += w1*a
            dst[j2] += w2*a
            dst[j3] += w3*a
        end
    elseif α != 0
        alpha = convert(T, α)
        @inbounds for i in eachindex(src)
            x = f(i) :: T
            j1, j2, j3, w1, w2, w3 = getcoefs(ker, lim, x)
            a = alpha*src[i]
            dst[j1] += w1*a
            dst[j2] += w2*a
            dst[j3] += w3*a
        end
    end
    return dst
end

function apply!(α::Scalar,
                ::Type{Adjoint},
                ker::Kernel{T,4,B},
                f::Function,
                src::AbstractArray{T,N},
                β::Scalar,
                dst::AbstractVector{T}) where {T,B,N}
    vscale!(dst, β)
    lim = limits(ker, length(dst))
    if α == 1
        @inbounds for i in eachindex(src)
            x = f(i) :: T
            j1, j2, j3, j4, w1, w2, w3, w4 = getcoefs(ker, lim, x)
            a = src[i]
            dst[j1] += w1*a
            dst[j2] += w2*a
            dst[j3] += w3*a
            dst[j4] += w4*a
        end
    elseif α != 0
        alpha = convert(T, α)
        @inbounds for i in eachindex(src)
            x = f(i) :: T
            j1, j2, j3, j4, w1, w2, w3, w4 = getcoefs(ker, lim, x)
            a = alpha*src[i]
            dst[j1] += w1*a
            dst[j2] += w2*a
            dst[j3] += w3*a
            dst[j4] += w4*a
        end
    end
    return dst
end

#
# unidimensional.jl --
#
# Unidimensional interpolation (the result may however be multi-dimensional).
#
#------------------------------------------------------------------------------
#
# This file is part of the LinearInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2016-2021, Éric Thiébaut.
#

# All code is in a module to "hide" private methods.
module UnidimensionalInterpolators

using InterpolationKernels

using LazyAlgebra
using LazyAlgebra.Foundations
import LazyAlgebra: apply, apply!, vcreate

using ..LinearInterpolators
using ..LinearInterpolators: limits, getcoefs
import ..LinearInterpolators.Meta

#------------------------------------------------------------------------------
# Out-of place versions (coordinates cannot be a function).

# FIXME: This is type-piracy because the only non-standard type, `Kernel`, comes
#        from another package.

function apply(ker::Kernel,
               x::AbstractArray,
               src::AbstractVector)
    return apply(Direct, ker, x, src)
end

function apply(::Type{Direct}, ker::Kernel,
               x::AbstractArray,
               src::AbstractVector)
    # The element type of the result is the type T of the product of the
    # interpolation weights by the elements of the source array.
    T = promote_type(eltype(ker), eltype(src))
    return apply!(Array{T}(undef, size(x)), Direct, ker, x, src)
end

function apply(::Type{Adjoint}, ker::Kernel,
               x::AbstractArray,
               src::AbstractArray, len::Integer)
    # The element type of the result is the type T of the product of the
    # interpolation weights by the elements of the source array.
    T = promote_type(eltype(ker), eltype(src))
    return apply!(Array{T}(undef, len), Adjoint, ker, x, src)
end

#------------------------------------------------------------------------------
# In-place wrappers.

function apply!(dst::AbstractArray{<:Any,N},
                ker::Kernel,
                x::Union{Function,AbstractArray{<:Any,N}},
                src::AbstractVector) where {N}
    apply!(1, Direct, ker, x, src, 0, dst)
end

function apply!(dst::AbstractArray{<:Any,N},
                ::Type{Direct},
                ker::Kernel,
                x::Union{Function,AbstractArray{<:Any,N}},
                src::AbstractVector) where {N}
    apply!(1, Direct, ker, x, src, 0, dst)
end

function apply!(dst::AbstractVector,
                ::Type{Adjoint},
                ker::Kernel,
                x::Union{Function,AbstractArray{<:Any,N}},
                src::AbstractArray{<:Any,N}) where {N}
    apply!(1, Adjoint, ker, x, src, 0, dst)
end

#------------------------------------------------------------------------------
# In-place direct operation.

function __generate_interp(n::Integer, ker::Symbol, lim::Meta.Arg,
                           pos::Meta.Arg, arr::Symbol)
    J = [Symbol(:j_,s) for s in 1:n]
    W = [Symbol(:w_,s) for s in 1:n]
    code = Meta.generate_getcoefs(J, W, ker, lim, pos)
    expr = Meta.generate_interp_expr(arr, J, W)
    return (code, expr)
end

@generated function apply!(α::Number,
                           ::Type{Direct},
                           ker::Kernel{<:Any,S},
                           x::AbstractArray{<:Any,N},
                           src::AbstractVector,
                           β::Number,
                           dst::AbstractArray{<:Any,N}) where {S,N}
    code, expr = __generate_interp(S, :ker, :lim, :pos, :src)
    quote
        @assert size(dst) == size(x)
        lim = limits(ker, length(src))
        if α == 0
            vscale!(dst, β)
        elseif β == 0
            if α == 1
                @inbounds for i in eachindex(dst, x)
                    pos = x[i]
                    $code
                    dst[i] = $expr
                end
            else
                alpha = promote_multiplier(α, eltype(ker), eltype(src))
                @inbounds for i in eachindex(dst, x)
                    pos = x[i]
                    $code
                    dst[i] = $expr*alpha
                end

            end
        else
            alpha = promote_multiplier(α, eltype(ker), eltype(src))
            beta  = promote_multiplier(β, eltype(dst))
            @inbounds for i in eachindex(dst, x)
                pos = x[i]
                $code
                dst[i] = $expr*alpha + beta*dst[i]
            end
        end
        return dst
    end
end

@generated function apply!(α::Number,
                           ::Type{Direct},
                           ker::Kernel{<:Any,S},
                           f::Function,
                           src::AbstractVector,
                           β::Number,
                           dst::AbstractArray{<:Any,N}) where {S,N}
    code, expr = __generate_interp(S, :ker, :lim, :pos, :src)
    quote
        lim = limits(ker, length(src))
        if α == 0
            vscale!(dst, β)
        elseif β == 0
            if α == 1
                @inbounds for i in eachindex(dst)
                    pos = f(i)
                    $code
                    dst[i] = $expr
                end
            else
                alpha = promote_multiplier(α, eltype(ker), eltype(src))
                @inbounds for i in eachindex(dst)
                    pos = f(i)
                    $code
                    dst[i] = $expr*alpha
                end
            end
        else
            alpha = promote_multiplier(α, eltype(ker), eltype(src))
            beta  = promote_multiplier(β, eltype(dst))
            @inbounds for i in eachindex(dst)
                pos = f(i)
                $code
                dst[i] = $expr*alpha + beta*dst[i]
            end
        end
        return dst
    end
end

#------------------------------------------------------------------------------
# In-place adjoint operation.

function __generate_interp_adj(n::Integer, ker::Symbol, lim::Meta.Arg,
                               pos::Meta.Arg, dst::Symbol, val::Symbol)
    J = [Symbol(:j_,s) for s in 1:n]
    W = [Symbol(:w_,s) for s in 1:n]
    return (Meta.generate_getcoefs(J, W, ker, lim, pos),
            [:($dst[$(J[i])] += $(W[i])*$val) for i in 1:n]...)
end

@generated function apply!(α::Number,
                           ::Type{Adjoint},
                           ker::Kernel{<:Any,S},
                           x::AbstractArray{<:Any,N},
                           src::AbstractArray{<:Any,N},
                           β::Number,
                           dst::AbstractVector) where {S,N}
    code = __generate_interp_adj(S, :ker, :lim, :pos, :dst, :val)
    quote
        @assert size(src) == size(x)
        vscale!(dst, β)
        lim = limits(ker, length(dst))
        if α == 1
            @inbounds for i in eachindex(src, x)
                pos = x[i]
                val = src[i]
                $(code...)
        end
        elseif α != 0
            alpha = promote_multiplier(α, eltype(ker), eltype(src))
            @inbounds for i in eachindex(src, x)
                pos = x[i]
                val = alpha*src[i]
                $(code...)
            end
        end
        return dst
    end
end

@generated function apply!(α::Number,
                           ::Type{Adjoint},
                           ker::Kernel{<:Any,S},
                           f::Function,
                           src::AbstractArray{<:Any,N},
                           β::Number,
                           dst::AbstractVector) where {S,N}
    code = __generate_interp_adj(S, :ker, :lim, :pos, :dst, :val)
    quote
        vscale!(dst, β)
        lim = limits(ker, length(dst))
        if α == 1
            @inbounds for i in eachindex(src)
                pos = f(i)
                val = src[i]
                $(code...)
            end
        elseif α != 0
            alpha = promote_multiplier(α, eltype(ker), eltype(src))
            @inbounds for i in eachindex(src)
                pos = f(i)
                val = alpha*src[i]
                $(code...)
            end
        end
        return dst
    end
end

end # module

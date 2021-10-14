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

# FIXME: this is type-piracy

function apply(ker::Kernel,
               x::AbstractArray,
               src::AbstractVector)
    return apply(Direct, ker, x, src)
end

function apply(::Type{Direct}, ker::Kernel{Tk,S},
               x::AbstractArray{Tx},
               src::AbstractVector{Ts}) where {Tk,Tx,Ts,S}
    Td = promote_type(Tk, Tx, Ts)
    return apply!(Array{Td}(undef, size(x)), Direct, ker, x, src)
end

function apply(::Type{Adjoint}, ker::Kernel{Tk,S},
               x::AbstractArray{Tx,N},
               src::AbstractArray{Ts,N}, len::Integer) where {Tk,Tx,Ts,S,N}
    Td = promote_type(Tk, Tx, Ts)
    return apply!(Array{Td}(undef, len), Adjoint, ker, x, src)
end

#------------------------------------------------------------------------------
# In-place wrappers.

function apply!(dst::AbstractArray{Td,N},
                ker::Kernel,
                x::Union{Function,AbstractArray{Tx,N}},
                src::AbstractVector) where {Td,Tx,N}
    apply!(1, Direct, ker, x, src, 0, dst)
end

function apply!(dst::AbstractArray{Td,N},
                ::Type{Direct},
                ker::Kernel,
                x::Union{Function,AbstractArray{Tx,N}},
                src::AbstractVector) where {Td,Tx,N}
    apply!(1, Direct, ker, x, src, 0, dst)
end

function apply!(dst::AbstractVector,
                ::Type{Adjoint},
                ker::Kernel,
                x::Union{Function,AbstractArray{Tx,N}},
                src::AbstractArray{Ts,N}) where {Ts,Tx,N}
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

@generated function apply!(α::Real,
                           ::Type{Direct},
                           ker::Kernel{Tk,S},
                           x::AbstractArray{Tx,N},
                           src::AbstractVector{Ts},
                           β::Real,
                           dst::AbstractArray{Td,N}) where {Tk,Tx,Ts,Td,S,N}
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
                alpha = convert(Td, α)
                @inbounds for i in eachindex(dst, x)
                    pos = x[i]
                    $code
                    dst[i] = $expr*alpha
                end

            end
        else
            alpha = convert(Td, α)
            beta = convert(Td, β)
            @inbounds for i in eachindex(dst, x)
                pos = x[i]
                $code
                dst[i] = $expr*alpha + beta*dst[i]
            end
        end
        return dst
    end
end

@generated function apply!(α::Real,
                           ::Type{Direct},
                           ker::Kernel{Tk,S},
                           f::Function,
                           src::AbstractVector{Ts},
                           β::Real,
                           dst::AbstractArray{Td,N}) where {Tk,Ts,Td,S,N}
    code, expr = __generate_interp(S, :ker, :lim, :pos, :src)
    quote
        lim = limits(ker, length(src))
        if α == 0
            vscale!(dst, β)
        elseif β == 0
            if α == 1
                @inbounds for i in eachindex(dst)
                    pos = f(i) :: Td
                    $code
                    dst[i] = $expr
                end
            else
                alpha = convert(T, α)
                @inbounds for i in eachindex(dst)
                    x = f(i) :: Td
                    $code
                    dst[i] = $expr*alpha
                end
            end
        else
            alpha = convert(Td, α)
            beta = convert(Td, β)
            @inbounds for i in eachindex(dst)
                x = f(i) :: Td
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

@generated function apply!(α::Real,
                           ::Type{Adjoint},
                           ker::Kernel{Tk,S},
                           x::AbstractArray{Tx,N},
                           src::AbstractArray{Ts,N},
                           β::Real,
                           dst::AbstractVector{Td}) where {Tk,Tx,Ts,Td,S,N}
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
            alpha = convert(Td, α)
            @inbounds for i in eachindex(src, x)
                pos = x[i]
                val = alpha*src[i]
                $(code...)
            end
        end
        return dst
    end
end

@generated function apply!(α::Real,
                           ::Type{Adjoint},
                           ker::Kernel{Tk,S},
                           f::Function,
                           src::AbstractArray{Ts,N},
                           β::Real,
                           dst::AbstractVector{Td}) where {Tk,Ts,Td,S,N}
    code = __generate_interp_adj(S, :ker, :lim, :pos, :dst, :val)
    quote
        vscale!(dst, β)
        lim = limits(ker, length(dst))
        if α == 1
            @inbounds for i in eachindex(src)
                pos = f(i) :: Td
                val = src[i]
                $(code...)
            end
        elseif α != 0
            alpha = convert(Td, α)
            @inbounds for i in eachindex(src)
                pos = f(i) :: Td
                val = alpha*src[i]
                $(code...)
            end
        end
        return dst
    end
end

end # module

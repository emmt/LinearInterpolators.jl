#
# interp/meta.jl --
#
# Functions to generate code for interpolation methods.
#
#------------------------------------------------------------------------------
#
# This file is part of the LinearInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2016-2018, Éric Thiébaut.
#

module Meta

const Arg = Union{Number,Symbol,Expr}

#"""
#```julia
#@getcoefs N j w ker lim pos
#```
#
#yields the code for extracting the interpolation coefficients, `N` is the
#interpolation kernel size, `j` and `w` are prefixes for the variables to store
#the indices and weights for interpolation (the actual variable names are given
#by appending a `_i` suffix for `i` in `1:N`), `ker` is the interpolation
#kernel, `lim` are the limits and `pos` is the position of interpolation.
#
#"""
#macro getcoefs(N, j, w, ker, lim, pos)
#    generate_getcoefs(N, j, w, ker, lim, pos)
#end
#
#macro nonsep_interp(alpha, src, beta, dst,
#                    s1, ker1, lim1, pos1,
#                    s2, ker2, lim2, pos2)
#    return generate_nonsep_interp(alpha, src, beta, dst,
#                          s1, ker1, lim1, pos1,
#                          s2, ker2, lim2, pos2)
#end
#
#macro nonsep_interp_adj(src, dst,
#                        s1, ker1, lim1, pos1,
#                        s2, ker2, lim2, pos2)
#    return generate_nonsep_interp_adj(src, dst,
#                              s1, ker1, lim1, pos1,
#                              s2, ker2, lim2, pos2)
#end

"""
```julia
generate_getcoefs(n, j, w, ker, lim, pos)
```

generates the code for getting the interpolation coefficients.  Here `n` is the
size of `ker` the interpolation kernel, `j` and `w` are symbols or strings used
as prefixes for local variables to store the interpolation indices and weights
respectively, `ker` and `lim` are symbols with the name of the local variables
which store the interpolation kernel and the limits along the dimension of
interpolation, `pos` is a symbol or an expression which gives the position to
interpolate.  For instance:

```julia
generate_getcoefs(2, :i, :c, :kr, :lm, :(x[i]))
-> :((i_1, i_2, c_1, c_2) = getcoefs(kr, lm, x[i]))
```

Another possibility is:

```julia
generate_getcoefs(J, W, ker, lim, pos)
```

where `J` and `W` are vectors of length `n`, the size of `ker` the
interpolation kernel, with the symbolic names of the local variables to store
the interpolation indices and weights respectively.

"""
function generate_getcoefs(n::Integer,
                           j::Union{AbstractString,Symbol},
                           w::Union{AbstractString,Symbol},
                           ker::Symbol, lim::Symbol, pos::Arg)
    return generate_getcoefs(make_varlist(j, n),
                             make_varlist(w, n), ker, lim, pos)
end

function generate_getcoefs(J::AbstractVector{Symbol},
                           W::AbstractVector{Symbol},
                           ker::Symbol, lim::Symbol, pos::Arg)
    @assert length(J) == length(W)
    vars =  Expr(:tuple, J..., W...)
    return :($vars = getcoefs($ker, $lim, $pos))
end

"""
```julia
make_varlist(pfx, n)
```

generates a list of symbols to be used as variable names.  Here `pfx` is a
symbol or a string used as a prefix and `n` is the number of variables or a
range of indices.  The generated names vahe the form "pfx_i" with `i` an
integer index.

"""
make_varlist(pfx::Union{AbstractString,Symbol}, n::Integer) =
    make_varlist(pfx, 1:n)
make_varlist(pfx::Union{AbstractString,Symbol}, I::UnitRange{<:Integer}) =
    [Symbol(pfx,:_,i) for i in I]


"""
```julia
generate_sum(ex)
```

generates an expression whose result is the sum of the elements of the vector
`ex` which are symbols (being interpreted as the name of variables) or
expressions.

"""
generate_sum(args::Arg...) = generate_sum(args)
generate_sum(ex::Union{AbstractVector,Tuple{Vararg{Arg}}}) =
    (length(ex) == 1 ? ex[1] : Expr(:call, :+, ex...))


"""
```julia
group_expressions(ex...) -> code
```

generates a single expression from all expressions given in argument.  The
result may be the same as the input if it is a single expression or a block of
expressions if several expressions are specified.

To insert the result as a block of statements (like a `begin` ... `end` block)
in a quoted expression, write something like:

```julia
quote
    some_pre_code
    \$code
    some_post_code
end
```

To strip the surrounding `begin` ... `end` keywords, write instead:

```julia
quote
    some_pre_code
    \$(code.args...)
    some_post_code
end
```

"""
group_expressions(args::Expr...) = group_expressions(args)
group_expressions(ex::Union{AbstractVector{Expr},Tuple{Vararg{Expr}}}) =
    (length(ex) == 1 ? ex[1] : Expr(:block, ex...))


function generate_interp_expr(arr::Symbol,
                              J::AbstractVector{Symbol},
                              W::AbstractVector{Symbol})
    @assert length(J) == length(W)
    return generate_sum([:($arr[$(J[i])]*$(W[i])) for i in 1:length(J)])
end

function generate_interp_expr(arr::Symbol,
                              J::AbstractVector{Symbol},
                              W::AbstractVector{Symbol},
                              inds::Arg)
    @assert length(J) == length(W)
    return generate_sum([:($arr[$(J[i]),$inds]*$(W[i])) for i in 1:length(J)])
end

function generate_interp_expr(arr::Symbol,
                              J1::AbstractVector{Symbol},
                              W1::AbstractVector{Symbol},
                              J2::AbstractVector{Symbol},
                              W2::AbstractVector{Symbol})
    @assert length(J2) == length(W2)
    n = length(J2)
    ex = Array{Expr}(n)
    for i in 1:n
        sub = generate_interp_expr(arr, J1, W1, J2[i])
        ex[i] = :($sub*$(W2[i]))
    end
    return generate_sum(ex)
end

end # module

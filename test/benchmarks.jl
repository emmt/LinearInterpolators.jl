module BenchmarkingLinearInterpolators

using Printf
using LinearInterpolators
using LazyAlgebra
using BenchmarkTools

mimeshow(io::IO, x) = show(io, MIME"text/plain"(), x)
mimeshow(x) = mimeshow(stdout, x)

prt(op::AbstractString, t) = begin
    @printf("%s:\n", op)
    mimeshow(t)
end

prt(op::AbstractString, t, nops::Integer) = begin
    # times in nanoseconds, speed in Gigaflops
    @printf("%s [%.6f Gflops]:\n", op, nops/minimum(t.times))
    mimeshow(t)
    println()
end

n1, n2 = 300, 200
m1, m2 =  43,  29
y = randn(n1, n2)
ker = CatmullRomSpline()
A1 = SparseUnidimensionalInterpolator(ker, 1, 1:n1, range(1,n1,length=m1))
A2 = SparseUnidimensionalInterpolator(ker, 2, 1:n2, range(1,n2,length=m2))
H = A1*A2

function LazyAlgebra.vcreate(::Type{LazyAlgebra.Direct},
                             ::Gram,
                             x::AbstractArray{T,N},
                             scratch::Bool) where {T<:AbstractFloat,N}
    return similar(x)
end

r1 = 2*length(A1.C)
r2 = 2*length(A2.C)
x = conjgrad(H'*H, H'*y; verb=true, ftol=1e-8, maxiter=50)
z = vcopy(y)
prt("A1*x",      @benchmark($A1*$x),        r1*m2)
prt("A2*x",      @benchmark($A2*$x),        r2*m1)
prt("A2*A1*x",   @benchmark($(A2*A1)*$x),   r1*m2 + r2*n1)
prt("A1*A2*x",   @benchmark($(A1*A2)*$x),   r2*m1 + r1*n2)
prt("A1'*y",     @benchmark($(A1')*$y),     r1*m2)
prt("A2'*y",     @benchmark($(A2')*$y),     r2*m1)
prt("A2'*A1'*y", @benchmark($(A2'*A1')*$y), r1*m2 + r2*n1)
prt("A1'*A2'*y", @benchmark($(A1'*A2')*$y), r2*m1 + r1*n2)
println("")

n = 100
pos = [1.1*i + 0.2*j for i in 1:n1, j in 1:n2]
pmin, pmax = extrema(pos)
A = SparseInterpolator(ker, pos, range(pmin, pmax, length=n))
x = randn(n)
y = randn(n1,n2)
nops = 2*length(A.C)

prt("A*x",  @benchmark($A*$x), nops)
prt("A'*x", @benchmark($(A')*$y), nops)

end # module

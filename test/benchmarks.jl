module BenchmarkingLinearInterpolators

using Printf
using LinearInterpolators
using LazyAlgebra
using BenchmarkTools

mimeshow(io::IO, x) = show(io, MIME"text/plain"(), x)
mimeshow(x) = mimeshow(stdout, x)

n1, n2 = 300, 200
m1, m2 =  43,  29
y = randn(n1, n2)
ker = CatmullRomSpline()
A1 = SparseUnidimensionalInterpolator(ker, 1, 1:n1, range(1,n1,length=m1))
A2 = SparseUnidimensionalInterpolator(ker, 2, 1:n2, range(1,n2,length=m2))
H = A1*A2

if false
    x = conjgrad(H'*H, H'*y; verb=true, ftol=1e-8, maxiter=50)
    z = vcopy(y)
    println("\n\nA1*x:")
    mimeshow(@benchmark(A1*x))
    println("\n\nA2*x:")
    mimeshow(@benchmark(A2*x))
    println("\n\nA2*A1*x:")
    mimeshow(@benchmark(A2*A1*x))
    println("\n\nA1*A2*x:")
    mimeshow(@benchmark(A1*A2*x))
    println("\n\n\nA1'*y:")
    mimeshow(@benchmark(A1'*y))
    println("\n\nA2'*y:")
    mimeshow(@benchmark(A2'*y))
    println("\n\nA2'*A1'*y:")
    mimeshow(@benchmark(A2'*A1'*y))
    println("\n\nA1'*A2'*y:")
    mimeshow(@benchmark(A1'*A2'*y))
    println("")
end

units = "Gflops" # times in nanoseconds, speed in Gigaflops
n = 100
pos = [1.1*i + 0.2*j for i in 1:n1, j in 1:n2]
pmin, pmax = extrema(pos)
A = SparseInterpolator(ker, pos, range(pmin, pmax, length=n))
x = randn(n)
y = randn(n1,n2)
nops = 2*length(A.C)

t = @benchmark(A*x)
@printf("\nA*x [%.6f %s]:\n", nops/median(t.times), units)
mimeshow(t)
println("")

t = @benchmark(A'*y)
@printf("\nA'*y [%.6f %s]:\n", nops/median(t.times), units)
mimeshow(t)
println("")

end # module

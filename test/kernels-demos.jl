isdefined(:LinearInterpolators) || include("../src/LinearInterpolators.jl")

module LinearInterpolatorsKernelsDemos

using LinearInterpolators.Kernels

using PyCall
# pygui(:gtk); # can be: :wx, :gtk, :qt
using LaTeXStrings
import PyPlot
const plt = PyPlot

box = RectangularSpline()
triangle = LinearSpline()
quadratic = QuadraticSpline()
cubic = CubicSpline()
catmull_rom = CatmullRomSpline()
mitchell_netravali = MitchellNetravaliSpline()
keys = KeysSpline(0.5)
lanczos2 = LanczosKernel(2)
lanczos4 = LanczosKernel(4)
lanczos6 = LanczosKernel(6)
lanczos8 = LanczosKernel(8)
lanczos10 = LanczosKernel(10)

plt.figure(10)
plt.clf()
x = linspace(-6,7,1000)
plt.plot(x, box(x), color="darkgreen",
         linewidth=2.0, linestyle="-", label="box")
plt.plot(x, triangle(x), color="darkblue",
         linewidth=2.0, linestyle="-", label="triangle (linear B-spline)")
plt.plot(x, quadratic(x), color="darkcyan",
         linewidth=2.0, linestyle="-", label="quadratic B-spline")
plt.plot(x, cubic(x), color="firebrick",
         linewidth=2.0, linestyle="-", label="cubic B-spline")
plt.plot(x, catmull_rom(x), color="orange",
         linewidth=2.0, linestyle="-", label="Catmull-Rom spline")
plt.plot(x, mitchell_netravali(x), color="violet",
         linewidth=2.0, linestyle="-", label="Mitchell & Netravali spline")
plt.plot(x, lanczos8(x), color="black",
         linewidth=2.0, linestyle="-", label="Lanczos 8")
plt.plot(x, lanczos10(x), color="gray",
         linewidth=2.0, linestyle="-", label="Lanczos 10")
plt.title("Some kernel functions")
plt.legend()

plt.figure(11)
plt.clf()
ker1 = CardinalCubicSpline(-1/2)
ker2 = CardinalCubicSpline(0)
x = linspace(-3,4,141)
plt.plot(x, ker1(x), color="darkgreen",
         linewidth=2.0, linestyle="-", label="f(x; c = -1/2)")
plt.plot(x, ker1'(x), color="darkblue",
         linewidth=2.0, linestyle="-", label="f'(x; c = -1/2)")
plt.plot(x, ker2(x), color="darkcyan",
         linewidth=2.0, linestyle="-", label="f(x; c = 0)")
plt.plot(x, ker2'(x), color="firebrick",
         linewidth=2.0, linestyle="-", label="f'(x; c = 0)")
plt.title("Cardinal cubic splines")
plt.legend()

plt.figure(12)
plt.clf()
ker = LinearSpline()
x = linspace(-3,4,141)
plt.plot(x, ker(x), color="darkgreen",
         linewidth=2.0, linestyle="-", label="f(x)")
plt.plot(x, ker'(x), color="darkblue",
         linewidth=2.0, linestyle="-", label="f'(x)")
plt.title("Linear B-spline")
plt.legend()

plt.figure(13)
plt.clf()
ker = QuadraticSpline()
x = linspace(-3,4,141)
plt.plot(x, ker(x), color="darkgreen",
         linewidth=2.0, linestyle="-", label="f(x)")
plt.plot(x, ker'(x), color="darkblue",
         linewidth=2.0, linestyle="-", label="f'(x)")
plt.title("Quadratic B-spline")
plt.legend()

plt.figure(14)
plt.clf()
ker = CubicSpline()
x = linspace(-3,4,141)
plt.plot(x, ker(x), color="darkgreen",
         linewidth=2.0, linestyle="-", label="f(x)")
plt.plot(x, ker'(x), color="darkblue",
         linewidth=2.0, linestyle="-", label="f'(x)")
plt.title("Cubic B-spline")
plt.legend()

end # module

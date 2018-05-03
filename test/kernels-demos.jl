isdefined(:LinearInterpolators) || include("../src/LinearInterpolators.jl")

module LinearInterpolatorsKernelsDemos

import LinearInterpolators.Kernels

using PyCall
# pygui(:gtk); # can be: :wx, :gtk, :qt
using LaTeXStrings
import PyPlot
const plt = PyPlot

box = Kernels.RectangularSpline()
triangle = Kernels.LinearSpline()
quadratic = Kernels.QuadraticSpline()
cubic = Kernels.CubicSpline()
catmull_rom = Kernels.CatmullRomSpline()
mitchell_netravali = Kernels.MitchellNetravaliSpline()
keys = Kernels.KeysSpline(0.5)
lanczos2 = Kernels.LanczosKernel(2)
lanczos4 = Kernels.LanczosKernel(4)
lanczos6 = Kernels.LanczosKernel(6)
lanczos8 = Kernels.LanczosKernel(8)
lanczos10 = Kernels.LanczosKernel(10)

plt.figure(1)
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

end # module

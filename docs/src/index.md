# Introduction

The `LinearInterpolators` package provides many linear interpolation methods
for [Julia](https://julialang.org/). These interpolations are *linear* in the
sense that the result depends linearly on the input.

The source code is on [GitHub](https://github.com/emmt/LinearInterpolators.jl).


## Features

* Separable interpolations are supported for arrays of any dimensionality.
  Interpolation kernels can be different along each interpolated dimension.

* For 2D arrays, interpolations may be separable or not (*e.g.* to apply an
  image rotation).

* Undimensional interpolations may be used to produce multi-dimensional
  results.

* Many interpolation kernels are provided by the package
  [`InterpolationKernels`](https://github.com/emmt/InterpolationKernels.jl)
  (B-splines of degree 0 to 3, cardinal cubic splines, Catmull-Rom spline,
  Mitchell & Netravali spline, Lanczos resampling kernels of arbitrary size,
  *etc.*).

* **Interpolators** are linear maps such as the ones defined by the
  [`LazyAlgebra`](https://github.com/emmt/LazyAlgebra.jl) framework.

  - Applying the adjoint of interpolators is fully supported.  This can be
    exploited for iterative fitting of data given an interpolated model.

  - Interpolators may have coefficients computed *on the fly* or tabulated
    (that is computed once).  The former requires almost no memory but can be
    slower than the latter if the same interpolation is applied more than once.


## Table of contents

```@contents
Pages = ["install.md", "interpolation.md", "library.md", "notes.md"]
```

## Index

```@index
```

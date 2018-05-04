# LinearInterpolators

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Build Status](https://travis-ci.org/emmt/LinearInterpolators.jl.svg?branch=master)](https://travis-ci.org/emmt/LinearInterpolators.jl)
[![Coverage Status](https://coveralls.io/repos/emmt/LinearInterpolators.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/emmt/LinearInterpolators.jl?branch=master)
[![codecov.io](http://codecov.io/github/emmt/LinearInterpolators.jl/coverage.svg?branch=master)](http://codecov.io/github/emmt/LinearInterpolators.jl?branch=master)

The **LinearInterpolators** package provides many linear interpolation methods
for [`Julia`](http://julialang.org/).  These interpolations are *linear* in the
sense that the result depends linearly on the input.


## Features

* Separable interpolations are supported for arrays of any dimensionality.
  Interpolation kernel can be different along each interpolated dimension.

* For 2D arrays, interpolations may be separable or not (*e.g.* to apply an
  image rotation).

* Undimensional interpolations may be used to produce multi-dimensional
  results.

* Many interpolation kernels are provided (B-splines of degree 0 to 3, cardinal
  cubic splines, Catmull-Rom spline, Mitchell & Netravali spline, Lanczos
  resampling kernels of arbitrary size, *etc.*).  These interpolation
  kernels may be used as regular functions or to define *interpolators*.

* **Interpolators** are linear maps such as the ones defined by the
  [LazyAlgebra](https://github.com/emmt/LazyAlgebra.jl) framework.

  - Applying the adjoint of interpolators is fully supported.  This can be
    exploited for iterative fitting of data given an interpolated model.

  - Interpolators may have coefficients computed *on the fly* or tabulated
    (that is computed once).  The former requires almost no memory but can be
    slower than the latter if the same interpolation is applied more than once.


## Installation

[LazyAlgebra](https://github.com/emmt/LazyAlgebra.jl) is a prerequisite for
.  To install `LazyAlgebra`, simply do:

```julia
Pkg.clone("https://github.com/emmt/LazyAlgebra.jl.git")
```

LinearInterpolators is not yet an
[official Julia package](https://pkg.julialang.org/) so you have to clone the
repository:

```julia
Pkg.clone("https://github.com/emmt/LinearInterpolators.jl.git")
```

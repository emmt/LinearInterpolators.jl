# LinearInterpolators

[![Doc. Dev][doc-dev-img]][doc-dev-url]
[![License][license-img]][license-url]
[![Build Status][github-ci-img]][github-ci-url]
[![Build Status][appveyor-img]][appveyor-url]
[![Coverage][codecov-img]][codecov-url]

The `LinearInterpolators` package provides many linear interpolation methods
for [Julia][julia-url].  These interpolations are *linear* in the sense
that the result depends linearly on the input.

The documentation for the master version is [here][doc-dev-url].


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


## Installation

The easiest way to install `LinearInterpolators` is via Julia's package manager:

```julia
using Pkg
pkg"add LinearInterpolators"
```

[doc-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[doc-stable-url]: https://emmt.github.io/LinearInterpolators.jl/stable

[doc-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[doc-dev-url]: https://emmt.github.io/LinearInterpolators.jl/dev

[license-url]: ./LICENSE.md
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat

[github-ci-img]: https://github.com/emmt/LinearInterpolators.jl/actions/workflows/CI.yml/badge.svg?branch=master
[github-ci-url]: https://github.com/emmt/LinearInterpolators.jl/actions/workflows/CI.yml?query=branch%3Amaster

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/emmt/LinearInterpolators.jl?branch=master
[appveyor-url]: https://ci.appveyor.com/project/emmt/LinearInterpolators-jl/branch/master

[codecov-img]: http://codecov.io/github/emmt/LinearInterpolators.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/emmt/LinearInterpolators.jl?branch=master

[julia-url]: https://julialang.org/

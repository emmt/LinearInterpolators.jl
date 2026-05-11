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


## Multi-dimensional interpolation

This package aims at providing linear operators to interpolate Julia's arrays.
A general formula to express interpolation of source array `src` to produce
destination array `dst` writes:

$$
\begin{align*}
\mathtt{dst}[i_{\mathrm{pre}}..., i_1, ..., i_M, i_{\mathrm{post}}...] = \sum_{j_1,...,j_N}&\mathtt{ker}_1\bigl((\mathtt{pos}[i_1, ..., i_M]_1 - \mathtt{rng}_1[j_1])/\mathtt{step}(\mathtt{rng}_1)\bigr)\times...\\&\times\mathtt{ker}_N\bigl((\mathtt{pos}[i_1, ..., i_M]_N - \mathtt{rng}_N[j_N])/\mathtt{step}(\mathtt{rng}_N)\bigr)\\&\times\mathtt{src}[i_{\mathrm{pre}}..., j_1, ..., j_N, i_{\mathrm{post}}...]
\end{align*}
$$

where $\mathtt{ker}_j$ is the interpolation kernel along $j$-th dimension of
interpolation, $\mathtt{pos}[i_1, ..., i_M]$ yields the `N`-dimensional
coordinates where to interpolate, and $\mathtt{rng}_j$ is the range (an
instance of `AbstractRange`) specifying the uni-dimensional coordinates along
the $j$-th interpolated dimension of the source array.  Here leading indices
$i_{\mathtt{pre}}$ and trailing indices $i_{\mathrm{post}}$ may have any number
of dimensions (including none). Hence interpolation produces `M`-dimensional
result by interpolating `N` dimensions. Note that the above formula assumes
infinite interpolated dimensions (for indices $j_1$, ..., $j_N$).  In practice,
**boundary conditions** are taken into account to deal with finite size arrays.


## Operators

`SparseInterpolator{T,L,M,N,O,S,W,I}` is the type of linear interpolators with
**pre-computed** coefficients of type `T` and with `L` the number of leading
non-interpolated dimensions (the length of $i_{\mathtt{pre}}$), `M` the number
of dimensions produced by the interpolation, `N` the number of interpolated
dimensions, `O` the ordering of loops, `S` a `N`-tuple of interpolation kernel
sizes, `W` and `I` the types of the objects storing the interpolation weights
and indices.

`LazyInterpolator{T,L,M,N,O,S,P,K,B}` is the type of linear interpolators with
coefficientsof type `T` **computed on the fly** and with `L` the number of
leading non-interpolated dimensions (the length of $i_{\mathtt{pre}}$), `M` the
number of dimensions produced by the interpolation, `N` the number of
interpolated dimensions, `O` the ordering of loops, `S` a `N`-tuple of
interpolation kernel sizes, `P`, `K`, and `B` are the types of the objects
storing the positions where to interpolate, the interpolation kernel(s), and
the boundary conditions. Positions where to interpolate may be specified as an
array of coordinates, as an affine transform (e.g. of type
`AffineTransform{M,N,T}`), or as a coordinate transform function mapping a
`M`-tuple of output indices `(i_1,...,i_M)` to a `N`-tuple of input
coordinates.

Ordering `O` is a 3-digit decimal number.  These digits are a permutation of
`(1,2,3)`: `1` for the index $i_{\mathtt{pre}}$, `2` for the index $i$, and `3`
for the index $i_{\mathtt{post}}$.  For instance, `O = 213` indicates that the
innermost loop is on the index $i$, then on the $i_{\mathtt{pre}}$, and finally
on the $i_{\mathtt{post}}$.

Type parameters `S`, `W`, `I`, `P`, `K`, and `R` are automatically deduced
from the arguments of the constructors.


## Features

- Separable interpolations are supported for arrays of any dimensionality.
  Interpolation kernels can be different along each interpolated dimension.

- For 2D arrays, interpolations may be separable or not (*e.g.* to apply an
  image rotation).

- Undimensional interpolations may be used to produce multi-dimensional
  results.

- Many interpolation kernels are provided by the package
  [`InterpolationKernels`](https://github.com/emmt/InterpolationKernels.jl)
  (B-splines of degree 0 to 3, cardinal cubic splines, Catmull-Rom spline,
  Mitchell & Netravali spline, Lanczos resampling kernels of arbitrary size,
  *etc.*).

- **Interpolators** are linear maps such as the ones defined by the
  [`LazyAlgebra`](https://github.com/emmt/LazyAlgebra.jl) framework.

  - Applying the adjoint of interpolators is fully supported.  This can be
    exploited for iterative fitting of data given an interpolated model.

  - Interpolators may have coefficients computed *on the fly* or tabulated
    (that is computed once).  The former requires almost no memory but can be
    slower than the latter if the same interpolation is applied more than once.


## Restrictions

- **Column-major order** is assumed by default for all arrays.  If this is not
  the case, type parameter `O` must be correctly specified otherwise
  performances may be reduced by [page memory
  faults](https://en.wikipedia.org/wiki/Page_fault).

- Multi-dimensional interpolation is **separable** along the dimensions of the
  source array.

- A **constant step** is imposed for coordinates along the interpolated
  dimensions of the source array.  This is needed to efficiently locate the
  neighborhood of the source entries involved in the interpolation formula for
  a given interpolated position.  Indeed, to interpolate at position `x` with
  kernel `ker` along a dimension whose coordinates are specified by range
  `rng`, the following simplifications can be made:
  ```julia
  ker((x - rng[j])/step(rng))
      = ker((x - (first(rng) + step(rng)*(j - firstindex(rng)))/step(rng))
      = ker((x - b)/a - j)
  ```
  with:
  ```julia
  a = step(rng)
  b = first(rng) - a*firstindex(rng)
  ```
  which are constants (given `rng`).


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

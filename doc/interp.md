# Notes About Interpolation

## Definitions

### Notations

Round parenthesis, as in `f(x)`, denote a continuous function (`x` is a real
number), square brakets, as in `a[k]`, denote a sampled function (`k ∈ ℤ` is an
integer number).


### Interpolation

Interpolation amounts to convolving with a kernel `ker(x)`:

```
f(x) = sum_k a[clip(k)]*ker(x - k)
```

where `clip(k)` imposes the boundary conditions and makes sure that the
resulting index is within the bounds of array `a`.

It can be seen that interpolation acts as a linear filter.  Finite impulse
response (FIR) filters have a finite support.  By convention we use centered
kernels whose support is `(-s/2,+s/2)` with `s`the width of the support.
Infinite impulse response (IIR) filters have an infinite support.


### Floor and ceil functions

Definitions of the `floor()` and `ceil()` functions (`∀ x ∈ ℝ`):

```
floor(x) = ⌊x⌋ = k ∈ ℤ   s.x.  k ≤ x < k+1
 ceil(x) = ⌈x⌉ = k ∈ ℤ   s.x.  k-1 < x ≤ k
```

As a consequence (`∀ x ∈ ℝ` and `∀ k ∈ ℤ`):

```
floor(x) ≤ k   <=>   x < k+1       (1a)
floor(x) < k   <=>   x < k         (1b)

floor(x) ≥ k   <=>   x ≥ k         (1c)
floor(x) > k   <=>   x ≥ k+1       (1d)

ceil(x) ≤ k    <=>   x ≤ k         (2a)
ceil(x) < k    <=>   x ≤ k-1       (2b)

ceil(x) ≥ k    <=>   x > k-1       (2c)
ceil(x) > k    <=>   x > k         (2d)
```


## Kernel support and neighbors indices

### General support

Let `(a,b)` with `a < b` be the support of the kernel.  We assume that the
support size is strict, i.e. `ker(x) = 0` if `x ≤ a` or `x ≥ b`.  Thus, for a
given `x`, the *neighbors* indices `k` to take into account in the
interpolation formula are such that:

```
a < x - k < b     <=>    x - b < k < x - a
```

because outside this range, `ker(x - k) = 0`.

Using the equivalences `(1b)` and `(2d)`, the *neighbors* indices `k`
are those for which:

```
floor(x - b) < k < ceil(x - a)
```

holds.  Equivalently:

```
floor(x - b + 1) ≤ k ≤ ceil(x - a - 1)
```

The first index to take into account is `kfirst = floor(x - b + 1)` and the
last index to take into account is `klast = ceil(x - a - 1)`.


### Symmetric integer support

Let `s = |b - a|` denotes the width of the support of the kernel.  We now
assume that the support size is integer (`s ∈ ℕ`), symmetric (`a = -s/2` and `b
= +s/2`), and strict (`ker(x) = 0` if `|x| ≥ s/2`).  Thus, for a given `x`, the
*neighbors* indices `k` to take into account are such that:

```
|x - k| < s/2   <=>   x - s/2 < k < x + s/2
                <=>   floor(x - s/2 + 1) ≤ k ≤ ceil(x + s/2 - 1)
```

The number of indices in the above range is equal to `s` unless `x` is integer
while `s` is even or `x` is half-integer while `s` is odd.  For these specific
cases, there are `s - 1` indices in the range.  However, always having the same
number (`s`) of indices to consider yields code easier to write and optimize.
We therefore choose that the first index `k1` and last index `ks` to take
into account are either:

- `k1 = floor(x - s/2 + 1)` and `ks = k1 + s - 1`;

- or `ks = ceil(x + s/2 - 1)` and `k1 = ks - s + 1`.

For the specific values of `x` aforementioned, one of `ker(x - k1) = 0` or
`ker(x - ks) = 0` holds.  For other values of `x`, the two choices are
equivalent.

In what follows, we choose to define the first index (before clipping) by:

```
k1 = k0 + 1
```

with

```
k0 = floor(x - s/2)
```

and all indices to consider are:

```
k = k0 + 1, k0 + 2, ..., k0 + s
```


## Clipping

Now we have the constraint that: `kmin ≤ k ≤ kmax`.  If we apply a *"nearest
bound"* condition, then:

- if `ks = k0 + s ≤ kmin`, then **all** infices `k` are clipped to `kmin`;
  using the fact that `s` is integer and equivalence `(1a)`, this occurs
  whenever:

  ```
        kmin ≥ k0 + s = floor(x - s/2) + s = floor(x + s/2)
  <=>   x < kmin - s/2 + 1
  ```

- if `kmax ≤ k1 = k0 + 1`, then **all** indices `k` are clipped to `kmax`;
  using equivalence `(1c)`, this occurs whenever:

  ```
        kmax ≤ k0 + 1 = floor(x - s/2 + 1)
  <=>   x ≥ kmax + s/2 - 1
  ```

These cases have to be considered before computing `k0 = (int)floor(x - s/2)`
not only for optimization reasons but also because `floor(...)` may be beyond
the limits of a numerical integer.

The most simple case is when all considered indices are within the bounds
which, using equivalences `(1a)` and `(1c)`, implies:

```
      kmin ≤ k0 + 1   and   k0 + s ≤ kmax
<=>   kmin + s/2 - 1 ≤ x < kmax - s/2 + 1
```


## Efficient computation of coefficients

For a given value of `x` the coefficients of the interpolation are given by:

```
w[i] = ker(x - k0 - i)
```

with `k0 = floor(x - s/2)` and for `i = 1, 2, ..., s`.

Note that there must be no clipping of the indices here, clipping is only for
indexing the interpolated array and depends on the boundary conditions.

Many interpolation kernels (see below) are *splines* which are piecewise
polynomials defined over sub-intervals of size 1.  That is:

```
ker(x) = h[1](x)    for -s/2 ≤ x ≤ 1 - s/2
         h[2](x)    for 1 - s/2 ≤ x ≤ 2 - s/2
         ...
         h[j](x)    for j - 1 - s/2 ≤ x ≤ j - s/2
         ...
         h[s](x)    for s/2 - 1 ≤ x ≤ s/2
```

Hence

```
w[i] = ker(x - k0 - i) = h[s + 1 - i](x - k0 - i)
```

In Julia implementation the interpolation coefficients are computed by
the `getweights()` method specialized for each type of kernel an called
as:

```julia
getweights(ker, t) -> w1, w2, ..., wS
```

to get the `S` interpolation weights for a given offset `t` computed as:

```
t = x - floor(x)        if s is even
    x - round(x)        if s is odd
```

Thus `t ∈ [0,1]` if `S` is even or or for `t ∈ [-1/2,+1/2]` if `S` is odd.

There are 2 cases depending on the parity of `s`:

- If `s` is even, then `k0 = floor(x - s/2) = floor(x) - s/2` hence
  `t = x - floor(x) = x - k0 - s/2`.

- If `s` is odd, then `k0 = floor(x - s/2) = floor(x + 1/2) - (s + 1)/2`
  `round(x) = floor(x + 1/2) = k0 + (s + 1)/2` and
  `t = x - round(x) = x - k0 - (s + 1)/2`.

Therefore the argument of `h[s + 1 - i](...)` is:

```
x - k0 - i = t + s/2 - i          if s is even
             t + (s + 1)/2 - i    if s is odd
```

or:

```
x - k0 - i = t + ⌊(s + 1)/2⌋ - i
```

whatever the parity of `s`.


## Cubic Interpolation

Keys's cubic interpolation kernels are given by:

```
ker(x) = (a+2)*|x|^3 - (a+3)*|x|^2 + 1          for |x| ≤ 1
       = a*|x|^3 - 5*a*|x|^2 + 8*a*|x| - 4*a    for 1 ≤ |x| ≤ 2
       = 0                                      else
```

Mitchell and Netravali family of piecewise cubic filters (which depend on 2
parameters, `b` and `c`) are given by:

```
ker(x) = (1/6)*((12 - 9*b - 6*c)*|x|^3
         + (-18 + 12*b + 6*c)*|x|^2 + (6 - 2*B))        for |x| ≤ 1
       = (1/6)*((-b - 6*c)*|x|^3 + (6*b + 30*c)*|x|^2
         + (-12*b - 48*c)*|x| + (8*b + 24*c))           for 1 ≤ |x| ≤ 2
       = 0                                              else
```

These kernels are continuous, symmetric, have continuous 1st derivatives and
sum of coefficients is one (needs not be normalized).  Using the constraint:

```
b + 2*c = 1
```

yields a cubic filter with, at least, quadratic order approximation.

```
(b,c) = (1,0)     ==> cubic B-spline
(b,c) = (0, -a)   ==> Keys's cardinal cubics
(b,c) = (0,1/2)   ==> Catmull-Rom cubics
(b,c) = (b,0)     ==> Duff's tensioned B-spline
(b,c) = (1/3,1/3) ==> recommended by Mitchell-Netravali
```

See paper by [Mitchell and Netravali, *"Reconstruction Filters in Computer
Graphics Computer Graphics"*, Volume **22**, Number 4,
(1988)][Mitchell-Netravali-pdf].


## Resampling

Resampling/interpolation is done by:

```
dst[i] = sum_j ker(grd[j] - pos[i])*src[j]
```

with `dst` the resulting array, `src` the values of a function sampled on a
regular grid `grd`, `ker` the interpolation kernel, and `pos` the coordinates
where to interpolate the sampled function.

To limit the storage and the number of operations, we want to determine the
range of indices for which the kernel function is non-zero.  We have that:

```
abs(x) ≥ s/2  ==>  ker(x) = 0
```

with `s = length(ker)`.  Taking `x = grd[j] - pos[i]`, then:

```
abs(x) < s/2  <==>  pos[i] - s/2 < grd[j] < pos[i] + s/2
```

`grd` is a `Range` so an "exact" formula for its elements is given by a linear
interpolation:

```
grd[j] = t0*(j1 - j)/(j1 - j0) + t1*(j - j0)/(j1 - j0)
```

with:

```
    j0 = 1             # first grid index
    j1 = length(grd)   # last grid index
    t0 = first(grd)    # first grid coordinate
    t1 = last(grd)     # last grid coordinate
```

Note that:

```
    d = (t1 - t0)/(j1 - j0)
      = step(grd)
```

is the increment between adjacent grip nodes (may be negative).

The reverse formula in the form of a linear interpolation is:

```
    f[i] = j0*(t1 - pos[i])/(t1 - t0) + j1*(pos[i] - t0)/(t1 - t0)
```

which yields a (fractional) grid index `f` for coordinate `pos[i]`.  Taking
care of the fact that the grid step may be negative, the maximum number of
non-zero coefficients is given by:

```
    M = floor(Int, length(ker)/abs(step(grd)))
      = floor(Int, w)
```

with:

```
    w = length(ker)/abs(d)    # beware d = step(grd) may be negative
```

the size of the support in grid index units.  The indices of the first non-zero
coefficients are:

```
    j = l[i] + k
```

for `k = 1, 2, ..., M` and

```
    l[i] = ceil(Int, f[i] - w/2) - 1
```

the corresponding offsets are (assuming the grid extends infinitely):

```
    x[i,j] = pos[i] - grd[j]
           = pos[i] - (grd[0] + (l[i] + k)*d)
           = ((pos[i] - grd[0])/d - l[i] - k)*d
           = (f[i] - l[i] - k)*d
```

The final issue to address is to avoid `InexactError` exceptions.


[Mitchell-Netravali-pdf]: http://www.cs.utexas.edu/users/fussell/courses/cs384g/lectures/mitchell/Mitchell.pdf

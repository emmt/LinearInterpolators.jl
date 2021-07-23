var documenterSearchIndex = {"docs":
[{"location":"notes/#Notes-about-interpolation","page":"Notes about interpolation","title":"Notes about interpolation","text":"","category":"section"},{"location":"notes/#Definitions","page":"Notes about interpolation","title":"Definitions","text":"","category":"section"},{"location":"notes/#Notations","page":"Notes about interpolation","title":"Notations","text":"","category":"section"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"Round parenthesis, as in f(x), denote a continuous function (x is a real number), square brakets, as in a[k], denote a sampled function (k ∈ ℤ is an integer number).","category":"page"},{"location":"notes/#Interpolation","page":"Notes about interpolation","title":"Interpolation","text":"","category":"section"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"Interpolation amounts to convolving with a kernel ker(x):","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"f(x) = sum_k a[clip(k)]*ker(x - k)","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"where clip(k) imposes the boundary conditions and makes sure that the resulting index is within the bounds of array a.","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"It can be seen that interpolation acts as a linear filter.  Finite impulse response (FIR) filters have a finite support.  By convention we use centered kernels whose support is (-s/2,+s/2) with sthe width of the support. Infinite impulse response (IIR) filters have an infinite support.","category":"page"},{"location":"notes/#Floor-and-ceil-functions","page":"Notes about interpolation","title":"Floor and ceil functions","text":"","category":"section"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"Definitions of the floor() and ceil() functions (∀ x ∈ ℝ):","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"floor(x) = ⌊x⌋ = k ∈ ℤ   s.x.  k ≤ x < k+1\n ceil(x) = ⌈x⌉ = k ∈ ℤ   s.x.  k-1 < x ≤ k","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"As a consequence (∀ x ∈ ℝ and ∀ k ∈ ℤ):","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"floor(x) ≤ k   <=>   x < k+1       (1a)\nfloor(x) < k   <=>   x < k         (1b)\n\nfloor(x) ≥ k   <=>   x ≥ k         (1c)\nfloor(x) > k   <=>   x ≥ k+1       (1d)\n\nceil(x) ≤ k    <=>   x ≤ k         (2a)\nceil(x) < k    <=>   x ≤ k-1       (2b)\n\nceil(x) ≥ k    <=>   x > k-1       (2c)\nceil(x) > k    <=>   x > k         (2d)","category":"page"},{"location":"notes/#Kernel-support-and-neighbors-indices","page":"Notes about interpolation","title":"Kernel support and neighbors indices","text":"","category":"section"},{"location":"notes/#General-support","page":"Notes about interpolation","title":"General support","text":"","category":"section"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"Let (a,b) with a < b be the support of the kernel.  We assume that the support size is strict, i.e. ker(x) = 0 if x ≤ a or x ≥ b.  Thus, for a given x, the neighbors indices k to take into account in the interpolation formula are such that:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"a < x - k < b     <=>    x - b < k < x - a","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"because outside this range, ker(x - k) = 0.","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"Using the equivalences (1b) and (2d), the neighbors indices k are those for which:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"floor(x - b) < k < ceil(x - a)","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"holds.  Equivalently:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"floor(x - b + 1) ≤ k ≤ ceil(x - a - 1)","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"The first index to take into account is kfirst = floor(x - b + 1) and the last index to take into account is klast = ceil(x - a - 1).","category":"page"},{"location":"notes/#Symmetric-integer-support","page":"Notes about interpolation","title":"Symmetric integer support","text":"","category":"section"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"Let s = |b - a| denotes the width of the support of the kernel.  We now assume that the support size is integer (s ∈ ℕ), symmetric (a = -s/2 and b = +s/2), and strict (ker(x) = 0 if |x| ≥ s/2).  Thus, for a given x, the neighbors indices k to take into account are such that:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"|x - k| < s/2   <=>   x - s/2 < k < x + s/2\n                <=>   floor(x - s/2 + 1) ≤ k ≤ ceil(x + s/2 - 1)","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"The number of indices in the above range is equal to s unless x is integer while s is even or x is half-integer while s is odd.  For these specific cases, there are s - 1 indices in the range.  However, always having the same number (s) of indices to consider yields code easier to write and optimize. We therefore choose that the first index k1 and last index ks to take into account are either:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"k1 = floor(x - s/2 + 1) and ks = k1 + s - 1;\nor ks = ceil(x + s/2 - 1) and k1 = ks - s + 1.","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"For the specific values of x aforementioned, one of ker(x - k1) = 0 or ker(x - ks) = 0 holds.  For other values of x, the two choices are equivalent.","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"In what follows, we choose to define the first index (before clipping) by:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"k1 = k0 + 1","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"with","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"k0 = floor(x - s/2)","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"and all indices to consider are:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"k = k0 + 1, k0 + 2, ..., k0 + s","category":"page"},{"location":"notes/#Clipping","page":"Notes about interpolation","title":"Clipping","text":"","category":"section"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"Now we have the constraint that: kmin ≤ k ≤ kmax.  If we apply a \"nearest bound\" condition, then:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"if ks = k0 + s ≤ kmin, then all infices k are clipped to kmin; using the fact that s is integer and equivalence (1a), this occurs whenever:\n      kmin ≥ k0 + s = floor(x - s/2) + s = floor(x + s/2)\n<=>   x < kmin - s/2 + 1\nif kmax ≤ k1 = k0 + 1, then all indices k are clipped to kmax; using equivalence (1c), this occurs whenever:\n      kmax ≤ k0 + 1 = floor(x - s/2 + 1)\n<=>   x ≥ kmax + s/2 - 1","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"These cases have to be considered before computing k0 = (int)floor(x - s/2) not only for optimization reasons but also because floor(...) may be beyond the limits of a numerical integer.","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"The most simple case is when all considered indices are within the bounds which, using equivalences (1a) and (1c), implies:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"      kmin ≤ k0 + 1   and   k0 + s ≤ kmax\n<=>   kmin + s/2 - 1 ≤ x < kmax - s/2 + 1","category":"page"},{"location":"notes/#Efficient-computation-of-coefficients","page":"Notes about interpolation","title":"Efficient computation of coefficients","text":"","category":"section"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"For a given value of x the coefficients of the interpolation are given by:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"w[i] = ker(x - k0 - i)","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"with k0 = floor(x - s/2) and for i = 1, 2, ..., s.","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"Note that there must be no clipping of the indices here, clipping is only for indexing the interpolated array and depends on the boundary conditions.","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"Many interpolation kernels (see below) are splines which are piecewise polynomials defined over sub-intervals of size 1.  That is:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"ker(x) = h[1](x)    for -s/2 ≤ x ≤ 1 - s/2\n         h[2](x)    for 1 - s/2 ≤ x ≤ 2 - s/2\n         ...\n         h[j](x)    for j - 1 - s/2 ≤ x ≤ j - s/2\n         ...\n         h[s](x)    for s/2 - 1 ≤ x ≤ s/2","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"Hence","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"w[i] = ker(x - k0 - i) = h[s + 1 - i](x - k0 - i)","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"In Julia implementation the interpolation coefficients are computed by the getweights() method specialized for each type of kernel an called as:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"getweights(ker, t) -> w1, w2, ..., wS","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"to get the S interpolation weights for a given offset t computed as:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"t = x - floor(x)        if s is even\n    x - round(x)        if s is odd","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"Thus t ∈ [0,1] if S is even or or for t ∈ [-1/2,+1/2] if S is odd.","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"There are 2 cases depending on the parity of s:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"If s is even, then k0 = floor(x - s/2) = floor(x) - s/2 hence t = x - floor(x) = x - k0 - s/2.\nIf s is odd, then k0 = floor(x - s/2) = floor(x + 1/2) - (s + 1)/2 round(x) = floor(x + 1/2) = k0 + (s + 1)/2 and t = x - round(x) = x - k0 - (s + 1)/2.","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"Therefore the argument of h[s + 1 - i](...) is:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"x - k0 - i = t + s/2 - i          if s is even\n             t + (s + 1)/2 - i    if s is odd","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"or:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"x - k0 - i = t + ⌊(s + 1)/2⌋ - i","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"whatever the parity of s.","category":"page"},{"location":"notes/#Cubic-Interpolation","page":"Notes about interpolation","title":"Cubic Interpolation","text":"","category":"section"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"Keys's cubic interpolation kernels are given by:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"ker(x) = (a+2)*|x|^3 - (a+3)*|x|^2 + 1          for |x| ≤ 1\n       = a*|x|^3 - 5*a*|x|^2 + 8*a*|x| - 4*a    for 1 ≤ |x| ≤ 2\n       = 0                                      else","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"Mitchell and Netravali family of piecewise cubic filters (which depend on 2 parameters, b and c) are given by:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"ker(x) = (1/6)*((12 - 9*b - 6*c)*|x|^3\n         + (-18 + 12*b + 6*c)*|x|^2 + (6 - 2*B))        for |x| ≤ 1\n       = (1/6)*((-b - 6*c)*|x|^3 + (6*b + 30*c)*|x|^2\n         + (-12*b - 48*c)*|x| + (8*b + 24*c))           for 1 ≤ |x| ≤ 2\n       = 0                                              else","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"These kernels are continuous, symmetric, have continuous 1st derivatives and sum of coefficients is one (needs not be normalized).  Using the constraint:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"b + 2*c = 1","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"yields a cubic filter with, at least, quadratic order approximation.","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"(b,c) = (1,0)     ==> cubic B-spline\n(b,c) = (0, -a)   ==> Keys's cardinal cubics\n(b,c) = (0,1/2)   ==> Catmull-Rom cubics\n(b,c) = (b,0)     ==> Duff's tensioned B-spline\n(b,c) = (1/3,1/3) ==> recommended by Mitchell-Netravali","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"See paper by [Mitchell and Netravali, \"Reconstruction Filters in Computer Graphics Computer Graphics\", Volume 22, Number 4, (1988)][Mitchell-Netravali-pdf].","category":"page"},{"location":"notes/#Resampling","page":"Notes about interpolation","title":"Resampling","text":"","category":"section"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"Resampling/interpolation is done by:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"dst[i] = sum_j ker(grd[j] - pos[i])*src[j]","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"with dst the resulting array, src the values of a function sampled on a regular grid grd, ker the interpolation kernel, and pos the coordinates where to interpolate the sampled function.","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"To limit the storage and the number of operations, we want to determine the range of indices for which the kernel function is non-zero.  We have that:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"abs(x) ≥ s/2  ==>  ker(x) = 0","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"with s = length(ker).  Taking x = grd[j] - pos[i], then:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"abs(x) < s/2  <==>  pos[i] - s/2 < grd[j] < pos[i] + s/2","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"grd is a Range so an \"exact\" formula for its elements is given by a linear interpolation:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"grd[j] = t0*(j1 - j)/(j1 - j0) + t1*(j - j0)/(j1 - j0)","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"with:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"    j0 = 1             # first grid index\n    j1 = length(grd)   # last grid index\n    t0 = first(grd)    # first grid coordinate\n    t1 = last(grd)     # last grid coordinate","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"Note that:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"    d = (t1 - t0)/(j1 - j0)\n      = step(grd)","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"is the increment between adjacent grip nodes (may be negative).","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"The reverse formula in the form of a linear interpolation is:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"    f[i] = j0*(t1 - pos[i])/(t1 - t0) + j1*(pos[i] - t0)/(t1 - t0)","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"which yields a (fractional) grid index f for coordinate pos[i].  Taking care of the fact that the grid step may be negative, the maximum number of non-zero coefficients is given by:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"    M = floor(Int, length(ker)/abs(step(grd)))\n      = floor(Int, w)","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"with:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"    w = length(ker)/abs(d)    # beware d = step(grd) may be negative","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"the size of the support in grid index units.  The indices of the first non-zero coefficients are:","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"    j = l[i] + k","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"for k = 1, 2, ..., M and","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"    l[i] = ceil(Int, f[i] - w/2) - 1","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"the corresponding offsets are (assuming the grid extends infinitely):","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"    x[i,j] = pos[i] - grd[j]\n           = pos[i] - (grd[0] + (l[i] + k)*d)\n           = ((pos[i] - grd[0])/d - l[i] - k)*d\n           = (f[i] - l[i] - k)*d","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"The final issue to address is to avoid InexactError exceptions.","category":"page"},{"location":"notes/","page":"Notes about interpolation","title":"Notes about interpolation","text":"[Mitchell-Netravali-pdf]: http://www.cs.utexas.edu/users/fussell/courses/cs384g/lectures/mitchell/Mitchell.pdf","category":"page"},{"location":"library/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"library/","page":"Reference","title":"Reference","text":"The following provides detailled documentation about types and methods provided by the LinearInterpolators package.  This information is also available from the REPL by typing ? followed by the name of a method or a type.","category":"page"},{"location":"library/#Tabulated-Interpolators","page":"Reference","title":"Tabulated Interpolators","text":"","category":"section"},{"location":"library/","page":"Reference","title":"Reference","text":"LinearInterpolators.Interpolations.TabulatedInterpolators.TabulatedInterpolator","category":"page"},{"location":"library/#LinearInterpolators.Interpolations.TabulatedInterpolators.TabulatedInterpolator","page":"Reference","title":"LinearInterpolators.Interpolations.TabulatedInterpolators.TabulatedInterpolator","text":"TabulatedInterpolator([T,] [d = nothing,] ker, pos, nrows, ncols)\n\nyields a linear map to interpolate with kernel ker along a dimension of length ncols to produce a dimension of length nrows.  The function pos(i) for i ∈ 1:nrows gives the positions to interpolate in fractional index unit along a dimension of length ncols.  Argument d is the rank of the dimension along which to interpolate when the operator is applied to a multidimensional array.  If it is unspecifed when the operator is created, it will have to be specified each time the operator is applied (see below).\n\nThe positions to interpolate can also be specified by a vector x as in:\n\nTabulatedInterpolator([T,] [d = nothing,] ker, x, ncols)\n\nto produce an interpolator whose output dimension is nrows = length(x).  Here x can be an abstract vector.\n\nOptional argument T is the floating-point type of the coefficients of the linear map.  By default, it is given by the promotion of the element type of the arguments ker and, if specified, x.\n\nThe tabulated operator, say A, can be applied to an argument x:\n\napply!(α, P::Operations, A, [d,] x, scratch, β, y) -> y\n\nto overwrite y with α*P(A)⋅x + β*y.  If x and y are multi-dimensional, the dimension d to interpolate must be specified.\n\n\n\n\n\n","category":"type"},{"location":"library/#Two-Dimensional-Interpolators","page":"Reference","title":"Two Dimensional Interpolators","text":"","category":"section"},{"location":"library/","page":"Reference","title":"Reference","text":"LinearInterpolators.Interpolations.TwoDimensionalTransformInterpolator","category":"page"},{"location":"library/#LinearInterpolators.Interpolations.TwoDimensionalTransformInterpolator","page":"Reference","title":"LinearInterpolators.Interpolations.TwoDimensionalTransformInterpolator","text":"TwoDimensionalTransformInterpolator(rows, cols, ker1, ker2, R)\n\nyields a linear mapping which interpolate its input of size cols to produce an output of size rows by 2-dimensional interpolation with kernels ker1 and ker2 along each dimension and applying the affine coordinate transform specified by R.\n\nAs a shortcut:\n\nTwoDimensionalTransformInterpolator(rows, cols, ker, R)\n\nis equivalent to TwoDimensionalTransformInterpolator(rows,cols,ker,ker,R) that is the same kernel is used along all dimensions.\n\n\n\n\n\n","category":"type"},{"location":"library/#Sparse-Interpolators","page":"Reference","title":"Sparse Interpolators","text":"","category":"section"},{"location":"library/","page":"Reference","title":"Reference","text":"LinearInterpolators.Interpolations.SparseInterpolators.SparseInterpolator\nLinearInterpolators.Interpolations.SparseInterpolators.fit\nLinearInterpolators.Interpolations.SparseInterpolators.regularize\nLinearInterpolators.Interpolations.SparseInterpolators.regularize!\nLinearInterpolators.Interpolations.SparseInterpolators.SparseUnidimensionalInterpolator","category":"page"},{"location":"library/#LinearInterpolators.Interpolations.SparseInterpolators.SparseInterpolator","page":"Reference","title":"LinearInterpolators.Interpolations.SparseInterpolators.SparseInterpolator","text":"A = SparseInterpolator([T=eltype(ker),] ker, pos, grd)\n\nyields a sparse linear interpolator suitable for interpolating with kernel ker a function sampled on the grid grd at positions pos.  Optional argument T is the floating-point type of the coefficients of the operator A.  Call eltype(A) to query the type of the coefficients of the sparse interpolator A.\n\nThen y = apply(A, x) or y = A(x) or y = A*x yields the interpolated values for interpolation weights x.  The shape of y is the same as that of pos.  Formally, this amounts to computing:\n\ny[i] = sum_j ker((pos[i] - grd[j])/step(grd))*x[j]\n\nwith step(grd) the (constant) step size between the nodes of the grid grd and grd[j] the j-th position of the grid.\n\n\n\n\n\n","category":"type"},{"location":"library/#LinearInterpolators.Interpolations.fit","page":"Reference","title":"LinearInterpolators.Interpolations.fit","text":"fit(A, y [, w]; epsilon=1e-9, mu=0.0) -> x\n\nperforms a linear fit of y by the model A*x with A a linear interpolator. The returned value x minimizes:\n\nsum(w.*(A*x - y).^2)\n\nwhere w are given weights.  If w is not specified, all weights are assumed to be equal to one; otherwise w must be an array of nonnegative values and of same size as y.\n\nKeywords epsilon and mu may be specified to regularize the solution and minimize:\n\nsum(w.*(A*x - y).^2) + rho*(epsilon*norm(x)^2 + mu*norm(D*x)^2)\n\nwhere D is a finite difference operator, rho is the maximum diagonal element of A'*diag(w)*A and norm is the Euclidean norm.\n\n\n\n\n\n","category":"function"},{"location":"library/#LinearInterpolators.Interpolations.regularize","page":"Reference","title":"LinearInterpolators.Interpolations.regularize","text":"regularize(A, ϵ, μ) -> R\n\nregularizes the symmetric matrix A to produce the matrix:\n\nR = A + ρ*(ϵ*I + μ*D'*D)\n\nwhere I is the identity, D is a finite difference operator and ρ is the maximum diagonal element of A.\n\n\n\n\n\n","category":"function"},{"location":"library/#LinearInterpolators.Interpolations.regularize!","page":"Reference","title":"LinearInterpolators.Interpolations.regularize!","text":"regularize!(A, ϵ, μ) -> A\n\nstores the regularized matrix in A (and returns it).  This is the in-place version of [LinearInterpolators.SparseInterpolators.regularize].\n\n\n\n\n\n","category":"function"},{"location":"library/#LinearInterpolators.Interpolations.SparseInterpolators.SparseUnidimensionalInterpolator","page":"Reference","title":"LinearInterpolators.Interpolations.SparseInterpolators.SparseUnidimensionalInterpolator","text":"SparseUnidimensionalInterpolator([T=eltype(ker),] ker, d, pos, grd)\n\nyields a linear mapping which interpolates the d-th dimension of an array with kernel ker at positions pos along the dimension of interpolation d and assuming the input array has grid coordinates grd along the the d-th dimension of interpolation.  Argument pos is a vector of positions, argument grd may be a range or the length of the dimension of interpolation.  Optional argument T is the floating-point type of the coefficients of the operator.\n\nThis kind of interpolator is suitable for separable multi-dimensional interpolation with precomputed interpolation coefficients.  Having precomputed coefficients is mostly interesting when the operator is to be applied multiple times (for instance in iterative methods).  Otherwise, separable operators which compute the coefficients on the fly may be preferable.\n\nA combination of instances of SparseUnidimensionalInterpolator can be built to achieve sperable multi-dimensional interpolation.  For example:\n\nusing LinearInterpolators\nker = CatmullRomSpline()\nn1, n2 = 70, 50\nx1 = linspace(1, 70, 201)\nx2 = linspace(1, 50, 201)\nA1 = SparseUnidimensionalInterpolator(ker, 1, x1, 1:n1)\nA2 = SparseUnidimensionalInterpolator(ker, 2, x2, 1:n2)\nA = A1*A2\n\n\n\n\n\n","category":"type"},{"location":"library/#Utilities","page":"Reference","title":"Utilities","text":"","category":"section"},{"location":"library/#Limits","page":"Reference","title":"Limits","text":"","category":"section"},{"location":"library/","page":"Reference","title":"Reference","text":"LinearInterpolators.Interpolations.Limits\nLinearInterpolators.Interpolations.limits","category":"page"},{"location":"library/#LinearInterpolators.Interpolations.Limits","page":"Reference","title":"LinearInterpolators.Interpolations.Limits","text":"All interpolation limits inherit from the abstract type Limits{T} where T is the floating-point type.  Interpolation limits are the combination of an extrapolation method and the length of the dimension to interpolate.\n\n\n\n\n\n","category":"type"},{"location":"library/#LinearInterpolators.Interpolations.limits","page":"Reference","title":"LinearInterpolators.Interpolations.limits","text":"limits(ker::Kernel, len)\n\nyields the concrete type descendant of Limits for interpolation with kernel ker along a dimension of length len and applying the boundary conditions embedded in ker.\n\n\n\n\n\n","category":"function"},{"location":"library/#Interpolation-coefficients","page":"Reference","title":"Interpolation coefficients","text":"","category":"section"},{"location":"library/","page":"Reference","title":"Reference","text":"LinearInterpolators.Interpolations.getcoefs","category":"page"},{"location":"library/#LinearInterpolators.Interpolations.getcoefs","page":"Reference","title":"LinearInterpolators.Interpolations.getcoefs","text":"getcoefs(ker, lim, x) -> j1, j2, ..., w1, w2, ...\n\nyields the indexes of the neighbors and the corresponding interpolation weights for interpolating at position x by kernel ker with the limits implemented by lim.\n\n\n\n\n\n","category":"function"},{"location":"install/#Installation","page":"Installation","title":"Installation","text":"","category":"section"},{"location":"install/","page":"Installation","title":"Installation","text":"The easiest way to install LinearInterpolators is via Julia registry EmmtRegistry:","category":"page"},{"location":"install/","page":"Installation","title":"Installation","text":"using Pkg\npkg\"registry add https://github.com/emmt/EmmtRegistry\"\npkg\"add LinearInterpolators\"","category":"page"},{"location":"interpolation/#Linear-interpolation","page":"Linear interpolation","title":"Linear interpolation","text":"","category":"section"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"Here linear means that the result depends linearly on the interpolated array. The interpolation functions (or kernels) may be linear or not (e.g., cubic spline).","category":"page"},{"location":"interpolation/#Unidimensional-interpolation","page":"Linear interpolation","title":"Unidimensional interpolation","text":"","category":"section"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"Unidimensional interpolation is done by:","category":"page"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"apply(ker, x, src) -> dst","category":"page"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"which interpolates source array src with kernel ker at positions x, the result is an array of same dimensions as x.  The destination array can be provided:","category":"page"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"apply!(dst, ker, x, src) -> dst","category":"page"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"which overwrites dst with the result of the interpolation of source src with kernel ker at positions specified by x.  If x is an array, dst must have the same size as x; otherwise, x may be a fonction which is applied to all indices of dst (as generated by eachindex(dst)) to produce the coordinates where to interpolate the source.  The destination dst is returned.","category":"page"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"The adjoint/direct operation can be applied:","category":"page"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"apply(P, ker, x, src) -> dst\napply!(dst, P, ker, x, src) -> dst","category":"page"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"where P is either Adjoint or Direct.  If P is omitted, Direct is assumed.","category":"page"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"To linearly combine the result and the contents of the destination array, the following syntax is also implemented:","category":"page"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"apply!(α, P, ker, x, src, β, dst) -> dst","category":"page"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"which overwrites dst with β*dst plus α times the result of the operation implied by P (Direct or Adjoint) on source src with kernel ker at positions specified by x.","category":"page"},{"location":"interpolation/#Separable-multi-dimensional-interpolation","page":"Linear interpolation","title":"Separable multi-dimensional interpolation","text":"","category":"section"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"Separable multi-dimensional interpolation consists in interpolating each dimension of the source array with, possibly, different kernels and at given positions.  For instance:","category":"page"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"apply(ker1, x1, [ker2=ker1,] x2, src) -> dst","category":"page"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"yields the 2D separable interpolation of src with kernel ker1 at positions x1 along the first dimension of src and with kernel ker2 at positions x2 along the second dimension of src.  Note that, if omitted the second kernel is assumed to be the same as the first one.  The above example extends to more dimensions (providing it is implemented).  Positions x1, x2, ... must be unidimensional arrays their lengths give the size of the result of the interpolation.","category":"page"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"The apply the adjoint and/or linearly combine the result of the interpolation and the contents of the destination array, the same methods as for unidimensional interpolation are supported, it is sufficient to replace arguments ker,x by ker1,x1,[ker2=ker1,]x2.","category":"page"},{"location":"interpolation/#Nonseparable-multi-dimensional-interpolation","page":"Linear interpolation","title":"Nonseparable multi-dimensional interpolation","text":"","category":"section"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"Nonseparable 2D interpolation is implemented where the coordinates to interpolate are given by an affine transform which converts the indices in the destination array into fractional coordinates in the source array (for the direct operation).  The syntax is:","category":"page"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"apply!(dst, [P=Direct,] ker1, [ker2=ker1,] R, src) -> dst","category":"page"},{"location":"interpolation/","page":"Linear interpolation","title":"Linear interpolation","text":"where R is an AffineTransform2D and P is Direct (the default) or Adjoint.","category":"page"},{"location":"#Introduction","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"The LinearInterpolators package provides many linear interpolation methods for Julia. These interpolations are linear in the sense that the result depends linearly on the input.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"The source code is on GitHub.","category":"page"},{"location":"#Features","page":"Introduction","title":"Features","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Separable interpolations are supported for arrays of any dimensionality. Interpolation kernels can be different along each interpolated dimension.\nFor 2D arrays, interpolations may be separable or not (e.g. to apply an image rotation).\nUndimensional interpolations may be used to produce multi-dimensional results.\nMany interpolation kernels are provided by the package InterpolationKernels (B-splines of degree 0 to 3, cardinal cubic splines, Catmull-Rom spline, Mitchell & Netravali spline, Lanczos resampling kernels of arbitrary size, etc.).\nInterpolators are linear maps such as the ones defined by the LazyAlgebra framework.\nApplying the adjoint of interpolators is fully supported.  This can be exploited for iterative fitting of data given an interpolated model.\nInterpolators may have coefficients computed on the fly or tabulated (that is computed once).  The former requires almost no memory but can be slower than the latter if the same interpolation is applied more than once.","category":"page"},{"location":"#Table-of-contents","page":"Introduction","title":"Table of contents","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Pages = [\"install.md\", \"interpolation.md\", \"library.md\", \"notes.md\"]","category":"page"},{"location":"#Index","page":"Introduction","title":"Index","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"","category":"page"}]
}

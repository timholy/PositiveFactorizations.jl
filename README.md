# PositiveFactorizations

[![Build Status](https://travis-ci.org/timholy/PositiveFactorizations.jl.svg?branch=master)](https://travis-ci.org/timholy/PositiveFactorizations.jl)

## Overview

PositiveFactorizations is a package for computing a positive-definite
matrix decomposition (factorization) from an arbitrary symmetric
input.  The motivating application is optimization (Newton or
quasi-Newton methods), in which the cannonical search direction `-H\g`
(`H` being the Hessian and `g` the gradient) may not be a descent
direction if `H` is not positive definite.  This package provides an
efficient approach to computing `-Htilde\g`, where `Htilde` is equal
to `H` if `H` is positive-definite, and otherwise is a
positive-definite matrix that is "spiritually like `H`."

The approach favored here is different from the well-known
Gill-Murray-Wright approach of computing the Cholesky factorization of
`H+E`, where `E` is a minimal correction needed to make `H+E`
positive-definite (sometimes known as GMW81).  See the discussion
starting
[here](https://github.com/JuliaOpt/Optim.jl/issues/153#issuecomment-161268535)
for justification; briefly, the idea of a small correction conflates
large negative eigenvalues with numerical roundoff error, which (when
stated that way) seems like a puzzling choice.  In contrast, this
package provides methods that are closer in spirit to taking the
absolute value of the eigenvalues (for a diagonal matrix, this would
be twice the size of the correction in GMW81), and setting any "tiny"
eigenvalues (those consistent with roundoff error) to 1.

## Usage

Given a symmetric matrix `H`, compute a positive factorization `F` like this:

```jl
F = cholfact(Positive, H, [pivot=Val{false}])
```

Pivoting (turned on with `Val{true}`) can make the correction smaller,
but is not necessary for numeric stability.

`cholfact` is recommended because it is very efficient.  A slower alternative is to use `eigfact`:

```jl
F = eigfact(Positive, H)
```

which may be easier to reason about from the standpoint of fundamental linear algebra.

__precompile__()

module PositiveFactorizations

using Compat
using Compat: view
using Compat.LinearAlgebra

export Positive

struct Positive{T<:Real} end

include("cholesky.jl")
include("eig.jl")

end # module

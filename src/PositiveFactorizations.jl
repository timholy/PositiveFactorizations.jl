__precompile__()

module PositiveFactorizations

using Compat
using Compat.view

export Positive

immutable Positive{T<:Real} end

include("cholesky.jl")
include("eig.jl")

end # module

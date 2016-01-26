__precompile__()

module PositiveFactorizations

export Positive

immutable Positive{T<:Real} end

include("cholesky.jl")
include("eig.jl")

end # module

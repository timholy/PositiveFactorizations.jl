using Compat.LinearAlgebra: Eigen
import Compat.LinearAlgebra: eigfact

function eigfact(::Type{Positive{T}}, A::AbstractMatrix{T}, args...; tol=default_tol(A)) where {T}
    F = eigfact(A, args...)
    for i = 1:size(A,1)
        tmp = abs(F.values[i])
        if tmp < tol
            tmp = one(tmp)
        end
        F.values[i] = tmp
    end
    F
end
eigfact(::Type{Positive}, A::AbstractMatrix, args...; tol=default_tol(A)) = eigfact(Positive{floattype(eltype(A))}, A, args...; tol=tol)
eigfact(::Type{Positive{T}}, A::AbstractMatrix, args...; tol=default_tol(A)) where {T} = eigfact(Positive{T}, convert(Matrix{T}, A), args...; tol=tol)

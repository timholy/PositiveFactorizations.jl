using Base.LinAlg: Eigen

function Base.eigfact{T}(::Type{Positive{T}}, A::AbstractMatrix{T}, args...; tol=default_tol(A))
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
Base.eigfact(::Type{Positive}, A::AbstractMatrix, args...; tol=default_tol(A)) = eigfact(Positive{floattype(eltype(A))}, A, args...; tol=tol)
Base.eigfact{T}(::Type{Positive{T}}, A::AbstractMatrix, args...; tol=default_tol(A)) = eigfact(Positive{T}, convert(Matrix{T}, A), args...; tol=tol)

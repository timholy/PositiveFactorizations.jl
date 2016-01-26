using PositiveFactorizations
using Base.Test

for pivot in (Val{false}, Val{true})
    A = [1 0; 0 1]
    F = cholfact(Positive, A, pivot)
    @test_approx_eq full(F) A

    A = [1 0; 0 -1]
    F = cholfact(Positive, A, pivot)
    @test_approx_eq full(F) eye(2)

    A = [-1 0.5; 0.5 4]
    target = pivot == Val{false} ? [1 -0.5; -0.5 4.5] : [1.125 0.5; 0.5 4]
    F = cholfact(Positive, A, pivot)
    @test_approx_eq full(F) target
    F = cholfact(Positive, 10*A, pivot)
    @test_approx_eq full(F) 10*target

    A = [0 1; 1 0]
    F = cholfact(Positive, A, pivot)
    @test_approx_eq full(F) eye(2)

    A = rand(201,200); A = A'*A
    F = cholfact(Positive, A, pivot)
    @test_approx_eq full(F) A
end

A = [1 0; 0 -2]
F = eigfact(Positive, A)
@test_approx_eq full(F) abs(A)
A = [1 0; 0 0]
F = eigfact(Positive, A)
@test_approx_eq full(F) eye(2)

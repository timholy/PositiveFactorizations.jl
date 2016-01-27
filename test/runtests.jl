using PositiveFactorizations
using Base.Test

for pivot in (Val{false}, Val{true})
    A = [1 0; 0 1]
    F = cholfact(Positive, A, pivot)
    @test_approx_eq full(F) A
    F, d = ldltfact(Positive, A, pivot)
    @test_approx_eq full(F) A
    @test d == [1,1]

    A = [1 0; 0 -1]
    F = cholfact(Positive, A, pivot)
    @test_approx_eq full(F) eye(2)
    F, d = ldltfact(Positive, A, pivot)
    @test_approx_eq full(F) eye(2)
    @test d == [1,-1]

    A = [-1 0.5; 0.5 4]
    target = pivot == Val{false} ? [1 -0.5; -0.5 4.5] : [1.125 0.5; 0.5 4]
    dtarget = pivot == Val{false} ? [-1,1] : [1,-1]
    F = cholfact(Positive, A, pivot)
    @test_approx_eq full(F) target
    F = cholfact(Positive, 10*A, pivot)
    @test_approx_eq full(F) 10*target
    F, d = ldltfact(Positive, A, pivot)
    @test_approx_eq full(F) target
    @test d == dtarget

    A = [0 1; 1 0]
    F = cholfact(Positive, A, pivot)
    @test_approx_eq full(F) eye(2)
    F, d = ldltfact(Positive, A, pivot)
    @test_approx_eq full(F) eye(2)
    @test d == [0,0]

    A = rand(201,200); A = A'*A
    F = cholfact(Positive, A, pivot)
    @test_approx_eq full(F) A
    F, d = ldltfact(Positive, A, pivot)
    @test_approx_eq full(F) A
    @test all(d .== 1)
end

A = [1 0; 0 -2]
F = eigfact(Positive, A)
@test_approx_eq full(F) abs(A)
A = [1 0; 0 0]
F = eigfact(Positive, A)
@test_approx_eq full(F) eye(2)

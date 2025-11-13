using spectral1o
using Test
using SpecialFunctions

@testset "chepolsum test" begin
    mathematica_res = -0.707328

    a = [0.1, 0.5, 0.7, 0.5, 0.2]
    a[1] = 2a[1]
    y = 0.2

    julia_res = SpecialFunctions.chepolsum(y, a)*(1 + y)

    @test isapprox(julia_res, mathematica_res; atol=1e-6)
end

using Bess
using Test
using DataFrames, Random

Random.seed!(1234)
df1 = hcat(rand(Float64, (50, 20)), rand([0, 1], (50, 1))) # n>p
df2 = hcat(rand(Float64, (10, 20)), rand([0, 1], (10, 1))) # n<p

@testset "Fit- Forward Step-wise" begin
    @test fit(ModelSelection(), df1) isa Array # Forward, Matrix(n>p), Default
    @test fit(ModelSelection(), df2) isa Array # Forward, Matrix(n<p), Default

    @test fit(ModelSelection("forward", "deviance", "aic"), DataFrame(df1, :auto)) isa Array # Forward, DataFrame(n<p), Main
    @test fit(ModelSelection("forward_stepwise", "deviance", "bic"), DataFrame(df2, :auto)) isa Array # Forward, DataFrame(n<p), Main
end

@testset "Fit-Best Subset" begin
    @test fit(ModelSelection("bess", "deviance", "bic"), df1) isa Array # Best, Matrix(n>p)
    @test fit(ModelSelection("best", "deviance", "aic"), df2) isa Array # Best, Matrix(n<p)

    @test fit(ModelSelection("best_subset", "deviance", "aic"), DataFrame(df1, :auto)) isa Array # Best, DataFrame(n<p)
    @test fit(ModelSelection("best_subset_selection", "deviance", "bic"), DataFrame(df2, :auto)) isa Array # Best, DataFrame(n<p)
end

@testset "Fit-Backward Step-wise" begin
    @test fit(ModelSelection("backward", "deviance", "bic"), df1) isa Array # Backward, Matrix(n>p)
    @test fit(ModelSelection("backward_stepwise", "deviance", "aic"), df2) isa Array # Backward, Matrix(n<p)

    @test fit(ModelSelection("backward", "deviance", "aic"), DataFrame(df1, :auto)) isa Array # Backward, DataFrame(n<p)
    @test fit(ModelSelection("backward_stepwise_selection", "deviance", "bic"), DataFrame(df2, :auto)) isa Array # Backward, DataFrame(n<p)
end
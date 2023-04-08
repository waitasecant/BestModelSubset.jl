using SubsetSelection
using Test
using DataFrames, Random

Random.seed!(1234)
df1 = hcat(rand(Float64, (50, 20)), rand([0, 1], (50, 1))) # n>p
df2 = hcat(rand(Float64, (10, 20)), rand([0, 1], (10, 1))) # n<p

df3 = hcat(rand(Float64, (50, 21))) # n>p
df4 = hcat(rand(Float64, (10, 21))) # n<p

@testset "Fit- Forward Step-wise" begin
    @test fit!(ModelSelection("forward", "deviance", "aic"), DataFrame(df1, :auto)) isa Array # DataFrame(n>p)
    @test fit!(ModelSelection("forward_stepwise", "deviance", "bic"), DataFrame(df2, :auto)) isa Array # DataFrame(n<p)
    @test fit!(ModelSelection("forward", "deviance", "aic"), df1) isa Array # Matrix(n>p)
    @test fit!(ModelSelection("forward_stepwise", "deviance", "bic"), df2) isa Array # Matrix(n<p)

    @test fit!(ModelSelection("forward", "r2", "aic"), DataFrame(df3, :auto)) isa Array # DataFrame(n>p)
    @test fit!(ModelSelection("forward_stepwise", "r2", "bic"), df3) isa Array # DataFrame(n>p)
    @test fit!(ModelSelection("forward", "R2", "adjr2"), DataFrame(df3, :auto)) isa Array # DataFrame(n>p)
    @test fit!(ModelSelection("forward_stepwise", "R2", "aic"), df4) isa Array # Matrix(n<p)
    @test fit!(ModelSelection("forward", "R2", "bic"), DataFrame(df4, :auto)) isa Array # Matrix(n<p)
    @test fit!(ModelSelection("forward_stepwise", "r2", "adjr2"), df4) isa Array # Matrix(n<p)
end

@testset "Fit- Best Subset" begin
    @test fit!(ModelSelection("bess", "deviance", "aic"), DataFrame(df1, :auto)) isa Array # DataFrame(n>p)
    @test fit!(ModelSelection("best", "deviance", "bic"), DataFrame(df2, :auto)) isa Array # DataFrame(n<p)
    @test fit!(ModelSelection("best_subset", "deviance", "aic"), df1) isa Array # Matrix(n>p)
    @test fit!(ModelSelection("best_subset_selection", "deviance", "bic"), df2) isa Array # Matrix(n<p)

    @test fit!(ModelSelection("bess", "r2", "aic"), DataFrame(df3, :auto)) isa Array # DataFrame(n>p)
    @test fit!(ModelSelection("best", "r2", "bic"), df3) isa Array # DataFrame(n>p)
    @test fit!(ModelSelection("best_subset", "R2", "adjr2"), DataFrame(df3, :auto)) isa Array # DataFrame(n>p)
    @test fit!(ModelSelection("bess", "R2", "aic"), df4) isa Array # Matrix(n<p)
    @test fit!(ModelSelection("best", "R2", "bic"), DataFrame(df4, :auto)) isa Array # Matrix(n<p)
    @test fit!(ModelSelection("best_subset", "r2", "adjr2"), df4) isa Array # Matrix(n<p)
end

@testset "Fit- Backward Step-wise" begin
    @test fit!(ModelSelection("backward", "deviance", "aic"), DataFrame(df1, :auto)) isa Array # DataFrame(n>p)
    @test fit!(ModelSelection("backward_stepwise", "deviance", "bic"), DataFrame(df2, :auto)) isa Array # DataFrame(n<p)
    @test fit!(ModelSelection("backward", "deviance", "aic"), df1) isa Array # Matrix(n>p)
    @test fit!(ModelSelection("backward_stepwise", "deviance", "bic"), df2) isa Array # Matrix(n<p)

    @test fit!(ModelSelection("backward", "r2", "aic"), DataFrame(df3, :auto)) isa Array # DataFrame(n>p)
    @test fit!(ModelSelection("backward_stepwise", "r2", "bic"), df3) isa Array # DataFrame(n>p)
    @test fit!(ModelSelection("backward", "R2", "adjr2"), DataFrame(df3, :auto)) isa Array # DataFrame(n>p)
    @test fit!(ModelSelection("backward_stepwise", "R2", "aic"), df4) isa Array # Matrix(n<p)
    @test fit!(ModelSelection("backward", "R2", "bic"), DataFrame(df4, :auto)) isa Array # Matrix(n<p)
    @test fit!(ModelSelection("backward_stepwise", "r2", "adjr2"), df4) isa Array # Matrix(n<p)
end
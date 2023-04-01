using Bess
using Test
using CSV, DataFrames, Random
# using UrlDownload
Random.seed!(1234)

# df1 = DataFrame(urldownload("https://raw.githubusercontent.com/waitasecant/Bess.jl/main/test/new_brca.csv"))

df1 = hcat(rand(Float64, (50, 20)), rand([0, 1], (50, 1))) # n>p
df2 = hcat(rand(Float64, (10, 20)), rand([0, 1], (10, 1))) # n<p

@testset "Fit- Forward Stepwise" begin
    @test fit(ModelSelection(), df1) isa Array # Forward, Matrix(n>p), Default
    @test fit(ModelSelection(), df2) isa Array # Forward, Matrix(n<p), Default

    @test fit(ModelSelection("forward", "deviance", "aic"), DataFrame(df1, :auto)) isa Array # Forward, DataFrame(n<p), Main
    @test fit(ModelSelection("forward", "deviance", "bic"), DataFrame(df2, :auto)) isa Array # Forward, DataFrame(n<p), Main
end

@testset "Fit-Best Subset" begin
    @test fit(ModelSelection("bess", "deviance", "bic"), df1) isa Array # Best, Matrix(n>p)
    @test fit(ModelSelection("bess", "deviance", "aic"), df2) isa Array # Best, Matrix(n<p)

    @test fit(ModelSelection("bess", "deviance", "aic"), DataFrame(df1, :auto)) isa Array # Best, DataFrame(n<p)
    @test fit(ModelSelection("bess", "deviance", "bic"), DataFrame(df2, :auto)) isa Array # Best, DataFrame(n<p)
end
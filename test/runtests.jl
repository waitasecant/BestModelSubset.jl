using Bess
using Test
using CSV, DataFrames, UrlDownload

df = urldownload("https://raw.githubusercontent.com/waitasecant/Bess.jl/main/test/new_brca.csv")

@testset "Best Subset Selection" begin
    @test Bess.best_subset_selection(df) isa Array
end

@testset "Forward Step-wise Selection" begin
    @test Bess.forward_stepwise_selection(df) isa Array
end

@testset "Backward Step-wise Selection" begin
    @test Bess.backward_stepwise_selection(df) isa Array
end
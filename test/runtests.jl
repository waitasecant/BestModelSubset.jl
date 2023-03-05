using Bess
using Test
using CSV, DataFrames, UrlDownload


df = CSV.read("C:\\Users\\Dell\\.julia\\dev\\Bess\\test\\new_brca.csv", DataFrame, normalizenames=true);

@testset "Best Subset Selection" begin
    @test Bess.best_subset_selection(df) isa Array
    @test Bess.best_subset_selection(Array(df)) isa Array
end

@testset "Forward Step-wise Selection" begin
    @test Bess.forward_stepwise_selection(df) isa Array
    @test Bess.forward_stepwise_selection(Array(df)) isa Array
end

@testset "Backward Step-wise Selection" begin
    @test Bess.backward_stepwise_selection(df) isa Array
    @test Bess.backward_stepwise_selection(Array(df)) isa Array
end
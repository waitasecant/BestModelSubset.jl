using Bess
using Test
using CSV, DataFrames

@testset "Best Subset Selection" begin
    @test Bess.best_subset_selection(brca) isa Array
end

@testset "Forward Step-wise Selection" begin
    @test Bess.forward_stepwise_selection(brca) isa Array
end

@testset "Backward Step-wise Selection" begin
    @test Bess.backward_stepwise_selection(brca) isa Array
end
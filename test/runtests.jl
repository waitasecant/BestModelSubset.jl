using Bess
using Test
using CSV, DataFrames
brca = CSV.read("D:\\Julia\\Sem3\\Seminar Report Code\\Logistic Regression\\new_brca.csv", DataFrame, normalizenames=true)

@testset "Basic Test" begin
    @test size(brca)[1] == 510
end

@testset "Best Subset Selection" begin
    @test Bess.best_subset_selection(brca) isa Array
end

@testset "Forward Step-wise Selection" begin
    @test Bess.forward_stepwise_selection(brca) isa Array
end

@testset "Backward Step-wise Selection" begin
    @test Bess.backward_stepwise_selection(brca) isa Array
end
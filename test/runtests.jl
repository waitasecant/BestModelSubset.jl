using Bess
using Test
using CSV, DataFrames

# df1 = DataFrame(urldownload("https://raw.githubusercontent.com/waitasecant/Bess.jl/main/test/new_brca.csv"))
df1 = hcat(rand(Float64, (50, 20)), rand([0, 1], (50, 1)))
df2 = hcat(rand(Float64, (12, 20)), rand([0, 1], (12, 1)))


@testset "Best Subset Selection" begin
    @test Bess.best_subset_selection(df1) isa Array
    @test Bess.best_subset_selection(Array(df1)) isa Array
    @test Bess.best_subset_selection(df2) isa Array
    @test Bess.best_subset_selection(DataFrame(df2, :auto)) isa Array
end

@testset "Forward Step-wise Selection" begin
    @test Bess.forward_stepwise_selection(df1) isa Array
    @test Bess.forward_stepwise_selection(Array(df1)) isa Array
end

@testset "Backward Step-wise Selection" begin
    @test Bess.backward_stepwise_selection(df1) isa Array
    @test Bess.backward_stepwise_selection(Array(df1)) isa Array
    @test Bess.backward_stepwise_selection(df2) isa Array
    @test Bess.backward_stepwise_selection(DataFrame(df2, :auto)) isa Array
end

@testset "Evaluation Selection" begin
    @test Bess.eval_selection(Bess.ModelSelection(), DataFrame(df1, :auto)) isa Array
end
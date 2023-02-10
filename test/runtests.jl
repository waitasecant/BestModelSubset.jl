using Bess
using Test

@testset "Best_Subset" begin

    brca = CSV.read("D:\\Julia\\Sem3\\Seminar Report Code\\Logistic Regression\\new_brca.csv", DataFrame, normalizenames=true)
    @test size(brca)[1] == 510
end
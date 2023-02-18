module Bess
using DataFrames
using CSV
using GLM
using Combinatorics
using MLBase

include("best_subset.jl")
include("forward_stepwise.jl")
include("backward_stepwise.jl")

# struct ModelSelection
#     algorithm::Function
#     param1::Function
#     param2::Function
# end
# typeof(Bess.best_subset_selection)
# model1 = ModelSelection(Bess.best_subset_selection, MLBase.r2, MLBase.aic)
# println(model1)
end
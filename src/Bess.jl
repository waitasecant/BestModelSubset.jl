module Bess
using DataFrames
using CSV
using GLM
using Combinatorics
using MLBase

include("best_subset.jl")
include("forward_stepwise.jl")
include("backward_stepwise.jl")
include("model_selection.jl")

# mutable struct ModelSelection
#     algorithm::Union{Function,Nothing}
#     deviance::Union{Function,Real,Nothing}
#     aic::Union{Function,Real,Nothing}
#     bic::Union{Function,Real,Nothing}
#     param1::Union{Function,Real,Nothing}
#     param2::Union{Function,Real,Nothing}

#     ModelSelection(algorithm, deviance, aic, bic, param1, param2) =
#         typeof(aic) <: Nothing ? new(algorithm, deviance, aic, bic, deviance, bic) : new(algorithm, deviance, aic, bic, deviance, aic)
# end

# function ModelSelection()
#     return ModelSelection(best_subset_selection, MLBase.deviance, MLBase.aic, nothing, nothing, nothing)
# end

# function ModelSelection(algorithm::AbstractString, param1::AbstractString, param2::AbstractString)
#     dict = Dict([
#         "best" => best_subset_selection,
#         "bess" => best_subset_selection,
#         "best_subset_selection" => best_subset_selection,
#         "aic" => MLBase.aic,
#         "bic" => MLBase.bic,
#         "deviance" => MLBase.deviance
#     ])
#     if param2 == "aic"
#         return ModelSelection(values(dict[algorithm]), values(dict[param1]), values(dict[param2]), nothing, nothing, nothing)
#     end
#     if param2 == "bic"
#         return ModelSelection(values(dict[algorithm]), values(dict[param1]), nothing, values(dict[param2]), nothing, nothing)
#     end
# end

# function eval_selection(obj::ModelSelection, data::DataFrame)
#     dev = best_subset_selection(obj, data)
#     final = []
#     for d in dev
#         logreg = glm(Array(data[:, d]), Array(data[:, end]), Binomial(), ProbitLink())
#         push!(final, obj.param2(logreg))
#     end
#     obj.param2 = minimum(final)
#     obj.aic = MLBase.aic(glm(Array(data[:, dev[indexin(minimum(final), final)[1]]]), Array(data[:, end]), Binomial(), ProbitLink()))
#     obj.bic = MLBase.bic(glm(Array(data[:, dev[indexin(minimum(final), final)[1]]]), Array(data[:, end]), Binomial(), ProbitLink()))
#     obj.deviance = MLBase.deviance(glm(Array(data[:, dev[indexin(minimum(final), final)[1]]]), Array(data[:, end]), Binomial(), ProbitLink()))
#     obj.param1 = MLBase.deviance(glm(Array(data[:, dev[indexin(minimum(final), final)[1]]]), Array(data[:, end]), Binomial(), ProbitLink()))
#     return [dev[indexin(minimum(final), final)[1]]]
# end

end

module Bess

using DataFrames
using CSV
using GLM
using Combinatorics
using MLBase

export ModelSelection
export fit
export best_subset
export forward_stepwise
export backward_stepwise

"""
Mutable struct taking input as algorithm, params with an inner constructor.
The inner constructor check if aic is given as input for param2 and creates a new ModelSelection
object with param2 set as bic else aic.
"""
mutable struct ModelSelection
    algorithm::Union{Function,Nothing}
    deviance::Union{Function,Real,Nothing}
    aic::Union{Function,Real,Nothing}
    bic::Union{Function,Real,Nothing}
    param1::Union{Function,Real,Nothing}
    param2::Union{Function,Real,Nothing}

    ModelSelection(algorithm, deviance, aic, bic, param1, param2) =
        typeof(aic) <: Nothing ? new(algorithm, deviance, aic, bic, deviance, bic) :
        new(algorithm, deviance, aic, bic, deviance, aic)
end

# An outer (default) constructor.
# The outer (default) constructor sets algorithm to be Forward Stepwise, param1 to be deviance and param2 to be aic.
# The inner constructor comes into play and set param2 as aic.
"""
    ModelSelection()

Creates ModelSelection(forward_stepwise, MLBase.deviance, MLBase.aic, nothing, nothing, nothing)
"""
function ModelSelection()
    return ModelSelection(forward_stepwise, MLBase.deviance, MLBase.aic, nothing, nothing, nothing)
end


# An outer (main) constructor.
# The outer (main) constructor inputs algorithm, param1 and param2 as strings
# and then assigns corresponding functions using the dictionary and creates a ModelSelection object.
"""
    ModelSelection(algorithm::AbstractString, param1::AbstractString, param2::AbstractString)

Return a ModelSelection object
"""
function ModelSelection(algorithm::AbstractString, param1::AbstractString, param2::AbstractString)
    dict = Dict([
        "best" => best_subset,
        "bess" => best_subset,
        "best_subset" => best_subset,
        "best_subset_selection" => best_subset,
        "forward" => forward_stepwise,
        "forward_stepwise" => forward_stepwise,
        "forward_stepwise_selection" => forward_stepwise,
        "backward" => backward_stepwise,
        "backward_stepwise" => backward_stepwise,
        "backward_stepwise_selection" => backward_stepwise,
        "aic" => MLBase.aic,
        "bic" => MLBase.bic,
        "deviance" => MLBase.deviance
    ])
    if param2 == "aic"
        return ModelSelection(values(dict[lowercase(algorithm)]), values(dict[lowercase(param1)]),
            values(dict[lowercase(param2)]), nothing, nothing, nothing)
    end
    if param2 == "bic"
        return ModelSelection(values(dict[lowercase(algorithm)]), values(dict[lowercase(param1)]),
            nothing, values(dict[lowercase(param2)]), nothing, nothing)
    end
end


# The fit function inputs the ModelSelection object created and data to be either A DataFrame or a matrix.
# and updates values into 
"""
    fit(obj::ModelSelection, data::Union{DataFrame,AbstractMatrix{<:Real}})

Fits the data to the inputed algorithm and parameter, returning a vector of vector containing
indexes of the columns corresponding to least value of param 2.
"""
function fit(obj::ModelSelection, data::Union{DataFrame,AbstractMatrix{<:Real}})
    dev = obj.algorithm(obj, data)
    final = []
    for d in dev
        logreg = glm(Array(data[:, d]), Array(data[:, end]), Binomial(), ProbitLink())
        push!(final, obj.param2(logreg))
    end
    obj.param2 = minimum(final)
    obj.aic = MLBase.aic(glm(Array(data[:, dev[indexin(minimum(final), final)[1]]]),
        Array(data[:, end]), Binomial(), ProbitLink()))
    obj.bic = MLBase.bic(glm(Array(data[:, dev[indexin(minimum(final), final)[1]]]),
        Array(data[:, end]), Binomial(), ProbitLink()))
    obj.deviance = MLBase.deviance(glm(Array(data[:, dev[indexin(minimum(final), final)[1]]]),
        Array(data[:, end]), Binomial(), ProbitLink()))
    obj.param1 = obj.deviance
    return [dev[indexin(minimum(final), final)[1]]]
end

# Forward Step-wise Selection
"""
forward_stepwise(obj::ModelSelection, df::DataFrame)

Executes the forward step-wise selection algorithm returning a vector of vectors containing
indexes of columns corresponding to least value of param1.
"""
function forward_stepwise(obj::ModelSelection, df::DataFrame)
    dev = []
    comb = collect(combinations(1:length(names(df))-1, 1))
    for num in 1:length(names(df))-1
        val = []
        for j in comb
            logreg = glm(Array(df[:, j]), Array(df[:, names(df)[end]]), Binomial(), ProbitLink())
            push!(val, obj.param1(logreg))
        end
        push!(dev, comb[indexin(minimum(val), val)])
        comb = collect(combinations(1:length(names(df))-1, num + 1))
        l = []
        for j in 1:length(comb)
            a = true
            for i in dev[end][1]
                a = a & (i in comb[j])
            end
            push!(l, a)
        end
        comb = comb[[i for i in l]]
    end
    return [dev[i][1] for i in 1:length(names(df))-1]
end

# Forward Step-wise Selection
"""
forward_stepwise(obj::ModelSelection, df::AbstractMatrix{<:Real})

Executes the forward step-wise selection algorithm returning a vector of vectors containing
indexes of columns corresponding to least value of param1.
"""
function forward_stepwise(obj::ModelSelection, df::AbstractMatrix{<:Real})
    df = DataFrame(df, :auto)
    forward_stepwise(obj, df)
end

# Best Subset Selection
"""
best_subset(obj::ModelSelection, df::DataFrame)

Executes the best subset selection algorithm returning a vector of vectors containing
indexes of columns corresponding to least value of param1.
"""
function best_subset(obj::ModelSelection, df::DataFrame)
    if size(df)[1] > size(df)[2]
        dev = []
        for num in 1:length(names(df))-1
            val = []
            for i in collect(combinations(1:length(names(df))-1, num))
                logreg = glm(Array(df[:, i]), Array(df[:, end]), Binomial(), ProbitLink())
                push!(val, obj.param1(logreg))
            end
            push!(dev, collect(combinations(1:length(names(df))-1, num))[indexin(minimum(val), val)])
        end
        return [dev[i][1] for i in 1:length(names(df))-1]
    else
        forward_stepwise(obj, df)
    end
end

# Best Subset Selection
"""
best_subset(obj::ModelSelection, df::AbstractMatrix{<:Real})

Executes the best subset selection algorithm returning a vector of vectors containing
indexes of columns corresponding to least value of param1.
"""
function best_subset(obj::ModelSelection, df::AbstractMatrix{<:Real})
    df = DataFrame(df, :auto)
    if size(df)[1] > size(df)[2]
        best_subset(obj, df)
    else
        forward_stepwise(obj, df)
    end
end

# Backward Step-wise Selection
"""
backward_stepwise(obj::ModelSelection, df::DataFrame)

Executes the backward step-wise selection algorithm returning a vector of vectors containing
indexes of columns corresponding to least value of param1.
"""
function backward_stepwise(obj::ModelSelection, df::DataFrame)
    if size(df)[1] > size(df)[2]
        dev = [[s for s in 1:length(names(df))-1]]
        comb = collect(combinations(dev[1], length(names(df)) - 2))
        while length(dev[end]) != 1
            val = []
            for j in comb
                logreg = glm(Array(df[:, j]), Array(df[:, names(df)[end]]), Binomial(), ProbitLink())
                push!(val, obj.param1(logreg))
            end
            push!(dev, comb[indexin(minimum(val), val)][1])
            comb = collect(combinations(dev[end], length(dev[end]) - 1))
        end
        return dev
    else
        forward_stepwise(obj, df)
    end
end

# Backward Step-wise Selection
"""
backward_stepwise(obj::ModelSelection, df::DataFrame)

Executes the backward step-wise selection algorithm returning a vector of vectors containing
indexes of columns corresponding to least value of param1.
"""
function backward_stepwise(obj::ModelSelection, df::AbstractMatrix{<:Real})
    df = DataFrame(df, :auto)
    if size(df)[1] > size(df)[2]
        backward_stepwise(obj, df)
    else
        forward_stepwise(obj, df)
    end
end

end
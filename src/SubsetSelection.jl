module SubsetSelection

using DataFrames: DataFrame
using GLM
using Combinatorics: combinations
using MLBase: deviance, r2, adjr2, aic, bic
using Suppressor: @suppress

export ModelSelection
export fit!
export best_subset
export forward_stepwise
export backward_stepwise

"""
    ModelSelection(algorithm::AbstractString, param1::AbstractString, param2::AbstractString)

Returns a ModelSelection object
    
    ModelSelection(algorithm::Union{Function,Nothing}
    deviance::Union{Function,Real,Nothing}
    r2::Union{Function,Real,Nothing}
    adjr2::Union{Function,Real,Nothing}
    aic::Union{Function,Real,Nothing}
    bic::Union{Function,Real,Nothing}
    param1::Union{Function,Real,Nothing}
    param2::Union{Function,Real,Nothing})

For example:

    ModelSelection("bess", "r2", "adjr2") returns
    ModelSelection(SubsetSelection.best_subset, nothing, StatsAPI.r2, StatsAPI.adjr2,
                   nothing, nothing, StatsAPI.r2, StatsAPI.adjr2)

    ModelSelection("forward","deviance","aic")
    ModelSelection(SubsetSelection.forward_stepwise, StatsAPI.deviance, nothing, nothing,
                   StatsAPI.aic, nothing, StatsAPI.deviance, StatsAPI.aic)
"""
mutable struct ModelSelection
    algorithm::Union{Function,Nothing}
    deviance::Union{Function,Real,Nothing}
    r2::Union{Function,Real,Nothing}
    adjr2::Union{Function,Real,Nothing}
    aic::Union{Function,Real,Nothing}
    bic::Union{Function,Real,Nothing}
    param1::Union{Function,Real,Nothing}
    param2::Union{Function,Real,Nothing}

    function ModelSelection(algorithm, deviance, r2, adjr2, aic, bic, param1, param2)
        if typeof(r2) <: Nothing
            if typeof(bic) <: Nothing
                return new(algorithm, deviance, r2, adjr2, aic, bic, deviance, aic)
            else
                return new(algorithm, deviance, r2, adjr2, aic, bic, deviance, bic)
            end
        end

        if typeof(deviance) <: Nothing
            if (typeof(aic) <: Nothing) & (typeof(bic) <: Nothing)
                return new(algorithm, deviance, r2, adjr2, aic, bic, r2, adjr2)
            end
            if (typeof(adjr2) <: Nothing) & (typeof(bic) <: Nothing)
                return new(algorithm, deviance, r2, adjr2, aic, bic, r2, aic)
            end
            if (typeof(adjr2) <: Nothing) & (typeof(aic) <: Nothing)
                return new(algorithm, deviance, r2, adjr2, aic, bic, r2, bic)
            end
        end
    end
end

function ModelSelection(algorithm::AbstractString, param1::AbstractString, param2::AbstractString)
    dict = Dict([
        "best" => best_subset, "bess" => best_subset,
        "best_subset" => best_subset, "best_subset_selection" => best_subset,
        "forward" => forward_stepwise, "forward_stepwise" => forward_stepwise,
        "forward_stepwise_selection" => forward_stepwise, "backward" => backward_stepwise,
        "backward_stepwise" => backward_stepwise, "backward_stepwise_selection" => backward_stepwise,
        "deviance" => deviance, "r2" => r2, "adjr2" => adjr2, "aic" => aic, "bic" => bic])

    if (lowercase(param1) == "deviance") & (lowercase(param2) == "aic")
        return ModelSelection(
            values(dict[lowercase(algorithm)]),
            values(dict[lowercase(param1)]),
            nothing,
            nothing,
            values(dict[lowercase(param2)]),
            nothing,
            nothing,
            nothing)
    end
    if (lowercase(param1) == "deviance") & (lowercase(param2) == "bic")
        return ModelSelection(
            values(dict[lowercase(algorithm)]),
            values(dict[lowercase(param1)]),
            nothing,
            nothing,
            nothing,
            values(dict[lowercase(param2)]),
            nothing,
            nothing)
    end
    if (lowercase(param1) == "r2") & (lowercase(param2) == "adjr2")
        return ModelSelection(
            values(dict[lowercase(algorithm)]),
            nothing,
            values(dict[lowercase(param1)]),
            values(dict[lowercase(param2)]),
            nothing,
            nothing,
            nothing,
            nothing)
    end
    if (lowercase(param1) == "r2") & (lowercase(param2) == "aic")
        return ModelSelection(
            values(dict[lowercase(algorithm)]),
            nothing,
            values(dict[lowercase(param1)]),
            nothing,
            values(dict[lowercase(param2)]),
            nothing,
            nothing,
            nothing)
    end
    if (lowercase(param1) == "r2") & (lowercase(param2) == "bic")
        return ModelSelection(
            values(dict[lowercase(algorithm)]),
            nothing,
            values(dict[lowercase(param1)]),
            nothing,
            nothing,
            values(dict[lowercase(param2)]),
            nothing,
            nothing)
    end
end

"""
    fit!(obj::ModelSelection, data::Union{DataFrame,AbstractMatrix{<:Real}}) -> Vector{Vector{T}}

Fit the data to the ModelSelection object.
"""
function fit!(obj::ModelSelection, data::Union{DataFrame,AbstractMatrix{<:Real}})
    @suppress begin
        if Set(Array(data[:, end])) == Set([0.0, 1.0])
            dev = obj.algorithm(obj, data)
            final = []
            for d in dev
                logreg = glm(Array(data[:, d]), Array(data[:, end]), Binomial(), ProbitLink())
                push!(final, obj.param2(logreg))
            end
            obj.param2 = minimum(final)
            obj.r2 = nothing
            obj.adjr2 = nothing
            obj.deviance = deviance(glm(Array(data[:, dev[indexin(minimum(final), final)[1]]]),
                Array(data[:, end]), Binomial(), ProbitLink()))
            obj.param1 = obj.deviance
            obj.aic = aic(glm(Array(data[:, dev[indexin(minimum(final), final)[1]]]),
                Array(data[:, end]), Binomial(), ProbitLink()))
            obj.bic = bic(glm(Array(data[:, dev[indexin(minimum(final), final)[1]]]),
                Array(data[:, end]), Binomial(), ProbitLink()))
        else
            dev = obj.algorithm(obj, data)
            final = []
            for d in dev
                logreg = lm(Array(data[:, d]), Array(data[:, end]))
                push!(final, obj.param2(logreg))
            end
            obj.param2 = minimum(final)
            obj.r2 = r2(lm(Array(data[:, dev[indexin(minimum(final), final)[1]]]),
                Array(data[:, end])))
            obj.adjr2 = adjr2(lm(Array(data[:, dev[indexin(minimum(final), final)[1]]]),
                Array(data[:, end])))
            obj.param1 = obj.r2
            obj.aic = aic(lm(Array(data[:, dev[indexin(minimum(final), final)[1]]]),
                Array(data[:, end])))
            obj.bic = bic(lm(Array(data[:, dev[indexin(minimum(final), final)[1]]]),
                Array(data[:, end])))
        end
        return [dev[indexin(minimum(final), final)[1]]]
    end
end

"""
    forward_stepwise(obj::ModelSelection, df::DataFrame) -> Vector{Vector{T}}

Executes the forward step-wise selection algorithm.
"""
function forward_stepwise(obj::ModelSelection, df::DataFrame)
    @suppress begin
        if Set(Array(df[:, end])) == Set([0.0, 1.0])
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
        else
            dev = []
            comb = collect(combinations(1:length(names(df))-1, 1))
            for num in 1:length(names(df))-1
                val = []
                for j in comb
                    logreg = lm(Array(df[:, j]), Array(df[:, names(df)[end]]))
                    push!(val, obj.param1(logreg))
                end
                push!(dev, comb[indexin(maximum(val), val)])
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
    end
end

"""
    forward_stepwise(obj::ModelSelection, df::AbstractMatrix{<:Real}) -> Vector{Vector{T}}

Executes the forward step-wise selection algorithm.
"""
function forward_stepwise(obj::ModelSelection, df::AbstractMatrix{<:Real})
    df = DataFrame(df, :auto)
    forward_stepwise(obj, df)
end

"""
    best_subset(obj::ModelSelection, df::DataFrame) -> Vector{Vector{T}}

Executes the best subset selection algorithm.
"""
function best_subset(obj::ModelSelection, df::DataFrame)
    @suppress begin
        if size(df)[1] > size(df)[2]
            if Set(Array(df[:, end])) == Set([0.0, 1.0])
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
                dev = []
                for num in 1:length(names(df))-1
                    val = []
                    for i in collect(combinations(1:length(names(df))-1, num))
                        logreg = lm(Array(df[:, i]), Array(df[:, end]))
                        push!(val, obj.param1(logreg))
                    end
                    push!(dev, collect(combinations(1:length(names(df))-1, num))[indexin(maximum(val), val)])
                end
                return [dev[i][1] for i in 1:length(names(df))-1]
            end
        else
            forward_stepwise(obj, df)
        end
    end
end

"""
    best_subset(obj::ModelSelection, df::AbstractMatrix{<:Real}) -> Vector{Vector{T}}

    Executes the best subset selection algorithm.
"""
function best_subset(obj::ModelSelection, df::AbstractMatrix{<:Real})
    df = DataFrame(df, :auto)
    best_subset(obj, df)
end

"""
    backward_stepwise(obj::ModelSelection, df::DataFrame) -> Vector{Vector{T}}

Executes the backward step-wise selection algorithm.
"""
function backward_stepwise(obj::ModelSelection, df::DataFrame)
    @suppress begin
        if size(df)[1] > size(df)[2]
            if Set(Array(df[:, end])) == Set([0.0, 1.0])
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
                dev = [[s for s in 1:length(names(df))-1]]
                comb = collect(combinations(dev[1], length(names(df)) - 2))
                while length(dev[end]) != 1
                    val = []
                    for j in comb
                        logreg = lm(Array(df[:, j]), Array(df[:, names(df)[end]]))
                        push!(val, obj.param1(logreg))
                    end
                    push!(dev, comb[indexin(maximum(val), val)][1])
                    comb = collect(combinations(dev[end], length(dev[end]) - 1))
                end
                return dev
            end
        else
            forward_stepwise(obj, df)
        end
    end
end

"""
    backward_stepwise(obj::ModelSelection, df::DataFrame) -> Vector{Vector{T}}

Executes the backward step-wise selection algorithm.
"""
function backward_stepwise(obj::ModelSelection, df::AbstractMatrix{<:Real})
    df = DataFrame(df, :auto)
    backward_stepwise(obj, df)
end

end
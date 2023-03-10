function best_subset_selection(df::DataFrame)
    if size(df)[1] > size(df)[2]
        dev = []
        for num in 1:length(names(df))-1
            val = []
            for i in collect(combinations(1:length(names(df))-1, num))
                logreg = glm(Array(df[:, i]), Array(df[:, names(df)[end]]), Binomial(), ProbitLink())
                push!(val, deviance(logreg))
            end
            push!(dev, collect(combinations(1:length(names(df))-1, num))[indexin(minimum(val), val)])
        end
        return [dev[i][1] for i in 1:length(names(df))-1]
    else
        forward_stepwise_selection(df::DataFrame)
    end
end

function best_subset_selection(df::Matrix{Float64})
    if size(df)[1] > size(df)[2]
        dev = []
        df = DataFrame(df, :auto)
        for num in 1:length(names(df))-1
            val = []
            for i in collect(combinations(1:length(names(df))-1, num))
                logreg = glm(Array(df[:, i]), Array(df[:, names(df)[end]]), Binomial(), ProbitLink())
                push!(val, deviance(logreg))
            end
            push!(dev, collect(combinations(1:length(names(df))-1, num))[indexin(minimum(val), val)])
        end
        return [dev[i][1] for i in 1:length(names(df))-1]
    else
        forward_stepwise_selection(df::Matrix{Float64})
    end
end
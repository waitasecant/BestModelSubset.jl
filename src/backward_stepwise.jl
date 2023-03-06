function backward_stepwise_selection(df::DataFrame)
    if size(df)[1] > size(df)[2]
        dev = [[s for s in 1:length(names(df))-1]]
        comb = collect(combinations(dev[1], length(names(df)) - 2))
        while length(dev[end]) != 1
            val = []
            for j in comb
                logreg = glm(Array(df[:, j]), Array(df[:, names(df)[end]]), Binomial(), ProbitLink())
                push!(val, deviance(logreg))
            end
            push!(dev, comb[indexin(minimum(val), val)][1])
            comb = collect(combinations(dev[end], length(dev[end]) - 1))
        end
        return dev
    else
        forward_stepwise_selection(df::DataFrame)
    end
end

function backward_stepwise_selection(df::Matrix{Float64})
    if size(df)[1] > size(df)[2]
        df = DataFrame(df, :auto)
        dev = [[s for s in 1:length(names(df))-1]]
        comb = collect(combinations(dev[1], length(names(df)) - 2))
        while length(dev[end]) != 1
            val = []
            for j in comb
                logreg = glm(Array(df[:, j]), Array(df[:, names(df)[end]]), Binomial(), ProbitLink())
                push!(val, deviance(logreg))
            end
            push!(dev, comb[indexin(minimum(val), val)][1])
            comb = collect(combinations(dev[end], length(dev[end]) - 1))
        end
        return dev
    else
        forward_stepwise_selection(df::Matrix{Float64})
    end
end
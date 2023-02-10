function backward_stepwise_selection(df)
    dev = [[s for s in 1:length(names(df))-1]]
    comb = collect(combinations(dev[1], length(names(df)) - 2))
    while length(dev[end]) != 1
        val = []
        for j in comb
            logreg = glm(Array(df[:, j]), Array(df[:, 15]), Binomial(), ProbitLink())
            push!(val, MLBase.deviance(logreg))
        end
        push!(dev, comb[indexin(minimum(val), val)][1])
        comb = collect(combinations(dev[end], length(dev[end]) - 1))
    end
    return dev
end
function best_subset_selection(df)
    dev = []
    for num in 1:length(names(df))-1
        val = []
        for i in collect(combinations(1:length(names(df))-1, num))
            logreg = glm(Array(df[:, i]), Array(df[:, 15]), Binomial(), ProbitLink())
            push!(val, MLBase.deviance(logreg))
        end
        push!(dev, collect(combinations(1:length(names(df))-1, num))[indexin(minimum(val), val)])
    end
    return [dev[i][1] for i in 1:length(names(df))-1]
end
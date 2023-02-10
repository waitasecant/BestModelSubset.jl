function forward_stepwise_selection(df)
    dev = []
    comb = collect(combinations(1:length(names(df))-1, 1))
    for num in 1:length(names(df))-1
        val = []
        for j in comb
            logreg = glm(Array(df[:, j]), Array(df[:, 15]), Binomial(), ProbitLink())
            push!(val, MLBase.deviance(logreg))
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
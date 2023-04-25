# BestModelSubset.jl
*A julia package to implement model selection algorithms on basic regression models.*

[![Build Status](https://ci.appveyor.com/api/projects/status/github/waitasecant/BestModelSubset.jl?svg=true)](https://ci.appveyor.com/project/waitasecant/bestmodelsubset-jl)
[![Coverage](https://codecov.io/gh/waitasecant/BestModelSubset.jl/branch/main/graph/badge.svg?token=CWQH7S8IGZ)](https://codecov.io/gh/waitasecant/BestModelSubset.jl)
[![License](https://img.shields.io/github/license/waitasecant/BestModelSubset.jl)](LICENSE)

## Installation

You can install BestModelSubset.jl using Julia's package manager
```julia
julia> using Pkg; Pkg.add(url="https://github.com/waitasecant/BestModelSubset.jl.git")
```

## Example 1

Instantiate a `ModelSelection` object
```julia
# To execute forward step-wise selection with primary parameter to be R-squared score  
# and secondary parameter to be aic.
julia> obj = ModelSelection("forward", "r2", "aic")
ModelSelection(BestModelSubset.forward_stepwise, nothing, StatsAPI.r2, nothing,
               StatsAPI.aic, nothing, StatsAPI.r2, StatsAPI.aic)
```
Fit the `ModelSelection`object to the data
```julia
# The fit! function updates the fields of the `ModelSelection` object.
julia> Random.seed!(123); df = hcat(rand(Float64, (50, 21))); # 50*21 Matrix

julia> fit!(obj, df)
1-element Vector{Vector{Int64}}:
 [4, 5, 6, 16, 17, 18, 20]
```
Access various statistics like r2, adjr2, aic and bic for the selected model
```julia
julia> obj.r2
0.8174272341858757

julia> obj.aic
22.84823396671912
```

## Example 2

Instantiate a `ModelSelection` object
```julia
# To execute best subset selection with primary parameter to be deviance  
# and secondary parameter to be bic.
julia> obj = ModelSelection("best", "deviance", "bic")
ModelSelection(BestModelSubset.best_subset, StatsAPI.deviance, nothing, nothing,
               nothing, StatsAPI.bic, StatsAPI.deviance, StatsAPI.bic)
```
Fit the `ModelSelection`object to the data
```julia
# The fit! function updates the fields of the `ModelSelection` object.
julia> Random.seed!(123); df = hcat(rand(Float64, (50, 20)), rand([0, 1], (50, 1))); # 50*21 Matrix

julia> fit!(obj, df)
1-element Vector{Vector{Int64}}:
 [8, 13]
```
Access various statistics like r2, adjr2, aic and bic for the selected model
```julia
julia> obj.deviance
61.39326332090434

julia> obj.bic
69.2173093317606
```

# SubsetSelection.jl
*A julia package to implement model selection algorithms on basic regression models.*

[![Build Status](https://ci.appveyor.com/api/projects/status/github/waitasecant/SubsetSelection.jl?svg=true)](https://ci.appveyor.com/project/waitasecant/subsetselection-jl)
[![Coverage](https://codecov.io/gh/waitasecant/SubsetSelection.jl/branch/main/graph/badge.svg?token=CWQH7S8IGZ)](https://codecov.io/gh/waitasecant/SubsetSelection.jl)
[![License](https://img.shields.io/github/license/waitasecant/SubsetSelection.jl)](LICENSE)

## Installation

You can install SubsetSelection.jl using Julia's package manager
```julia-repl
julia> using Pkg; Pkg.add("SubsetSelection.jl")
```
## Example

Instantiate a `ModelSelection` object
```julia-repl
# To execute best subset selection with primary parameter to be R-squared score  
# and secondary parameter to be aic.
julia> obj = ModelSelection("bess", "r2", "aic")
ModelSelection(SubsetSelection.best_subset, nothing, StatsAPI.r2, nothing, StatsAPI.aic,
               nothing, StatsAPI.r2, StatsAPI.aic)
```
Fit the `ModelSelection`object to the data
```julia-repl
# The fit! function updates the fields of the `ModelSelection` object.
julia> Random.seed!(123); df = hcat(rand(Float64, (50, 21))); # 50*21 Matrix

julia> fit!(obj, df)
1-element Vector{Vector{Int64}}:
 [5, 6, 16, 17, 18, 20]
```
Access various statistics like r2, adjr2, aic and bic for the selected model
```julia-repl
julia> obj.r2
0.8161760683631274

julia> obj.aic
21.189713760250477
```

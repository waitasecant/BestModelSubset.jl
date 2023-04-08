# Bess.jl
*A julia package to implement model selection algorithms on basic regression models.*

[![Build Status](https://ci.appveyor.com/api/projects/status/github/waitasecant/Bess.jl?svg=true)](https://ci.appveyor.com/project/waitasecant/Bess-jl)
[![Coverage](https://codecov.io/gh/waitasecant/Bess.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/waitasecant/Bess.jl)
[![License](https://img.shields.io/github/license/waitasecant/Bess.jl)](LICENSE)

## Installation

You can install Bess.jl using Julia's package manager
```julia-repl
julia> using Pkg; Pkg.add("Bess")
```

Bring `Bess`'s exported items into the namespace by
```julia-repl
julia> using Bess
```

## Example

Instantiate a `ModelSelection` object
```julia-repl
# To execute best subset selection with primary parameter to be $R^2$ and secondary parameter to be aic.
julia> obj = ModelSelection("bess", "r2", "aic")
ModelSelection(Bess.best_subset, nothing, StatsAPI.r2, StatsAPI.adjr2, nothing, nothing, StatsAPI.r2, StatsAPI.adjr2)
```
Fit the `ModelSelection`object to the data
```julia-repl
# The fit! function updates the fields of the `ModelSelection` object.
julia> Random.seed!(123); df = hcat(rand(Float64, (50, 21))); # 50\times21 Matrix
julia> fit!(obj, df)
```
Access various statistics like aic, bic for the selected model
```julia-repl
julia> obj.aic
40.51622767157482
```

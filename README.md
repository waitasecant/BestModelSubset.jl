# Bess.jl
*A julia package to implement model selection algorithms on basic regression models.*

[![Build Status](https://ci.appveyor.com/api/projects/status/github/waitasecant/Bess.jl?svg=true)](https://ci.appveyor.com/project/waitasecant/Bess-jl)
[![Coverage](https://codecov.io/gh/waitasecant/Bess.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/waitasecant/Bess.jl)
[![License](https://img.shields.io/github/license/waitasecant/Bess.jl)](LICENSE)

## Installation
Installation is straightforward: enter Pkg mode by hitting `]`, and then run
```julia-repl
(v1.8) pkg> add Bess
```

## Basic Examples
Bring `Bess`'s exported items into the namespace by running
```julia-repl
using Bess
```

### ModelSelection
Instantiate a defaut `ModelSelection` object:
```julia-repl
obj = ModelSelection()
```

Fit the object to the data:
```julia-repl
fit(obj, hcat(rand(Float64, (50, 20)), rand([0, 1], (50, 1))))
```
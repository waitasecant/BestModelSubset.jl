# Bess.jl
*A julia package to implement model selection algorithms on basic regression models.*

[![Build Status](https://ci.appveyor.com/api/projects/status/github/waitasecant/Bess.jl?svg=true)](https://ci.appveyor.com/project/waitasecant/Bess-jl)
[![Coverage](https://codecov.io/gh/waitasecant/Bess.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/waitasecant/Bess.jl)
[![License](https://img.shields.io/github/license/waitasecant/Bess.jl)](LICENSE)

## Installation
Installation is straightforward:
```julia-repl
Pkg.add("Bess")
```
then
Bring `Bess`'s exported items into the namespace by running
```julia-repl
using Bess
```

### ModelSelection
Instantiate a `ModelSelection` object by running, say:
```julia-repl
obj = ModelSelection("bess", "r2", "adjr2")
```

### Fit!
Fit the object to the data:
```julia-repl
fit!(obj, hcat(rand(Float64, (50, 21))))
```

Access various statistics like aic, bic by running:
```julia-repl
obj.aic
```
using Documenter, SPMR

makedocs(modules = [SPMR],
         sitename = "SPMR",
         authors = "Nicholas Hu",
         pages = ["Home" => "index.md",
                  "Solvers" => "solvers.md",
                  "Saddle-point Matrices" => "matrices.md"
                 ]
         )

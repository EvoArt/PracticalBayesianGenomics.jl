using Documenter
using PracticalBayesianGenomics

DocMeta.setdocmeta!(PracticalBayesianGenomics, :DocTestSetup,
                   :(using PracticalBayesianGenomics); recursive=true)

makedocs(;
    modules=[PracticalBayesianGenomics],
    checkdocs=:exports,
    sitename="PracticalBayesianGenomics.jl",
    repo="https://github.com/EvoArt/PracticalBayesianGenomics.jl/blob/{commit}{path}#{line}",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://EvoArt.github.io/PracticalBayesianGenomics.jl",
        repolink="https://github.com/EvoArt/PracticalBayesianGenomics.jl",
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/EvoArt/PracticalBayesianGenomics.jl")

using CFTimeSchemes
using Documenter

DocMeta.setdocmeta!(CFTimeSchemes, :DocTestSetup, :(using CFTimeSchemes); recursive=true)

makedocs(;
    modules=[CFTimeSchemes],
    authors="The ClimFlows contributors",
    sitename="CFTimeSchemes.jl",
    format=Documenter.HTML(;
        canonical="https://ClimFlows.github.io/CFTimeSchemes.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ClimFlows/CFTimeSchemes.jl",
    devbranch="main",
)

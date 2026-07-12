using Documenter
using Mimosa

makedocs(;
    sitename="Mimosa.jl",
    authors="Mimosa contributors",
    format=Documenter.HTML(;
        canonical="https://mimosajl.readthedocs.io/en/stable/",
        prettyurls=get(ENV, "CI", nothing) == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Quick Start" => "quickstart.md",
        "Julia API" => "api.md",
        "CLI" => "cli.md",
        "Supported Models" => "models.md",
        "Data Layout" => "data_layout.md",
        "Numerical Compatibility" => "numerical_compatibility.md",
        "Reproducibility" => "reproducibility.md",
        "Storage Format" => "storage.md",
        "Security" => "security.md",
        "Python Migration" => "migration.md",
        "Extending Mimosa" => "extending.md",
        "MotifHORDE Contract" => "downstream_contract.md",
        "Architecture" => "architecture.md",
    ],
    warnonly=true,
)

deploydocs(; repo="github.com/mimosa-jl/Mimosa.jl.git", devbranch="main", push_preview=true)

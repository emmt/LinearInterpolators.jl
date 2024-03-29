using Documenter

push!(LOAD_PATH, "../src/")
using LinearInterpolators

DEPLOYDOCS = (get(ENV, "CI", nothing) == "true")

makedocs(
    sitename = "Linear interpolators for Julia",
    format = Documenter.HTML(
        prettyurls = DEPLOYDOCS,
    ),
    authors = "Éric Thiébaut and contributors",
    pages = ["index.md",
             "install.md",
             "interpolation.md",
             "library.md",
             "notes.md"]
)

if DEPLOYDOCS
    deploydocs(
        repo = "github.com/emmt/LinearInterpolators.jl.git",
    )
end

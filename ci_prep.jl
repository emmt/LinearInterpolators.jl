# Small script to prepare CI testing (see https://github.com/HolyLab/HolyLabRegistry
# and https://github.com/HolyLab/ImagineInterface.jl).
if VERSION ≥ v"0.7.0"
    using Pkg
end
if VERSION < v"1.1.0"
    using LibGit2
    user_regs = joinpath(DEPOT_PATH[1],"registries")
    mkpath(user_regs)
    Base.shred!(LibGit2.CachedCredentials()) do creds
        for (reg, url) in (
            "General" => "https://github.com/JuliaRegistries/General.git",
            "EmmtRegistry" => "https://github.com/emmt/EmmtRegistry")
            path = joinpath(user_regs, reg);
            LibGit2.with(Pkg.GitTools.clone(
                url, path;
                header = "registry $reg from $(repr(url))",
                credentials = creds)) do repo end
        end
    end
else
    Pkg.Registry.add("General")
    Pkg.Registry.add(RegistrySpec(url="https://github.com/emmt/EmmtRegistry"))
end

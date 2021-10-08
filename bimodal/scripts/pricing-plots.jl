using DataFrames
using CSV
using StatsPlots

things = [
    "baseline",
    "a2-mean-shift",
    "a2-meanvar-shift",
    "more-corr-meanvarshift",
]
sort!(things)

function parse_array(val, nrow, ncol)
    ret = zeros(nrow, ncol)
    rows = split(val, ";")
    
    for i in 1:nrow
        rowvals = filter!(m -> length(m) > 0, split(rows[i], ' '))
        parsed = parse.(Float64, rowvals)
        for j in 1:ncol
            ret[i,j] = parsed[j]
        end
    end

    return ret
end

function parse_vec(z)
    z = map(x -> replace(x, "[" => ""), z)
    z = map(x -> replace(x, "]" => ""), z)

    if ';' in z[1]
        nrows = count(==(';'), z[1]) + 1
        fir = split(z[1], ';')
        ncols = count(==(' '), fir[1]) + 1

        return parse_array.(z, nrows, ncols)
    else
    end
    z = map(x -> parse.(Float64, split(x, ",")), z)
end

dfs = []

for thing in things
    df = DataFrame(CSV.File("data/individuals/$thing.csv"))
    unique!(df, [:payoff])
    select!(df, [:A, :B, :C, :price, :payoff])

    for col in [:A, :B, :C, :price, :payoff]
        df[!, col] = parse_vec(df[:, col])
    end
    df[!, :economy] .= thing
    push!(dfs, df)
end

data = vcat(dfs...)
titles = reshape(things, (1, 4))

!ispath("plots/pricing") && mkpath("plots/pricing")
adpi = 400
savefig(density(map(x -> x[1], data.A), group = data.economy, dpi=adpi, layout=4, title=titles, legend=false), "plots/pricing/a1.png")
savefig(density(map(x -> x[2], data.A), group = data.economy, dpi=adpi, layout=4, title=titles, legend=false), "plots/pricing/a2.png")
savefig(density(map(x -> x[1,1], data.B), group = data.economy, dpi=adpi, layout=4, title=titles, legend=false), "plots/pricing/b11.png")
savefig(density(map(x -> x[2,2], data.B), group = data.economy, dpi=adpi, layout=4, title=titles, legend=false), "plots/pricing/b22.png")
savefig(density(map(x -> x[1,2], data.B), group = data.economy, dpi=adpi, layout=4, title=titles, legend=false), "plots/pricing/b12.png")
savefig(density(map(x -> x[1,1], data.C), group = data.economy, dpi=adpi, layout=4, title=titles, legend=false), "plots/pricing/c11.png")
savefig(density(map(x -> x[2,2], data.C), group = data.economy, dpi=adpi, layout=4, title=titles, legend=false), "plots/pricing/c22.png")
savefig(density(map(x -> x[1,2], data.C), group = data.economy, dpi=adpi, layout=4, title=titles, legend=false), "plots/pricing/c12.png")
savefig(density(map(x -> x[1], data.price), group = data.economy, dpi=adpi, layout=4, title=titles, legend=false), "plots/pricing/p1.png")
savefig(density(map(x -> x[2], data.price), group = data.economy, dpi=adpi, layout=4, title=titles, legend=false), "plots/pricing/p2.png")
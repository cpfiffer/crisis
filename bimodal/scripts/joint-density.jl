using StatsPlots
using CSV
using DataFrames
using Dates
using KernelDensity

df = DataFrame(CSV.File("/home/cpfiffer/Dropbox/Research/stockbond-joint-returns.csv"))
df.date = Date.(df.date, dateformat"y/m/d")
df = unstack(df, :date, :TICKER, :PRC)
sort!(df, :date)

function calc_returns(x)
    return vcat(missing, x[2:end] ./ x[1:(end-1)] .- 1)
end

df.bnd = calc_returns(abs.(df.BND))
df.voo = calc_returns(abs.(df.VOO))

filter!(x -> !ismissing(x.bnd) && !ismissing(x.voo) && abs(x.bnd) <= 1, df)

df.voo = map(identity, df.voo)
df.bnd = map(identity, df.bnd)
#scatter(df.bnd, df.voo, xlabel="Bond returns (BND)", ylabel = "Equity returns (VOO)", legend=false)

est = kde((df.bnd, df.voo))

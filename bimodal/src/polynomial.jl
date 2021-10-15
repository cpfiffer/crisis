using DataFrames
using CSV
using StatsPlots
using Polynomials
using LinearAlgebra
using GLM
using Printf

# things = [
#     "baseline",
#     "a2-mean-shift",
#     "a2-meanvar-shift",
#     "more-corr-meanvarshift",
# ]

things = replace.(readdir("data/individuals/"), ".csv" => "")

sort!(things)
colnames = Symbol[:intercept]
varcols = [:A, :B, :C, :price, :payoff, :true_mu_1, :true_mu_2, :true_sigma_1, :true_sigma_2]

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

function splitcols(data)
    nms = Symbol[]
    cols = []
    for col in names(data)
        if eltype(data[:,col]) <: AbstractArray
            s = size(data[1,col])

            if length(s) == 1 # Vector
                for ind in eachindex(data[1,col])
                    newcol = Symbol("$col$ind")
                    push!(nms, newcol)
                    push!(cols, map(l -> l[ind], data[:,col]))
                end
            else
                for i in 1:s[1]
                    for j in 1:s[2]
                        newcol = Symbol("$col$i$j")
                        push!(nms, newcol)
                        push!(cols, map(l -> l[i,j], data[:,col]))
                    end
                end
            end
        else
            push!(nms, Symbol(col))
            push!(cols, data[:,col])
        end
    end

    return DataFrame(cols, nms)
end

combinations(n, r) = div(factorial(n), (factorial(r) * factorial(n-r)))

function basis(x, cnms, order=size(x,2); crossterm=false)
    nrow,ncol = size(x)
    n = ncol*order
    newcols = crossterm ? n + combinations(ncol, 2) : n
    B = zeros(nrow, newcols)
    # B[:,1] .= 1
    colnames = Symbol[]
    # colnames = Symbol[:intercept]

    ind = 1
    for c in 1:ncol
        for o in 1:order
            B[:,ind] = x[:,c] .^ o
            push!(colnames, Symbol("$(cnms[c])^$o"))
            ind += 1
        end
    end

    if crossterm
        for c1 in 1:ncol
            for c2 in c1+1:ncol
                B[:,ind] = x[:,c1] .* x[:,c2]
                push!(colnames, Symbol("$(cnms[c1])*$(cnms[c2])"))
                ind += 1
            end
        end
    end
    # return B, colnames
    return DataFrame(B, colnames)
end

function polyfunction(fits, d=2)
    fns = Function[]
    for (fitnum, fit) in enumerate(fits)
        cfs = coef(fit)
        nms = coefnames(fit)

        funstr = "function p$fitnum(payoff1, payoff2)\n"

        for i in 1:length(cfs)
            if i == 1
                funstr = funstr * "\ty=$(cfs[i])\n"
            else
                funstr *= "\ty += $(cfs[i]) * $(nms[i])\n"
            end
        end

        funstr *= "end\n"

        f = Meta.parse(funstr)
        push!(fns, eval(f))
    end

    final(x,y) = map(f -> f(x,y), fns)

    return final
end

dfs = []

for thing in things
    df = DataFrame(CSV.File("data/individuals/$thing.csv"))
    unique!(df, [:payoff])
    select!(df, varcols)

    for col in varcols
        df[!, col] = parse_vec(df[:, col])
    end

    df[!, :economy] .= thing
    push!(dfs, df)
end

data = vcat(dfs...)
parsed = splitcols(data)
selcols = [:payoff1, :payoff2]

table = ""

p1s = []
p2s = []
for thing in things
    sub = filter(x -> x.economy == thing, parsed)
    B = basis(sub[:,selcols], selcols, 2, crossterm=true)

    B[!,:Y] = sub.price1
    fitted1 = lm(Term(:Y) ~ sum(Term.(Symbol.(names(B[:, Not(:Y)])))), B)

    B[!,:Y] = sub.price2
    fitted2 = lm(Term(:Y) ~ sum(Term.(Symbol.(names(B[:, Not(:Y)])))), B)


    #display(coeftable(fitted1))
    #display(coeftable(fitted2))
    # pf = polyfunction([fitted1, fitted2])
    # fig = StatsPlots.scatter([tuple(pf(x,y)... for x in parsed.payoff1, y in parsed.payoff2] |> vec)
    #                          savefig("thing.png")
    push!(p1s, fitted1)
    push!(p2s, fitted2)
end

# V1
# open("results/price-functions.txt", write=true, create=true, append=false) do io
#     for (f1, f2, thing) in zip(p1s, p2s, things)
#         tab = "$thing\n  "
#         for n in coefnames(f1)
#             nn = replace(n, "payoff" => "f")
#             tab *= @sprintf("%17s", nn)
#         end
#         tab *= "\np1"
#         for c in coef(f1)
#             tab *= @sprintf("%17.2f", c)
#         end
#         # tab *= "\n  "
#         # for (c,s) in zip(coef(f1), stderror(f1))
#         #     tab *= @sprintf("%17.2f", c/s)
#         # end

#         tab *= "\np2"
#         for c in coef(f2)
#             tab *= @sprintf("%17.2f", c)
#         end
#         # tab *= "\n  "
#         # for (c,s) in zip(coef(f2), stderror(f2))
#         #     tab *= @sprintf("%17.2f", c/s)
#         # end
#         tab *= "\n"

#         println(io, tab)
#     end
# end

# V2
open("results/price-functions.tex", write=true, create=true, append=false) do io
    tab = "\\toprule\n\$p_1\$\\\\\n"
    tab *= @sprintf("%25s", "")
    for n in coefnames(p1s[1])
        nn = replace(n, "payoff" => "f_")
        nn = replace(nn, "(Intercept)" => "a")
        nn = "\$" * nn * "\$"
        tab *= @sprintf("&%12s", nn)
    end
    tab *= "\\\\\n"

    for (f1, thing) in zip(p1s, things)
        tab *= @sprintf("%25s", thing)
        for c in coef(f1)
            tab *= @sprintf("&%12.2f", c)
        end
        tab *= "\\\\\n"
    end

    tab *= "\\midrule\n\$p_2\$\\\\\n"
    tab *= @sprintf("%25s", "")
    for n in coefnames(p2s[1])
        nn = replace(n, "payoff" => "f_")
        nn = replace(nn, "(Intercept)" => "a")
        nn = "\$" * nn * "\$"
        tab *= @sprintf("&%12s", nn)
    end
    tab *= "\\\\\n"

    for (f2, thing) in zip(p2s, things)
        tab *= @sprintf("%25s", thing)
        for c in coef(f2)
            tab *= @sprintf("&%12.2f", c)
        end
        tab *= "\\\\\n"
    end

    tab *= "\\bottomrule"
    println(io, tab)

    # for (f1, f2, thing) in zip(p1s, p2s, things)
    #     tab = "$thing\n  "
    #     for n in coefnames(f1)
    #         nn = replace(n, "payoff" => "f")
    #         tab *= @sprintf("%17s", nn)
    #     end
    #     tab *= "\np1"
    #     for c in coef(f1)
    #         tab *= @sprintf("%17.2f", c)
    #     end
    #     # tab *= "\n  "
    #     # for (c,s) in zip(coef(f1), stderror(f1))
    #     #     tab *= @sprintf("%17.2f", c/s)
    #     # end

    #     tab *= "\np2"
    #     for c in coef(f2)
    #         tab *= @sprintf("%17.2f", c)
    #     end
    #     # tab *= "\n  "
    #     # for (c,s) in zip(coef(f2), stderror(f2))
    #     #     tab *= @sprintf("%17.2f", c/s)
    #     # end
    #     tab *= "\n"

    #     println(io, tab)
    # end
end
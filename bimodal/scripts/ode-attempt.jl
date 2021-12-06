using Distributions
using Roots
using NLsolve
using Plots
using TaylorSeries

function approx_inverse(f, x)
    v = Taylor1(3)
    taylor_f = f(v) - f(x)
    taylor_f |> display
    # return inverse(f(v))
    # return f(v)
    return inverse(taylor_f)(x - f(x))
end

func(x) = x > 0 ? log(x) : 0
# func(x) = 2*x + sin(x)

# bb = 2
# xs = -bb:0.01:bb
# xs = 1:0.01:2
# ys = map(l -> (approx_inverse(func, l)), xs)
# ys_true = map(func, xs)

# plot(xs, ys, label="inverse")
# plot!(xs, ys_true, label="true")

bb = 5


function approx_inverse_nl(f, x)
    tgt(y) = f(y) - x
    z = find_zeros(tgt, 0.01, bb)

    if length(z) == 0
        return missing
    else
        println("$x: $z")
        return z #ForwardDiff.derivative(f, z[1])
        # return z[1], ForwardDiff.derivative(f, z[1])
    end
end

xs = 0.01:0.01:bb
# xs = 1:0.01:2
ys = map(l -> (approx_inverse_nl(func, l)), xs)
ys_true = map(func, xs)

p1 = scatter(reduce(vcat, ys), xs, label="inverse")
p2 = plot(xs, ys_true, label="true")

plot(p1, p2)

# approx_inverse(func, 1.0)

# plot(xs, ys)

# function gauss(x, mu=0, sigma=1)
#     1/(sigma * sqrt(2 * pi)) * exp(-1/2 * ((x-mu) / sigma) ^ 2)
# end

# function chi(y)
#     y < 0 && return 0
#     return 1 / sqrt(2 * pi * y) * exp(-y / 2)
# end

# function adjustors(z, x)
#     z2(m) = z(m) - x
#     solutions = find_zeros(z2, -Inf, Inf)

#     if length(solutions) == 0
#         return missing
#     else
#         gx = map(gauss, solutions)
#         invs = map(ss -> ForwardDiff.derivative(z, ss), solutions)
#         res = sum(gx .* abs.(invs))
#         @info "" x gauss(x) res
#         return res
#         return solutions, invs
#     end
# end

# # set_variables("x", numvars=10, order=10)
# # x = set_variable("x", numvars=10, order=10)
# x = Taylor1(Float64, 5)

# z = gauss(x)


# xs = -2:0.01:3
# ys = map(z, xs)
# ys_true = map(gauss, xs)
# ys_inv = map(m -> adjustors(z, m), xs)
# ys_chi = map(chi, xs)

# plot(xs, ys, label="taylor")
# plot!(xs, ys_inv, label="inverse")
# plot!(xs, ys_true, label="true")
# plot!(xs, ys_chi, label="chi")

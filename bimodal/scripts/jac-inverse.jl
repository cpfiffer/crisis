using Distributions
using NLsolve
using LinearAlgebra
using FiniteDiff

function approx_inverse_nl(f, x)
    ismissing(x) && return missing

    x = collect(x)
    tgt(y) = begin
        r = f(y)
        ismissing(r) && return -x
        return r - x
    end
        
    # z = nlsolve(tgt, ones(eltype(x), length(x)))
    z = nlsolve(tgt, x)

    if length(z.zero) == 0
        return missing
    else
        return z.zero #ForwardDiff.derivative(f, z[1])
        # return z[1], ForwardDiff.derivative(f, z[1])
    end
end

lp(m) = m >= 0 ? log(m) : missing
function F(x)
    if any(x .<= 0)
        return missing
    else
        return [log(x[1]), log(x[2]) + log(x[1])]
        return lp.(collect(x))
    end
end

xs = 1:0.1:5
zs = copy(xs)
ys = map(F, Iterators.product(xs, zs))
ys_inv = map(m -> approx_inverse_nl(F, m), Iterators.product(xs, zs))

js = [det(inv(ForwardDiff.jacobian(F, y))) for y in ys_inv]

# plot(xs, zs, ys)
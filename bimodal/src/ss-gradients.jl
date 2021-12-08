include("posterior.jl")

# Set default parameter set
# paramd = baseline_params
paramd = two_meanshift
# paramd = morecorr_meanvar
# paramd = morecorr
# paramd = lesscorr
# paramd = similar_assets
# paramd = corr_test
# paramd = morecorr_meanvar_s
# paramd = complex_params(0.1)

# Generate prior density
mixture = prior_mixture(paramd)

# Generate a sampling grid
grid_size = 15
a1_range = range(5, 15, length=grid_size)
a2_range = range(5, 15, length=grid_size)
grid = [[a1, a2] for a1 in a1_range, a2 in a2_range]
gridx = map(z -> z[1], grid)
gridy = map(z -> z[2], grid)

# Create attention matrix
convec(x, p) = [x * p, x * (1-p)]
# Σ_bar = diagm(convec(all_k, 0.5))
# Σ_bar = diagm(convec(all_k, 0.1))
Σ_bar = diagm(convec(all_k, 0.9))

# Generate Jacobians at each point
function jac1(x; γ=1)
    @unpack μ1, Σ1, μ2, Σ2, true_s, K = paramd

    prob = prior_s(paramd, x)
    e_f = mean(mixture)
    diff = x - e_f

    return (diff * diff') .* 1/(γ)# .* point_entropy
    # return Σ_bar * (diff * diff') .* 1/(γ) .* point_entropy
end

function jac(x; γ=1)
    @unpack μ1, Σ1, μ2, Σ2, true_s, K = paramd

    prob = prior_s(paramd, x)
    entr = -(prob * log(prob) + (1-prob) * log(1-prob))

    d1 = x - μ1
    d2 = x - μ1

    diff1 = d1 * d1'
    diff2 = d2 * d2'

    V1 = diff1 - Σ1
    V2 = diff2 - Σ2

    Vdiff1 = eigen(V1).values |> diagm
    Vdiff2 = eigen(V2).values |> diagm

    return 1/γ * (convex_combo(prob, V1, V2) + convex_combo(prob, diff1, diff2))
    # return 1/γ * (convex_combo(prob, Vdiff1, Vdiff2) + convex_combo(prob, diff1, diff2))
end

# magnitude(x) = sum(x .^ 2)
magnitude(x) = sum(abs.(x))
unitvec(x) = x ./ magnitude(x)

function a1_vec(x)
    return unitvec(x[1,:])
end

function a2_vec(x)
    return unitvec(x[2,:])
end

function prices(delta, upper)
    xs = collect(range(0, upper, step=delta))
    ys = collect(range(0, upper, step=delta))
    zs = Matrix{Vector{Float64}}(undef, length(xs), length(ys))

    for iv in 1:length(xs)
        for jv in 1:length(ys)
            if 1 == iv && 1 == jv
                # Do nothing
                zs[iv,jv] = [0.0, 0.0]
            else
                i = max(1, iv-1)
                j = max(1, jv-1)

                # @info "" i j iv jv xs[i] xs[iv] ys[j] ys[jv] df

                df = [
                    xs[iv] - xs[i],
                    ys[jv] - ys[j],
                ]

                J = jac([xs[i], ys[j]])

                dp = J*df
                zs[iv, jv] = zs[i, j] + dp
            end
        end
    end

    return xs, ys, zs
end

jacs = map(jac, grid)
a1s = map(a1_vec, jacs)
a1_x = map(z -> z[1], a1s) |> vec
a1_y = map(z -> z[2], a1s) |> vec

a2s = map(a2_vec, jacs)
a2_x = map(z -> z[1], a2s) |> vec
a2_y = map(z -> z[2], a2s) |> vec

# quiver(vec(gridx), vec(gridy), quiver=(a1_x, a1_y))
# quiver!(vec(gridx), vec(gridy), quiver=(a2_x, a2_y))

between(x, l, u) = u >= x && x >= l
xs, ys, zs = prices(0.05, 20)

l, u = 7, 13

zs = zs[between.(xs, l, u), between.(ys, l, u)]
xs = xs[between.(xs, l, u)]
ys = ys[between.(ys, l, u)]

p1 = map(z -> z[1], zs)
p2 = map(z -> z[2], zs)
prior_pdf(f) = pdf(mixture, [f[1], f[2]])
prior_density(f...) = logpdf(mixture, [f[1], f[2]])
prior_grid = [[x, y] for x in xs, y in ys]

# Plot prices
prior_plot = contour(xs, ys, prior_density, title="Prior density")
plot1 = contour(xs, ys, p1, title="Asset 1 price")
plot2 = contour(xs, ys, p2, title="Asset 2 price")

plot(prior_plot, plot1, plot2) |> display

# Calculate average prices
prior_mat = map(prior_pdf, prior_grid)
prior_mat = prior_mat ./ sum(prior_mat)
ep1 = sum(prior_mat .* p1)
ep2 = sum(prior_mat .* p2)

@info "Expected prices" ep1 ep2

# Calculate expected returns
# er1 = (p1 .- ep1)
# er2 = (p2 .- ep2)
# V = cov([vec(er1) vec(er2)])

# surprise1 = density(vec(er1))
# surprise2 = density(vec(er2))
# plot(surprise1, surprise2)
# display(V)

# @info "Expected returns" er1 er2
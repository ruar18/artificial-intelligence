
using PyPlot

function gradient(θ, X, y)
    J = 1/(2*size(y,1)) * (X * θ - y)' * (X * θ - y)
    ∇ = X' * (X * θ - y)
    return J, ∇
end

function descend(X, y, iter, α)
    θ = zeros(size(X, 2))
    errors = zeros(iter, 1)
    for i = 1:iter
        (J, ∇) = gradient(θ, X, y)
        errors[i] = J
        θ = θ - α*∇
    end
    # Comment line below when showing line of best fit
    plot(errors)
    return θ
end

function featureNormalize(X)
    X_norm = X;
    mu = zeros(1, size(X, 2));
    sigma = zeros(1, size(X, 2));
    for i = 1:size(X,2)
        mu[i] = mean(X[:,i]);
        sigma[i] = std(X[:,i]);
        X_norm[:,i] = (X[:,i] - mu[i])/sigma[i];
    end
    return X_norm
end

data = readcsv("ex1data2.txt")

# Version 1
X = hcat(ones(size(data, 1)), (data[:, 1:end-1]))

# Uncomment line below for data2, comment line above
# X = hcat(ones(size(data, 1)), featureNormalize(data[:, 1:end-1]))

y = data[:, end]
θ = descend(X, y, 50, 0.01)

# Good values:
    # data1: iter = 1500, α = 0.0001
    # data2 : iter = 1500, α = 0.01

# Uncomment lines below to show line of best fit
# scatter(data[:, 1], data[:, 2]);
# x = 5:0.1:25; y = θ[2] * x + θ[1]
# plot(x, y)

using LinearAlgebra # for symmetric matrices

# Y is the design matrix (size d x n), Z the outputs (size n)

function constructLogisticEnergy(Y::Matrix{Float64},Z::Vector{Int64}, prior_precision::Float64 = 1.0)
    # construct energy function (i.e. negative log target density) for
    # Bayesian logistic regression.
    # assume a  normal prior with covariance matrix σ^2 I, where σ^2 = 1/prior_precision
    # (if prior_precision = 0, assume a flat prior)
    μ(x) = x' * Y # sum over the parameter dimensions, row vector of length n
    η(x) = exp.(μ(x)) # row vector of length n
    n(x) = sqrt(dot(x,x))
    E(x) = sum(log.(1 .+ η(x))-Z' .* μ(x)) + prior_precision * n(x)^2 # sum over observations
    dim = size(Y)[1]
    ∇E = function(x::Vector{Float64})
        η_val = η(x)
        sum(Y .* (η_val ./(1 .+ η_val) - Z'), dims=2)[:,1] + prior_precision * x # sum over observations
    end
    h_E = function(x::Vector{Float64})
        η_val = η(x)
        prior_precision * Symmetric(Matrix{Float64}(I, dim,dim)) + LinearAlgebra.symmetric(sum([view(Y,:,i) * view(Y,:,i)' * η_val[i]/ (1 + η_val[i])^2 for i in range(1,length=length(Z))]),:U)
    end
    partials_E = function(i::Int,x::Vector{Float64})
        η_val = η(x)
        dim = length(x)
        sum(transpose(view(Y, i, :)) .* (η_val ./(1 .+ η_val) - Z')) + x[i]
    end
    return (E, ∇E, h_E, partials_E)
end

function LogisticSubsamplingTools(Y::Matrix{Float64},Z::Vector{Int64}, prior_precision::Float64 = 1.0)
    μ(j, x) = sum(x .* view(Y,:,j)) # sum over the parameter dimensions
    η(j, x) = exp(μ(j, x))
    n = length(Z)
    dim = size(Y)[1]
    partials_ss = function(i::Int,j::Int,x::Vector{Float64})
        η_val = η(j,x)
        n * Y[i,j] * (η_val/(1 + η_val)-Z[j]) + x[i]
    end
    Q_ss = LinearAlgebra.symmetric(n/4 * [maximum(abs.(view(Y,i,:) .* view(Y,j,:))) for i = 1:dim, j = 1:dim],:U)
    ∇E_ss = function(j::Int,x::Vector{Float64})
        η_val = η(j,x)
        n * view(Y,:,j) * (η_val /(1 + η_val) - Z[j]) + prior_precision * x
    end
    Hessian_E_ss = function(j::Int,x::Vector{Float64},v::Vector{Float64})
        η_val = η(j,x)
        n * view(Y,:,j) * dot(view(Y,:,j),v) * η_val/ (1 + η_val)^2 + prior_precision * v
    end
    Hessian_bound_ss = n * maximum([norm(view(Y,:,j))^2 for j=1:n])/4 + prior_precision
    return (partials_ss, Q_ss, ∇E_ss, Hessian_E_ss, Hessian_bound_ss)
end

function LogisticDominatingMatrix(Y::Matrix{Float64}, prior_precision::Float64 = 1.0)
    # return Q such that Q - ∇^2 E(x) is positive definite for all x.

    dim = size(Y)[1]

    return LinearAlgebra.symmetric(Y * transpose(Y)/4+ prior_precision*Matrix{Float64}(I, dim, dim),:U);
end

function LogisticEntrywiseDominatingMatrix(Y::Matrix{Float64}, prior_precision::Float64 = 1.0)
    # return Q such that Q_{ij} >= |∇^2 U(x)|_{ij}  for all x, i, j.
    dim = size(Y)[1]
    return LinearAlgebra.symmetric(abs.(Y) * transpose(abs.(Y))/4 + prior_precision * Matrix{Float64}(I, dim, dim),:U);
end

function LogisticThirdDerivativeBound(Y::Matrix{Float64})

    n = size(Y)[2]
    1/10*sum([norm(view(Y,:,i))^3 for i=1:n])
end

function generateLogisticData(parameter::Vector{Float64}, n_observations::Int)

  dimension = length(parameter)
  X = [ones(1,n_observations);randn(dimension-1, n_observations)]
  q = exp.(sum(parameter .* X,dims=1))
  p = q./(1 .+ q)
  U = rand(n_observations)
  Y = [Int(U[i] <= p[i]) for i in 1:n_observations]
  return (X,Y)
end

function generateCorrelatedLogisticData(parameter::Vector{Float64}, n_observations::Int, epsilon::Float64 = 0.1)

    dimension = length(parameter)
    X = ones(dimension, n_observations) + epsilon * randn(dimension, n_observations);
    q = exp.(sum(parameter .* X,dims=1))
    p = q./(1 .+ q)
    U = rand(n_observations)
    Y = [Int(U[i] <= p[i]) for i in 1:n_observations]
    return (X,Y)

end

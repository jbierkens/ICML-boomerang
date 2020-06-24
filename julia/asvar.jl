function BlockMeans(X::Array{Float64}; alpha::Float64 = 1/2, block_size::Int = -1)
  # estimate the asymptotic variance for a stationary time series using block means
  # see v.d. Vaart, Time Series, Section 5.1
  # alpha determines block size through l = [n^alpha]
  # alpha is overruled by block_size
  if ndims(X) == 1
    n = length(X)
    X = reshape(X,(1,n))
  end
  (dimension,n) = size(X)
  if block_size < 0
    block_size = Int(floor(n^alpha))
  end
  Z = Array{Float64,2}(undef,dimension,n-block_size+1) # block means
  for i = 1:(n-block_size+1)
    Z[:,i] = mean(X[:,i:(i+block_size-1)], dims = 2)
  end
  mu = mean(X,dims=2)
  v = vec(block_size/(n-block_size+1) * sum((Z-repeat(mu,1, n-block_size+1)).^2, dims=2))
  #F(x) = 1/(n-l+1) * sum(sqrt(l)*(Z-mu).<=x)
  # return (v,F)
end

function BatchMeans(X::Matrix{Float64}; alpha::Float64 = 1/2, n_batches::Int = -1, true_mean::Vector{Float64} = Vector{Float64}(undef,0))
  # estimate the asymptotic variance for a stationary time series using batch means

  # if ndims(X) == 1
  #   n = length(X)
  #   X = reshape(X,(1,n))
  # end
  (dimension,n) = size(X)
  if (n_batches <= 0)
    batch_size = Int(floor(n^alpha))
    n_batches = Int(floor(n / n^alpha))
  else
    batch_size = Int(floor(n/n_batches))
  end
  if n_batches > n/2
    return Inf
  end
  Z = Array{Float64,2}(undef,dimension,n_batches) # batch means
  unused = n - batch_size * n_batches
  for i=1:n_batches
    Z[:,i] = mean(X[:,(unused + (i-1)*batch_size + 1):(unused + i * batch_size)], dims=2)
  end
  if length(true_mean) == 0
    vec(var(Z, dims=2)*batch_size)
  else
    sum((Z - repeat(true_mean, 1, n_batches)).^2, dims = 2)/n_batches * batch_size
  end
end

function ESS(X::AbstractArray{Float64};alpha::Float64=1/2, true_variance::Vector{Float64} = Vector{Float64}(undef,0), true_mean::Vector{Float64} = Vector{Float64}(undef,0), n_batches::Int = -1)

  # if ndims(X) == 1
  #   n = length(X)
  #   X = reshape(X,(1,n))
  # end
  asvar = vec(BatchMeans(X,alpha=alpha,n_batches=n_batches,true_mean=true_mean))
  if length(true_variance) == 0
    variance = vec(var(X,dims=2))
  else
    variance = true_variance
  end
  return size(X)[2] * variance ./ asvar
end

function SquaredJumpingDistance(X::AbstractArray{Float64})

  if ndims(X) == 1
    n_samples = length(X)
    differences = X[2:end] - X[1:end-1]
  else
    n_samples = size(X)[2]
    differences = X[:,2:end] - X[:,1:end-1]
  end
  return sumabs2(differences)/(n_samples-1)

end

function PooledVariance(AsVar::Vector{Float64}, n_samples::Vector{Int}, SampleVariance::Vector{Float64})

  return sum(n_samples .* SampleVariance ./ AsVar)/sum(n_samples ./AsVar)

end

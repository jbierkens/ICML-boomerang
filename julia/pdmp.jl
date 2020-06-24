using Statistics
include("asvar.jl")

function switchingtime(a::Float64,b::Float64,u::Float64=rand())
# generate switching time for rate of the form max(0, a + b s) + c
# under the assumptions that b > 0, c > 0
  if (b > 0)
    if (a < 0)
      return -a/b + switchingtime(0.0, b, u);
    else # a >= 0
      return -a/b + sqrt(a^2/b^2 - 2 * log(u)/b);
    end
  elseif (b == 0) # degenerate case
    if (a < 0)
      return Inf;
    else # a >= 0
      return -log(u)/a;
    end
  else # b <= 0
    if (a <= 0)
      return Inf;
    else # a > 0
      y = -log(u); t1=-a/b;
      if (y >= a * t1 + b *t1^2/2)
        return Inf;
      else
        return -a/b - sqrt(a^2/b^2 + 2 * y /b);
      end
    end
  end
end

function switchingtime_quadratic(a::Float64, b::Float64, c::Float64,u::Float64=rand())
  dummy::Float64 = (-b^3 + 6*a*b*c + 2*(-6*c^2*log(u) + c * sqrt(-3*a^2*b^2 + 16*a^3*c + 6*b^3*log(u) - 36*a*b*c*log(u) + 36*c^2*(log(u))^2)))^(1/3)
  return 1/(2c)*(-b + (b^2-4*a*c)/dummy + dummy)
end

function EstimateMomentAlongTrajectory(t_skeleton::Vector{Float64}, x_skeleton::Vector{Vector{Float64}}, v_skeleton::Vector{Vector{Float64}}; p = 1, ellipticQ::Bool = false, x_ref::Vector{Float64} = Vector{Float64}(undef,0))
  # assume every component follows dynamics x_t = x_ref + (x_0-x_ref)*cos(t) + v_0*sin(t)

  dim = length(x_skeleton[1])

  if ellipticQ && length(x_ref) == 0
    println("WARNING : EstimateMomentAlongTrajectory : No valid reference point provided.")
    x_ref = zeros(dim)
  end
  T = (t_skeleton[end]-t_skeleton[1])
  estimate = 0.0
  for i=2:length(t_skeleton)
    v = v_skeleton[i-1]
    x = x_skeleton[i-1]
    t0 = t_skeleton[i-1]
    t1 = t_skeleton[i]
    Δt = t1 - t0
    if ellipticQ
      if p==1
        estimate = estimate .+ 1/T * (x_ref * Δt + (x - x_ref) * sin(Δt) + v * (1-cos(Δt)))
      elseif p==2
        y = x - x_ref
        estimate = estimate .+ 1/(2*T) * (v .*(4 * x_ref + y) + Δt * (v.^2 + 2 * x_ref.^2 + y.^2) - (4 * x_ref + y * cos(Δt) + v * sin(Δt)).*(v*cos(Δt) - y *sin(Δt)))
      else
        error("Higher order moments (p >2) along Elliptic trajectories currently not implemented.")
      end
    else # assume linear trajectories
      estimate =  estimate .+ 1 ./((p+1)*v_skeleton[i-1] * T) .* ( x_skeleton[i].^(p+1) - x_skeleton[i-1].^(p+1))
    end
  end
  return estimate

end

function EllipticDynamics(t::Real, y0::Vector{Float64}, w0::Vector{Float64})
    # simulate dy/dt = w, dw/dt = -y

    y_new = y0 * cos(t) + w0 * sin(t);
    w_new = -y0 * sin(t) + w0 * cos(t);

    return (y_new, w_new);

end

function SkeletonIntoBatches(t_skeleton::Vector{Float64}, x_skeleton::Vector{Vector{Float64}}, v_skeleton::Vector{Vector{Float64}}, n_batches::Int, ellipticQ::Bool = false, x_ref::Vector{Float64} = Vector{Float64}(undef,0))

  t_skeletons::Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef,0)
  x_skeletons::Vector{Vector{Vector{Float64}}} = Vector{Vector{Vector{Float64}}}(undef,0)
  v_skeletons = similar(x_skeletons)

  if n_batches < 1
    error("n_batches < 1")
  end

  if (ellipticQ && length(x_ref) == 0)
    x_ref = zeros(length(x_skeleton[1]))
  end
  T_max = t_skeleton[end]
  j_max = length(t_skeleton)
  T_batch = T_max / n_batches
  t_current = t_skeleton[1]
  x_current = x_skeleton[1]
  v_current = v_skeleton[1]
  j = 2
  for i = 1:n_batches
    t_skeleton_batch = Vector{Float64}(undef,0)
    x_skeleton_batch = Vector{Vector{Float64}}(undef,0)
    v_skeleton_batch = similar(x_skeleton_batch)
    # add initial position to skeleton
    push!(t_skeleton_batch, t_current)
    push!(x_skeleton_batch, x_current)
    push!(v_skeleton_batch, v_current)

    while (t_skeleton[j] < i*T_batch && j < j_max)
      t_current = t_skeleton[j]
      x_current = x_skeleton[j]
      v_current = v_skeleton[j]
      push!(t_skeleton_batch, t_current)
      push!(x_skeleton_batch, x_current)
      push!(v_skeleton_batch, v_current)
      j = j + 1
    end
    # j = min (j : t_skeleton[j] > i * T_batch)

    # add final position of this batch
    push!(t_skeleton_batch, i*T_batch)
    Δt = (i*T_batch - t_current)
    if  (ellipticQ)
      (y,v_current) = EllipticDynamics(Δt,x_current-x_ref, v_current)
      x_current = x_ref + y
    else
      x_current = x_current + Δt * v_current
    end
    t_current = i*T_batch
    push!(x_skeleton_batch, x_current)
    push!(v_skeleton_batch, v_current)

    # add skeleton to list of skeletons
    push!(t_skeletons, t_skeleton_batch)
    push!(x_skeletons, x_skeleton_batch)
    push!(v_skeletons, v_skeleton_batch)
  end
  return (t_skeletons, x_skeletons, v_skeletons)

end

function BatchMeansPDMP(t_skeleton::Vector{Float64}, x_skeleton::Vector{Vector{Float64}}, v_skeleton::Vector{Vector{Float64}}; n_batches::Int = 100, ellipticQ::Bool = false, x_ref::Vector{Float64} = Vector{Float64}(undef,0))

  dimension = length(x_skeleton[1])
  if (n_batches <= 0)
    error("A valid number of batches is required.")
  end
  T_batch = (t_skeleton[end]-t_skeleton[1])/n_batches
  (t_skeletons, x_skeletons, v_skeletons) = SkeletonIntoBatches(t_skeleton, x_skeleton, v_skeleton, n_batches, ellipticQ, x_ref)
  Z = Array{Float64,2}(undef,dimension,n_batches) # batch means
  for i=1:n_batches
    Z[:,i] = EstimateMomentAlongTrajectory(t_skeletons[i],x_skeletons[i],v_skeletons[i],ellipticQ=ellipticQ,x_ref=x_ref,p=1)
  end
  vec(var(Z, dims=2)*T_batch)
end

function ess_pdmp_components(t_skeleton::Vector{Float64}, x_skeleton::Vector{Vector{Float64}}, v_skeleton::Vector{Vector{Float64}}; n_batches::Int = 50, ellipticQ::Bool = false, x_ref::Vector{Float64} = Vector{Float64}(undef,0))

  asymptotic_variances = BatchMeansPDMP(t_skeleton, x_skeleton, v_skeleton, n_batches = n_batches, ellipticQ = ellipticQ, x_ref = x_ref)
  if any(asymptotic_variances .< 0)
    println("WARNING : ess_pdmp : at least one component of the asymptotic variance is negative.")
    println("asymptotic_variances = ", asymptotic_variances)
  end
  variances = EstimateMomentAlongTrajectory(t_skeleton,x_skeleton,v_skeleton, ellipticQ = ellipticQ, x_ref = x_ref, p=2) .- (EstimateMomentAlongTrajectory(t_skeleton,x_skeleton,v_skeleton, ellipticQ = ellipticQ, x_ref = x_ref).^2)
  if any(variances .< 0)
    println("WARNING : ess_pdmp : at least one component of the variance is negative.")
    println("variances = ", variances)
  end
  ess = variances./asymptotic_variances*(t_skeleton[end]-t_skeleton[1])
end

function ess_pdmp_squared_radius(t_skeleton::Vector{Float64}, x_skeleton::Vector{Vector{Float64}}, v_skeleton::Vector{Vector{Float64}}; n_batches::Int = 50, ellipticQ::Bool = false, x_ref::Vector{Float64} = Vector{Float64}(undef,0), true_mean::Float64 = NaN, true_variance::Float64 = NaN)

  n_samples = n_batches * 1000
  samples = ExtractSamples(t_skeleton, x_skeleton, v_skeleton, n_samples, ellipticQ=ellipticQ, x_ref = x_ref);
  X = sum(samples.^2, dims = 1)
  if isnan(true_mean)
    mean = Vector{Float64}(undef,0)
  else
    mean = [true_mean]
  end
  if isnan(true_variance)
    variance = Vector{Float64}(undef,0)
  else
    variance = [true_variance]
  end
  ESS(X,n_batches = n_batches, true_mean = mean, true_variance = variance)[1]
end

function plot_pdmp(t_skeleton, x_skeleton, v_skeleton; ellipticQ::Bool = false, x_ref::Vector{Float64} = Vector{Float64}(undef,0), name::String = "", n_samples::Int = 0)

  if (n_samples == 0)
    n_samples = Int(round(t_skeleton[end] - t_skeleton[1]))
  end
  samples = ExtractSamples(t_skeleton, x_skeleton, v_skeleton, n_samples, ellipticQ=ellipticQ, x_ref = x_ref);
  plot_ref = plot(samples[1,:], samples[2,:],legend = false, ratio=:equal);
  savefig(plot_ref, string(name,"-trace.pdf"))
  # plot_ref = bar(autocor(samples'))
  # if (length(name) > 0)
  #     savefig(plot_ref, string(name,"-acf.pdf"))
  # end
end

function ExtractSamples(t_skeleton::Vector{Float64}, x_skeleton::Vector{Vector{Float64}}, v_skeleton::Vector{Vector{Float64}}, n_samples::Int; x_ref::Vector{Float64} = Vector{Float64}(undef,0), ellipticQ::Bool=false)

    T = t_skeleton[end]
    Δt = T/(n_samples-1)
    dim = size(x_skeleton[1])[1]
    samples = Matrix{Float64}(undef, dim, n_samples)
    t = 0.0
    if (ellipticQ && length(x_ref) == 0)
        println("WARNING : ExtractSamples : No valid reference point provided.")
        x_ref = zeros(dim)
    end
    x = x_skeleton[1]
    v = v_skeleton[1]
    samples[:,1] = x
    skeleton_index = 2
    counter = 1
    for i in range(2, stop=n_samples)
        t_max = (i-1) * Δt
        while (counter + 1 < length(t_skeleton) && t_skeleton[counter + 1] < t_max)
            counter = counter + 1
            x = x_skeleton[counter]
            v = v_skeleton[counter]
            t = t_skeleton[counter]
        end
        if (ellipticQ)
            (y,v) = EllipticDynamics(t_max - t, x-x_ref, v)
            x = y + x_ref
        else
            x = x + v * (t_max - t)
        end
        t = t_max
        samples[:,i] = x
    end

    return samples
end

# optimization routiness for preprocessing
using Optim

# for automatic differentiation
#using ReverseDiff: GradientTape, compile, gradient!, HessianTape, hessian!

# Cholesky decomposition, Symmetric
using LinearAlgebra

include("pdmp.jl")

function preprocess(E::Function, ∇E::Function, h_E::Function, dim::Int, x0::Vector{Float64} = zeros(dim))
    # minimize E (with gradient and Hessian g_E!, h_E!, respectively) to obtain x_ref
    # return x_ref and L, the Cholesky decomposition of Σ = [∇^2 E(x_ref)]^{-1} = L * L'

    # result = optimize(E,g_E!,h_E!,x0);

    ∇E! = function(storage, x)
        storage[:] = ∇E(x)
    end
    result = optimize(E,∇E!,x0);
    x_ref = Optim.minimizer(result);
    Σ_inv = h_E(x_ref);
    return (x_ref, Σ_inv);
end

function Boomerang(∇E::Function, M1::Float64, T::Float64, refresh_rate::Float64=1.0, x_ref::Vector{Float64} = Vector{Float64}(undef,0), Σ_inv::AbstractArray{Float64,2} = Matrix{Float64}(undef, 0,0))
    # g_E! and h_E! are gradient and hessian of negative log density E respectively
    #  (implemented to be compatible with the Optim package, so g_E!(storage,x), h_E!(storage,x))
    # M1 is a constant such that |∇^2 E(x)| <= M1  for all x.
    # M2 is a constant such that |∇E(x_ref)| ≤ M2
    # Boomerang is implemented with affine bound.


    if (length(x_ref) == 0 || size(Σ_inv)[1] == 0)
        error("Boomerang requires non-empty x_ref and Σ_inv")
    end

    Σ = inv(Σ_inv)
    if typeof(Σ) == Diagonal{Float64, Array{Float64,1}}
        Σ_sqrt = sqrt(Σ)
    else
        Σ_sqrt = cholesky(Symmetric(Σ)).L
    end


    dim = length(x_ref)

    # gradU(x) = g_E!(dummy,x) - Σ_inv * (x-x_ref); # O(d^2) to compute

    t = 0.0;
    x = x_ref; v = Σ_sqrt * randn(dim);
    gradU = ∇E(x) - Σ_inv * (x-x_ref); # O(d^2) to compute
    M2 = sqrt(dot(gradU,gradU));
    updateSkeleton = false;
    finished = false;
    x_skeleton = Vector{Vector{Float64}}(undef,0);
    v_skeleton = similar(x_skeleton);
    t_skeleton = Vector{Float64}(undef, 0);
    push!(x_skeleton, x);
    push!(v_skeleton, v);
    push!(t_skeleton, t);
    Δt_refresh = -log(rand())/refresh_rate;
    rejected_switches = 0;
    accepted_switches = 0;
    phaseSpaceNorm = sqrt(dot(x-x_ref,x-x_ref) + dot(v,v));
    a = dot(v, gradU)
    b = M1 * phaseSpaceNorm^2 + M2 * phaseSpaceNorm

    while (!finished)
        Δt_switch_proposed = switchingtime(a,b);
        Δt = min(Δt_switch_proposed,Δt_refresh);
        if t + Δt > T
            Δt = T - t
            finished = true
            updateSkeleton = true
        end
        (y, v) = EllipticDynamics(Δt, x-x_ref, v);
        x = y + x_ref;
        t = t + Δt;
        a = a + b*Δt;
        gradU = ∇E(x) - Σ_inv * (x-x_ref); # O(d^2) to compute
        if (!finished && Δt_switch_proposed < Δt_refresh)
            switch_rate = dot(v, gradU) # no need to take positive part
            simulated_rate = a
            if simulated_rate < switch_rate
                println("simulated rate: ", simulated_rate)
                println("actual switching rate: ", switch_rate)
                error("switching rate exceeds bound.")
            end
            if rand() * simulated_rate <= switch_rate
                # obtain new velocity by reflection
                skewed_grad = Σ_sqrt' * gradU
                v = v - 2 * switch_rate / dot(skewed_grad,skewed_grad) * Σ_sqrt * skewed_grad
                phaseSpaceNorm = sqrt(dot(x-x_ref,x-x_ref) + dot(v,v));
                a = -switch_rate
                b = M1 * phaseSpaceNorm^2 + M2 * phaseSpaceNorm;
                updateSkeleton = true
                accepted_switches += 1
            else
                a = switch_rate
                updateSkeleton = false
                rejected_switches += 1
            end
            # update refreshment time and switching time bound
            Δt_refresh = Δt_refresh - Δt_switch_proposed

        elseif !finished
            # so we refresh
            updateSkeleton = true
            v = Σ_sqrt * randn(dim);
            phaseSpaceNorm = sqrt(dot(x-x_ref,x-x_ref) + dot(v,v));
            a = dot(v, gradU)
            b = M1 * phaseSpaceNorm^2 + M2 * phaseSpaceNorm;

            # compute new refreshment time
            Δt_refresh = -log(rand())/refresh_rate;
        end

        if updateSkeleton
            push!(x_skeleton, x)
            push!(v_skeleton, v)
            push!(t_skeleton, t)
            updateSkeleton = false
        end

    end
    println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    println("number of proposed switches: ", accepted_switches + rejected_switches)
    return (t_skeleton, x_skeleton, v_skeleton, x_ref)
end

function BoomerangSubsampling(∇E_ss::Function, h_E_ss::Function, hessian_bound::Float64, N::Int, T::Float64, refresh_rate::Float64=1.0, x_ref::Vector{Float64} = Vector{Float64}(undef,0), grad_ref::Vector{Float64} = Vector{Float64}(undef,0), Σ_inv::AbstractArray{Float64,2} = Matrix{Float64}(undef,0,0))
    # ∇E_ss(i,x) is ∇E^i(x) where the negative log (Lebesgue) density is given by E(x) = \frac 1 N \sum_{i=1}^N E^i (x)
    # H_E_ss(i,x,v) = ∇^2 E^i(x) v
    # hessian_bound is a constant such that ∇^2 E^i(x) - ∇^2 E^i(y) <= (hessian_bound) I in positive definite ordering, for all i, x, y.

    if (length(x_ref) == 0 || size(Σ_inv)[1] == 0 || length(grad_ref) ==0)
        error("BoomerangSubsampling requires non-empty x_ref, grad_ref = ∇E(x_ref) and L = cholesky(Σ).L")
    end

    Σ = inv(Σ_inv)
    if typeof(Σ_inv) == Diagonal{Float64, Array{Float64,1}}
        Σ_sqrt = sqrt(Σ)
    else
        Σ_sqrt = cholesky(Symmetric(Σ)).L
    end

    dim = length(x_ref)
    t = 0.0;
    x = x_ref; v = Σ_sqrt * randn(dim);
    # ∇E_estimate = similar(x_ref)
    # ∇E_estimate_ref = similar(x_ref)
    # H_E_estimate_ref = similar(x_ref)
    statespacenorm(x,v) = sqrt(dot(x-x_ref,x-x_ref) + dot(v,v));
    simulated_rate = hessian_bound * statespacenorm(x,v)^2/2 + statespacenorm(x,v) * sqrt(dot(grad_ref,grad_ref));
    updateSkeleton = false;
    finished = false;
    x_skeleton = Vector{Vector{Float64}}(undef,0);
    v_skeleton = similar(x_skeleton);
    t_skeleton = Vector{Float64}(undef, 0);
    push!(x_skeleton, x);
    push!(v_skeleton, v);
    push!(t_skeleton, t);
    Δt_refresh = -log(rand())/refresh_rate;
    rejected_switches = 0;
    accepted_switches = 0;
    while !finished
        Δt_switch_proposed = -log(rand())/simulated_rate;
        Δt = min(Δt_switch_proposed,Δt_refresh);
        if t + Δt > T
            Δt = T - t
            finished = true
            updateSkeleton = true
        end
        (y, v) = EllipticDynamics(Δt, x-x_ref, v);
        x = y + x_ref;
        t = t + Δt;
        if (!finished && Δt_switch_proposed < Δt_refresh)
            ss_index = rand(1:N)
            stochastic_gradient = ∇E_ss(ss_index,x) - h_E_ss(ss_index, x_ref,x-x_ref) - ∇E_ss(ss_index, x_ref) + grad_ref
            switch_rate = dot(v, stochastic_gradient) # no need to take positive part
            if simulated_rate < switch_rate
                println("simulated rate: ", simulated_rate)
                println("actual switching rate: ", switch_rate)
                error("switching rate exceeds bound.")
            end
            if rand() * simulated_rate <= switch_rate
                # obtain new velocity by reflection
                skewed_grad = Σ_sqrt' * stochastic_gradient # O(d^2)
                v = v - 2 * switch_rate / dot(skewed_grad,skewed_grad) * Σ_sqrt * skewed_grad
                simulated_rate = hessian_bound * statespacenorm(x,v)^2/2 + statespacenorm(x,v) * sqrt(dot(grad_ref,grad_ref));
                updateSkeleton = true
                accepted_switches += 1
            else
                updateSkeleton = false
                rejected_switches += 1
            end
            # update refreshment time and switching time bound
            Δt_refresh = Δt_refresh - Δt
        elseif !finished
            # so we refresh
            updateSkeleton = true
            v = Σ_sqrt * randn(dim);
            simulated_rate = hessian_bound * statespacenorm(x,v)^2/2 + statespacenorm(x,v) * sqrt(dot(grad_ref,grad_ref));
            # compute new refreshment time
            Δt_refresh = -log(rand())/refresh_rate;
        end
        Δt_switch_proposed = -log(rand())/simulated_rate;

        if updateSkeleton
            push!(x_skeleton, x)
            push!(v_skeleton, v)
            push!(t_skeleton, t)
            updateSkeleton = false
        end
    end
    println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    println("number of proposed switches: ", accepted_switches + rejected_switches)
    return (t_skeleton, x_skeleton, v_skeleton)
end

function FactorizedBoomerang(∇E::Function, M1::Vector{Float64}, T::Float64, refresh_rate::Float64=0.1, x_ref::Vector{Float64} = Vector{Float64}(undef,0), Σ_inv::AbstractArray{Float64,2} = Matrix{Float64}(undef,0,0))
    # g_E! and h_E! are gradient and hessian of negative log density E respectively
    #  (implemented to be compatible with the Optim package, so g_E!(storage,x), h_E!(storage,x))
    # M1 is a constant such that ∇^2 E(x_1) - ∇^2 E(x_2) <= M1 I  for all x_1, x_2.

    if (length(x_ref) == 0 || size(Σ_inv)[1] == 0)
        error("FactorizedBoomerang requires non-empty x_ref and Σ_inv")
    end
    dim = length(x_ref)

    Σ = inv(Σ_inv)
    if typeof(Σ) == Diagonal{Float64, Array{Float64,1}}
        Σ_sqrt = sqrt(Σ)
    else
        Σ_sqrt = cholesky(Symmetric(Σ)).L
    end


    # gradU(x) = g_E!(dummy,x) - Σ_inv * (x-x_ref); # O(d^2) to compute
    α = 1.0
    t = 0.0;
    x = x_ref; v = Σ_sqrt * randn(dim);
    gradU = ∇E(x) - Σ_inv * (x-x_ref); # O(d^2) to compute # Vector
    M2 = abs.(gradU); #vector |∇U(0)| m in the paper
    updateSkeleton = false;
    finished = false;
    x_skeleton = Vector{Vector{Float64}}(undef,0);
    v_skeleton = similar(x_skeleton);
    t_skeleton = Vector{Float64}(undef, 0);
    push!(x_skeleton, x);
    push!(v_skeleton, v);
    push!(t_skeleton, t);
    Δt_refresh = -log(rand())/refresh_rate; #TOCHANGE
    rejected_switches = 0;
    accepted_switches = 0;
    x_bar = sqrt.((x-x_ref).^2 + v.^2); #vector
    phaseSpaceNorm = dot(x-x_ref, x-x_ref) + dot(v,v)
    a = max.(v.*gradU, 0 )
    b = x_bar.*M2 + 0.5*M1.*(α*x_bar.^2 .+ 1/α*phaseSpaceNorm)
    Δt_proposed_switches = switchingtime.(a,b);

    while (true)
        # simulated_rate = phaseSpaceNorm^2 * M1/2 + phaseSpaceNorm * M2;
        i = argmin(Δt_proposed_switches)
        Δt_switch_proposed = Δt_proposed_switches[i]
        Δt = min(Δt_switch_proposed, Δt_refresh);
        if t + Δt > T
            Δt = T - t
            finished = true
            updateSkeleton = true
        end
        (y, v) = EllipticDynamics(Δt, x-x_ref, v);
        x = y + x_ref;
        t = t + Δt;
        a = a + b*Δt;
        gradU = ∇E(x) - Σ_inv * (x-x_ref); # O(d^2) to compute
        # Δt_update = Δt_update + Δt # time since last update of a
        if (!finished && Δt_switch_proposed < Δt_refresh)
            switch_rate = v[i]*gradU[i] # no need to take positive part
            # simulated_rate = a + b * Δt_update
            simulated_rate = a[i]
            if simulated_rate < switch_rate
                println("simulated rate: ", simulated_rate)
                println("actual switching rate: ", switch_rate)
                error("switching rate exceeds bound.")
            end
            if rand() * simulated_rate <= switch_rate
                # obtain new velocity by reflection
                # we use that Σ = Σ_sqrt * Σ_sqrt'
                v[i] = -v[i]
                a[i] = - switch_rate
                x_bar = sqrt.((x-x_ref).^2 + v.^2); #vector
                phaseSpaceNorm = dot(x-x_ref, x-x_ref) + dot(v,v)
                b = x_bar.*M2 + 0.5*M1.*(α*x_bar.^2 .+ 1/α*phaseSpaceNorm)
                Δt_proposed_switches = switchingtime.(a,b);
                updateSkeleton = true
                accepted_switches += 1
            else
                Δt_proposed_switches .-= Δt_switch_proposed
                a[i] = switch_rate
                b[i] = x_bar[i]*M2[i] + 0.5*M1[i]*(α*x_bar[i]^2 + 1/α*phaseSpaceNorm)
                Δt_proposed_switches[i] = switchingtime(a[i],b[i])
                updateSkeleton = false
                rejected_switches += 1
            end
            # update refreshment time and switching time bound
            Δt_refresh = Δt_refresh - Δt
            # Δt_update = 0.0
        elseif !finished
            # so we refresh
            updateSkeleton = true
            v = Σ_sqrt * randn(dim);
            x_bar = sqrt.((x-x_ref).^2 + v.^2);
            phaseSpaceNorm =  dot(x-x_ref, x-x_ref) + dot(v,v)
            a = max.(v.*gradU,0) #vector Poistive part?
            b = x_bar.*M2 + 0.5*M1.*(α*x_bar.^2 .+ 1/α*phaseSpaceNorm)
            Δt_proposed_switches = switchingtime.(a,b);
            # compute new refreshment time
            Δt_refresh = -log(rand())/refresh_rate;
        end

        if updateSkeleton
            push!(x_skeleton, x)
            push!(v_skeleton, v)
            push!(t_skeleton, t)
            updateSkeleton = false
        end

        if finished
            break;
        end
    end
    println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    println("number of proposed switches: ", accepted_switches + rejected_switches)
    return (t_skeleton, x_skeleton, v_skeleton, x_ref)
end

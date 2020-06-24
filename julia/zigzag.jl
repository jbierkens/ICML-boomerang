include("pdmp.jl")

using LinearAlgebra

function ZigZag(∂E::Function, Q::Symmetric{Float64,Matrix{Float64}}, T::Float64, x_init::Vector{Float64} = Vector{Float64}(undef,0), v_init::Vector{Int} = Vector{Int}(undef,0), excess_rate::Float64 = 0.0)
    # E_partial_derivative(i,x) is the i-th partial derivative of the potential E, evaluated in x
    # Q is a symmetric matrix with nonnegative entries such that |(∇^2 E(x))_{ij}| <= Q_{ij} for all x, i, j
    # T is time horizon

    dim = size(Q)[1]
    if (length(x_init) == 0 || length(v_init) == 0)
        x_init = zeros(dim)
        v_init = rand((-1,1), dim)
    end

    b = Q * ones(dim);

    t = 0.0;
    x = x_init; v = v_init;
    updateSkeleton = false;
    finished = false;
    x_skeleton = Vector{Vector{Float64}}(undef,0);
    v_skeleton = similar(x_skeleton);
    t_skeleton = Vector{Float64}(undef, 0);
    push!(x_skeleton, x);
    push!(v_skeleton, v);
    push!(t_skeleton, t);
    rejected_switches = 0;
    accepted_switches = 0;
    initial_gradient = [∂E(i,x) for i in 1:dim];
    a = v .* initial_gradient

    Δt_proposed_switches = switchingtime.(a,b)
    if (excess_rate == 0.0)
        Δt_excess = Inf
    else
        Δt_excess = -log(rand())/(dim*excess_rate)
    end

    while (!finished)
        i = argmin(Δt_proposed_switches) # O(d)
        Δt_switch_proposed = Δt_proposed_switches[i]
        Δt = min(Δt_switch_proposed,Δt_excess);
        if t + Δt > T
            Δt = T - t
            finished = true
            updateSkeleton = true
        end
        x = x + v * Δt; # O(d)
        t = t + Δt;
        a = a + b * Δt; # O(d)

        if (!finished && Δt_switch_proposed < Δt_excess)
            switch_rate = v[i] * ∂E(i,x)
            proposedSwitchIntensity = a[i]
            if proposedSwitchIntensity < switch_rate
                println("ERROR: Switching rate exceeds bound.")
                println(" simulated rate: ", proposedSwitchIntensity)
                println(" actual switching rate: ", switch_rate)
                error("Switching rate exceeds bound.")
            end
            if rand() * proposedSwitchIntensity <= switch_rate
                # switch i-th component
                v[i] = -v[i]
                a[i] = -switch_rate
                updateSkeleton = true
                accepted_switches += 1
            else
                a[i] = switch_rate
                updateSkeleton = false
                rejected_switches += 1
            end
            # update refreshment time and switching time bound
            Δt_excess = Δt_excess - Δt_switch_proposed
            Δt_proposed_switches = Δt_proposed_switches .- Δt_switch_proposed
            Δt_proposed_switches[i] = switchingtime(a[i],b[i])
        elseif !finished
            # so we switch due to excess switching rate
            updateSkeleton = true
            i = rand(1:dim)
            v[i] = -v[i]
            a[i] = v[i] * ∂E(i,x)

            # update upcoming event times
            Δt_proposed_switches = Δt_proposed_switches .- Δt_excess
            Δt_excess = -log(rand())/(dim*excess_rate);
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
    return (t_skeleton, x_skeleton, v_skeleton)

end


function ZigZagSubsampling(∂E::Function, Q::Symmetric{Float64,Matrix{Float64}}, N::Int, T::Float64, x_ref::Vector{Float64} = Vector{Float64}(undef,0), ∇E_ref::Vector{Float64} = Vector{Float64}(undef,0), x_init::Vector{Float64} = Vector{Float64}(undef,0), v_init::Vector{Int} = Vector{Int}(undef,0), excess_rate::Float64 = 0.0)
# Zig-Zag with subsampling and control variates
# E_partial_derivatives(i,l,x) gives the ∂_i E^l(x), where 1/N ∑_{l=1}^N E^l(x) = E(x), the full potential function
# Q is a symmetric matrix with nonnegative entries such that ∂_i ∂_j E^l(x) ≤ Q_{ij} for all i,j,l,x


    if (length(x_ref) == 0 || length(∇E_ref) == 0)
        controlvariates = false;
        error("ZigZagSubampling without control variates currently not supported")
    else
        controlvariates = true;
    end

    dim = size(Q)[1]
    if length(x_init) == 0
        if controlvariates
            x_init = x_ref
        else
            x_init = zeros(dim)
        end
    end
    if length(v_init) == 0
        v_init = rand((-1,1), dim)
    end
    C = vec(sqrt.(sum(Q.^2, dims=2)))
    b = C * sqrt(dim); #  = C |v|

    t = 0.0;
    x = x_init; v = v_init;
    updateSkeleton = false;
    finished = false;
    x_skeleton = Vector{Vector{Float64}}(undef,0);
    v_skeleton = similar(x_skeleton);
    t_skeleton = Vector{Float64}(undef, 0);
    push!(x_skeleton, x);
    push!(v_skeleton, v);
    push!(t_skeleton, t);
    rejected_switches = 0;
    accepted_switches = 0;
    a = vec(v .* ∇E_ref + C * norm(x-x_ref))
    Δt_proposed_switches = switchingtime.(a,b)

    if (excess_rate == 0.0)
        Δt_excess = Inf
    else
        Δt_excess = -log(rand())/(dim*excess_rate)
    end

    while (!finished)
        # proposedSwitchIntensity = phaseSpaceNorm^2 * M1/2 + phaseSpaceNorm * M2;
        i = argmin(Δt_proposed_switches) # O(d)
        Δt_switch_proposed = Δt_proposed_switches[i]
        Δt = min(Δt_switch_proposed,Δt_excess);
        if t + Δt > T
            Δt = T - t
            finished = true
            updateSkeleton = true
        end
        x = x + v * Δt; # O(d)
        t = t + Δt;
        a = a + b * Δt; # O(d)

        if (!finished && Δt_switch_proposed < Δt_excess)
            j = rand(1:N)
            switch_rate = v[i] * (∇E_ref[i] + ∂E(i,j,x) - ∂E(i,j,x_ref))
            proposedSwitchIntensity = a[i]
            if proposedSwitchIntensity < switch_rate
                println("ERROR: Switching rate exceeds bound.")
                println(" simulated rate: ", proposedSwitchIntensity)
                println(" actual switching rate: ", switch_rate)
                error("Switching rate exceeds bound.")
            end
            if rand() * proposedSwitchIntensity <= switch_rate
                # switch i-th component
                v[i] = -v[i]
                a[i] = v[i] * ∇E_ref[i] + C[i] * norm(x-x_ref)
                updateSkeleton = true
                accepted_switches += 1
            else
                updateSkeleton = false
                rejected_switches += 1
            end
            # update refreshment time and switching time bound
            Δt_excess = Δt_excess - Δt_switch_proposed
            Δt_proposed_switches = Δt_proposed_switches .- Δt_switch_proposed
            Δt_proposed_switches[i] = switchingtime(a[i],b[i])
        elseif !finished
            # so we switch due to excess switching rate
            updateSkeleton = true
            i = rand(1:dim)
            v[i] = -v[i]
            a[i] = v[i] * ∇E_ref[i] + C[i] * norm(x-x_ref)

            # update upcoming event times
            Δt_proposed_switches = Δt_proposed_switches .- Δt_excess
            Δt_excess = -log(rand())/(dim*excess_rate);
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
    return (t_skeleton, x_skeleton, v_skeleton)

end

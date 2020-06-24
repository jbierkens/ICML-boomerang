include("pdmp.jl")

tolerance = 1e-7 # for comparing switching rate and bound

using LinearAlgebra

function reflect(gradient::Vector{Float64}, v::Vector{Float64})

    return v - 2 * (transpose(gradient) * v / dot(gradient,gradient)) * gradient

end

function BPS(∇E::Function, Q::Symmetric{Float64,Matrix{Float64}}, T::Float64; x_init::Vector{Float64} = Vector{Float64}(undef,0), v_init::Vector{Float64} = Vector{Float64}(undef,0), refresh_rate::Float64 = 1.0)
    # g_E! is the gradient of the energy function E
    # Q is a symmetric matrix such that Q - ∇^2 E(x) is positive semidefinite

    dim = size(Q)[1]
    if (length(x_init) == 0 || length(v_init) == 0)
        x_init = zeros(dim)
        v_init = randn(dim)
    end
    dummy = similar(x_init)

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
    gradient = ∇E(x);
    a = transpose(v) * gradient;
    b = transpose(v) * Q * v;

    Δt_switch_proposed = switchingtime(a,b)
    if refresh_rate <= 0.0
        Δt_refresh = Inf
    else
        Δt_refresh = -log(rand())/refresh_rate
    end

    while (!finished)
        Δt = min(Δt_switch_proposed,Δt_refresh);
        if t + Δt > T
            Δt = T - t
            finished = true
            updateSkeleton = true
        end
        x = x + v * Δt; # O(d)
        t = t + Δt;
        a = a + b * Δt; # O(d)
        gradient = ∇E(x)

        if (!finished && Δt_switch_proposed < Δt_refresh)
            switch_rate = transpose(v) * gradient
            proposedSwitchIntensity = a
            if proposedSwitchIntensity < switch_rate - tolerance
                println("ERROR: Switching rate exceeds bound.")
                println(" simulated rate: ", proposedSwitchIntensity)
                println(" actual switching rate: ", switch_rate)
                error("Switching rate exceeds bound.")
            end
            if rand() * proposedSwitchIntensity <= switch_rate
                # switch i-th component
                v = reflect(gradient,v)
                a = -switch_rate
                b = transpose(v) * Q * v
                updateSkeleton = true
                accepted_switches += 1
            else
                a = switch_rate
                updateSkeleton = false
                rejected_switches += 1
            end
            # update time to refresh
            Δt_refresh = Δt_refresh - Δt_switch_proposed
        elseif !finished
            # so we refresh
            updateSkeleton = true
            v = randn(dim)
            a = transpose(v) * gradient
            b = transpose(v) * Q * v

            # update time to refresh
            Δt_refresh = -log(rand())/refresh_rate;
        end

        if updateSkeleton
            push!(x_skeleton, x)
            push!(v_skeleton, v)
            push!(t_skeleton, t)
            updateSkeleton = false
        end
        Δt_switch_proposed = switchingtime(a,b)
    end
    println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    println("number of proposed switches: ", accepted_switches + rejected_switches)
    return (t_skeleton, x_skeleton, v_skeleton)

end


function BPS_subsampling(∇E_ss::Function, hessian_bound_ss::Float64, n_observations::Int, T::Float64, x_ref::Vector{Float64} = Vector{Float64}(undef,0), ∇E_ref::Vector{Float64} = Vector{Float64}(undef,0); x_init::Vector{Float64} = Vector{Float64}(undef,0), v_init::Vector{Int} = Vector{Int}(undef,0), refresh_rate::Float64 = 1.0)
# BPS with subsampling and control variates
# grad_E_ss(l,x) gives the ∇ E^l(x), where 1/N ∑_{l=1}^N E^l(x) = E(x), the full potential function
# Q is a symmetric matrix such that - Q ⪯ ∇^2 E^l(x) ⪯ Q for all l,x


    if (length(x_ref) == 0 || length(∇E_ref) == 0)
        controlvariates = false;
        error("BPS_Subampling without control variates currently not supported")
    else
        controlvariates = true;
    end

    dim = length(x_ref)
    if length(x_init) == 0
        if controlvariates
            x_init = x_ref
        else
            x_init = zeros(dim)
        end
    end
    if length(v_init) == 0
        v_init = randn(dim)
    end

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
    b = hessian_bound_ss * dot(v,v)
    a = dot(v, ∇E_ref) + sqrt(b) * sqrt(hessian_bound_ss) * sqrt(dot(x-x_ref,x-x_ref))
    Δt_switch_proposed = switchingtime(a,b)

    if (refresh_rate == 0.0)
        Δt_refresh = Inf
    else
        Δt_refresh = -log(rand())/refresh_rate
    end

    while (!finished)
        # proposedSwitchIntensity = phaseSpaceNorm^2 * M1/2 + phaseSpaceNorm * M2;
        Δt = min(Δt_switch_proposed,Δt_refresh);
        if t + Δt > T
            Δt = T - t
            finished = true
            updateSkeleton = true
        end
        x = x + v * Δt; # O(d)
        t = t + Δt;
        a = a + b * Δt; # O(d)

        if (!finished && Δt_switch_proposed < Δt_refresh)
            j = rand(1:n_observations)
            ∇E_est = ∇E_ref + ∇E_ss(j,x) - ∇E_ss(j,x_ref)
            switch_rate = dot(v, ∇E_est)
            proposal_intensity = a
            if proposal_intensity < switch_rate
                println("ERROR: Switching rate exceeds bound.")
                println(" simulated rate: ", proposal_intensity)
                println(" actual switching rate: ", switch_rate)
                error("Switching rate exceeds bound.")
            end
            if rand() * proposal_intensity <= switch_rate
                # reflect
                v = v - 2 * dot(v, ∇E_est)/dot(∇E_est,∇E_est)*∇E_est
                updateSkeleton = true
                accepted_switches += 1
            else
                updateSkeleton = false
                rejected_switches += 1
            end
            # update refreshment time and switching time bound
            Δt_refresh = Δt_refresh - Δt_switch_proposed
        elseif !finished
            # so we refresh
            updateSkeleton = true
            v = randn(dim)
            # update upcoming event times
            Δt_refresh = -log(rand())/refresh_rate;
        end

        if updateSkeleton
            push!(x_skeleton, x)
            push!(v_skeleton, v)
            push!(t_skeleton, t)
            updateSkeleton = false
            b = hessian_bound_ss * dot(v,v) # norm v changes so we update this
        end

        a = dot(v, ∇E_ref) + sqrt(b) * sqrt(hessian_bound_ss) * sqrt(dot(x-x_ref,x-x_ref))
        Δt_switch_proposed = switchingtime(a,b)

    end
    println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    println("number of proposed switches: ", accepted_switches + rejected_switches)
    return (t_skeleton, x_skeleton, v_skeleton)

end

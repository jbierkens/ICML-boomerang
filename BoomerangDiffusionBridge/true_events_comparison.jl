include("./ZZDiffusionBridge/scripts/examples/sin.jl")
include("./boomerang.jl")


function boomerang_count_ind_ref(α::Float64, T::Float64, L::Int64, u::Float64, v::Float64, clock::Float64)
    N = 2^(L+1)-1
    num_events = fill(0, N)
    ξ = zeros(N)
    θ = randn(N)
    t = 0.0
    c = 0.01
    ϕ = generate(L, T)
    #initialize quantities
    ∇Utilde = zeros(N)
    [∇U_tilde_ind!(∇Utilde, i, ξ, ϕ, α, L, T, u, v) for i in 1:N]
    ∇Ubar = ∇U_bar(α, ϕ, L, T)# does not depend on x, v so precompile
    λ_bar =  [λbar_ind(ξ[i], θ[i], ∇Ubar[i]) for i in 1:N] #vector
    τ = event_λ_const.(λ_bar) #vector
    τ_ref = [event_λ_const(c) for i in 1:N] #vector
    τ0, i0 = findmin(τ)
    τ_ref0, i_ref0 = findmin(τ_ref)
    num_events[i0] += 1
    while t < clock
        if τ_ref0 < τ0
            #println("STEP: refreshment with τ_ref = ", τ_ref)
            ξ, θ =  boomerang_traj(ξ, θ, τ_ref0)
            t += τ_ref0
            θ[i_ref0] = randn()
            ### RESCALE
            #τ = rescale!(i_ref0, N, τ_ref0, τ_ref)
            for i in 1:(i_ref0 - 1)
                τ_ref[i] -= τ_ref0
            end
            for i in  (i_ref0 + 1): N
                τ_ref[i] -=  τ_ref0
            end
            τ .-= τ_ref0
            ## Draw new time
            λ_bar[i_ref0] =  λbar_ind(ξ[i_ref0], θ[i_ref0], ∇Ubar[i_ref0])
            τ[i_ref0] = event_λ_const(λ_bar[i_ref0])
            τ_ref[i_ref0] = event_λ_const(c)
            num_events[i_ref0] += 1
        else
            ξ, θ =  boomerang_traj(ξ, θ, τ0)
            t +=  τ0
            τ_ref .-= τ0
            ∇U_tilde_ind!(∇Utilde, i0, ξ, ϕ, α, L, T, u, v)
            acc_ratio = max(∇Utilde[i0]*θ[i0], 0)/λ_bar[i0] #max not necessary
            if acc_ratio > rand()
                θ[i0] = -θ[i0]
                λ_bar[i0] = λbar_ind(ξ[i0], θ[i0], ∇Ubar[i0])
                τ[i0] = event_λ_const(λ_bar[i0])
                num_events[i0] += 1
            else
                τ[i0] = event_λ_const(λ_bar[i0])
            end
            for i in 1:(i0-1)
                τ[i] -= τ0
            end
            for i in  (i0 + 1): N
                τ[i] -=  τ0
            end
        end
        τ0, i0 = findmin(τ)
        τ_ref0 , i_ref0 = findmin(τ_ref)
    end
    return  num_events
end



#### SIN DIFFUSION α = 0.5

function run_all()
    Random.seed!(0)
    L = 10
    ξ =fill(0.0, 2<<L - 1)
    θ = fill(sqrt(2)/sqrt(π), 2<<L - 1)
    Random.seed!(0)
    T = 50.0
    clock = 2000.0
    α = 0.5#sin
    u = - Float64(π)
    v = 3*Float64(π)
    X = SinSDE(α, L, T)
    xx_count = boomerang_count_ind_ref(α, T, L, u, v, clock)
    zz_count = zz_sampler_count(X, T, L, u, v, clock, ξ, θ)
    return  zz_count, xx_count
end
zz_count, xx_count = run_all()

#####################BROWNIAN MOTION α = 0

function run_all()
    Random.seed!(0)
    L = 10
    ξ =fill(0.0, 2<<L - 1)
    θ = fill(sqrt(2)/sqrt(π), 2<<L - 1)
    Random.seed!(0)
    T = 50.0
    clock = 2000.0
    α = 0.0#sin
    u = - Float64(π)
    v = 3*Float64(π)
    X = SinSDE(α, L, T)
    zz_count = zz_sampler_count(X, T, L, u, v, clock, ξ, θ)
    xx_count = boomerang_count_ind_ref(α, T, L, u, v, clock)
    return  zz_count, xx_count
end
zz_count_bb, xx_count_bb = run_all()



function average_event(L, count)
    if length(count) < 2^(L+1) - 1
        error("L too large")
    end
    ub = 2^(L+1)-1
    lb = 2^(L)
    ave = sum(count[lb:ub])/length(count[lb:ub])
end



L = 10
xx_ave_bb = [average_event(i, xx_count_bb) for i in 0:L]
zz_ave_bb = [average_event(i, zz_count_bb) for i in 0:L]
xx_ave = [average_event(i, xx_count) for i in 0:L]
zz_ave = [average_event(i, zz_count) for i in 0:L]

using Plots
p = plot(0:L, [log.(zz_ave), log.(xx_ave)],
    xlims = (0:10),
    xticks = 0:1:10,
    xlabel = :Levels,
    line=(:solid, 0.5),
    c = [:blue :red],
    lab = ["ZZ, alpha = 0.7", "Boom, alpha = 0.7"]
    )

plot!(p, 0:L, log.(zz_ave_bb), lab = "ZZ, alpha = 0.0", c = :blue, line=(:dash, 0.5))
plot!(p, 0:L, log.(xx_ave_bb), lab = "Boom, alpha = 0.0",  c = :red, line=(:dash, 0.5))
#savefig("./boomerang/events_comparison/comparison1.pdf")

include("./boomerang.jl")
function boomerang_ind_refresh(T::Float64, L::Int64, u::Float64, v::Float64, clock::Float64)
    clearconsole()
    N = 2^(L+1)-1
    ξ = zeros(N)
    θ = randn(N)
    t = 0.0
    α = 1.0
    c = 0.05
    ϕ = generate(L, T)
    #initialize quantities
    ∇Utilde = zeros(N)
    [∇U_tilde_ind!(∇Utilde, i, ξ, ϕ, α, L, T, u, v) for i in 1:N]
    ∇Ubar = ∇U_bar(α, ϕ, L, T)# does not depend on x, v so precompile
    λ_bar =  [λbar_ind(ξ[i], θ[i], ∇Ubar[i]) for i in 1:N] #vector
    τ = event_λ_const.(λ_bar) #vector
    τ_ref = [event_λ_const(c) for i in 1:N] #vector
    p = sortperm(τ)
    p_ref   = sortperm(τ_ref)
    Ξ = [Skeleton2(copy(ξ), copy(θ), t)]
    τ0, i0 = τ[p[1]], p[1]
    τ_ref0, i_ref0 = τ_ref[p_ref[1]], p_ref[1]
    while t < clock
        if τ_ref0 < τ0
            ξ, θ =  boomerang_traj(ξ, θ, τ_ref0)
            t += τ_ref0
            θ[i_ref0] = randn()
            push!(Ξ, (Skeleton2(copy(ξ), copy(θ), t)))
            for i in 1:(i_ref0 - 1)
                τ_ref[i] -= τ_ref0
            end
            for i in  (i_ref0 + 1): N
                τ_ref[i] -=  τ_ref0
            end
            τ .-= τ_ref0
            λ_bar[i_ref0] =  λbar_ind(ξ[i_ref0], θ[i_ref0], ∇Ubar[i_ref0])
            τ[i_ref0] = event_λ_const(λ_bar[i_ref0])
            τ_ref[i_ref0] = event_λ_const(c)
            first_event_ordering(τ_ref, p_ref)
            p = sortperm(τ)
        else
            ξ, θ =  boomerang_traj(ξ, θ, τ0)
            t +=  τ0
            τ_ref .-= τ0
            ∇U_tilde_ind!(∇Utilde, i0, ξ, ϕ, α, L, T, u, v)
            acc_ratio = ∇Utilde[i0]*θ[i0]/λ_bar[i0]
            if acc_ratio > rand()
                θ[i0] = -θ[i0]
                push!(Ξ, (Skeleton2(copy(ξ), copy(θ), t)))
                λ_bar[i0] = λbar_ind(ξ[i0], θ[i0], ∇Ubar[i0])
                τ[i0] = event_λ_const(λ_bar[i0])
            else
                τ[i0] = event_λ_const(λ_bar[i0])
            end
            for i in 1:(i0-1)
                τ[i] -= τ0
            end
            for i in  (i0 + 1): N
                τ[i] -=  τ0
            end
            first_event_ordering(τ, p)
        end
        τ0, i0 = τ[p[1]], p[1]
        τ_ref0 , i_ref0 =  τ_ref[p_ref[1]], p_ref[1]
    end
    return Ξ
end

Random.seed!(0)
T = 50.0
u = Float64(-π)
v =  Float64(3π)
L = 6
clock = 20000.0
Ξ = boomerang_ind_refresh(T, L, u, v, clock)

function find_boomerang_coordinates(Ξ, t)
    tt = [Ξ[i].t for i in 1:length(Ξ)]
    i = searchsortedfirst(tt, t)
    #i = findfirst(x -> x>t ,x in tt)
    ξ, _ = boomerang_traj(Ξ[i-1].ξ, Ξ[i-1].θ, t - Ξ[i-1].t)
    return ξ
end


function plot_boomerang(Ξ, b, L, T, u, v)
    p = Plots.plot(leg = false, colorbar = true)
    dt = range(0.0, stop=T, length=2<<(L) + 1)
    N = length(b)
    P = []
    for i in b
        ξ_interp = find_boomerang_coordinates(Ξ, i)
        dx = fs_expansion(ξ_interp, u, v, L, T)
        push!(P, dx)
    end
    p = plot(dt, P, color = :darkrainbow, line_z = (1:N)', linewidth=0.01, alpha = 0.3, leg = false, colorbar = true)
    hline!(p, [n*π for n in -1:2:3], color = :blue)
    display(p)
    return p
end



using Plots
using LaTeXStrings
b = 19:20.0:clock-.1
length(b)
p = plot_boomerang(Ξ, b, L, T, u, v)
xaxis!(p, L"t")
yaxis!(p, L"X_t")
ylims!((-2π,4π))


#savefig("boomerang/sin_boom10_prova.pdf")

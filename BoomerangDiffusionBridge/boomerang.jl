include("./ZZDiffusionBridge/src/ZZDiffusionBridge.jl")


"""
        struct Skeleton2
            ξ::Vector{Float64} #does saving take time?
            θ::Vector{Float64}
            t::Float64
        end

structure for saving the output of the process. `ξ` position, `θ` velocity, `t` time
"""
struct Skeleton2
    ξ::Vector{Float64} #does saving take time?
    θ::Vector{Float64}
    t::Float64
end
"""
    boomerang_traj(ξ, θ, t)

trajectories which in this case are circles in (ξ, θ)
"""
function boomerang_traj(ξ::Vector{Float64}, θ::Vector{Float64}, t::Float64)
    ξ_new = ξ.*cos(t) + θ.*sin(t)
    θ = θ.*cos(t) - ξ.*sin(t)
    return ξ_new, θ
end


"""
     ∇U_bar(α::Float64, ϕ::Vector{Fs}, L::Int64, T::Float64)

computes upper bound of the gradient of the potential function(-log(density)).
Does not depend on x --> pre-compile
"""
function ∇U_bar(α::Float64, ϕ::Vector{Fs}, L::Int64, T::Float64)
    return [0.5*ϕ[n].δ*(α^2 + α) for n in 1:length(ϕ)]
end


"""
         fs_expansion(t::Float64, ξ::Vector{Float64}, ϕ::Vector{Fs}, u::Float64, v::Float64, L::Int64, T::Float64, n = i -> 2^-(1 + i/2))

Local expansion: find value of the process (Piecewise linear) for any t ∈ [0, T]
does not evaluate the  whole path, but just the points needed for t
"""
function fs_expansion(t::Float64, ξ::Vector{Float64}, ϕ::Vector{Fs}, u::Float64, v::Float64, L::Int64, T::Float64, n = i -> 2^-(1 + i/2))
        dt = 0:T/(2<<L):T
        k = (searchsortedfirst(dt, t) - 1)
        j0 =  Int(ceil(k/2))-1
        n0 = Faber(L, j0)
        if k % 2 != 0
                return interpolate([dt[k], dt[k + 1]], fs_expansion(ϕ[n0], ξ, u, v, L, T)[1:2], t)
        else
                return interpolate([dt[k], dt[k + 1]], fs_expansion(ϕ[n0], ξ, u, v,  L, T)[2:3], t)
        end
end



"""
    event_λref(c::Float64)
Draw waiting time from homogeneous Poisson rate c. Used for refreshments
"""
function event_λ_const(c::Float64)
    return -log(rand())/c
end



"""
try new bounds
"""
function λbar_ind(ξ::Float64, θ::Float64, ∇Ubar_i::Float64 )
    θ_ubs = sqrt(ξ*ξ + θ*θ)
    return θ_ubs*∇Ubar_i
end


"""
    Λ(ϕ_n::Fs, t::Float64)
fast way to evaluate a basis function `ϕ_n` at a point t (inside its support)
"""
function  Λ(ϕ_n::Fs, t::Float64)
    ϕm = (ϕ_n.lb + ϕ_n.ub)*0.5
    if t < ϕm
        (t - ϕ_n.lb)*ϕ_n.supr/(ϕm - ϕ_n.lb)
    else
        (ϕ_n.ub - t)*ϕ_n.supr/(ϕ_n.ub - ϕm)
    end
end


"""
    ∇U_tilde_ind(ξ::Vector{Float64}, ϕ::Vector{Fs}, α::Float64, L::Int64, T::Float64, u::Float64, v::Float64)

accept reject time drwan from upper bound λbar relative to the coefficient `n`
of model `SinSDE` starting at `u` and ending at `v`
"""
function ∇U_tilde_ind!(∇U::Vector{Float64}, i0::Int64, ξ::Vector{Float64}, ϕ::Vector{Fs}, α::Float64, L::Int64, T::Float64, u::Float64, v::Float64)
        t = MCintegration(ϕ[i0])
        XX = fs_expansion(t, ξ, ϕ, u, v, L, T)
        ϕ_t = Λ(ϕ[i0], t)
        ∇U[i0] = 0.5*ϕ[i0].range*ϕ_t*(α*α*sin(2.0*(XX)) - α*sin(XX))
end


"""
Takes a quasi-oredered vector `v` with ordering `p` (ordered from the second element of v).
Update the ordering such that also the first element is ordered as above. This function replaces
findmin() which should be computationally more expensives.
#TODO generaliz for the ith compoment
"""
function first_event_ordering(v::Vector{Float64}, p::Vector{Int64})
    p1 = p[1]
    i_new = searchsortedfirst(v[p[2:end]], v[p1])
    if i_new == 1
        return
    else
        p[1:i_new - 1] = p[2: i_new]
        p[i_new] = p1
    end
end

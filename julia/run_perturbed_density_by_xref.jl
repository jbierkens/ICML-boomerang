include("boomerang.jl")
include("bps.jl")
using DataFrames
using LaTeXStrings
using Plots
using CSV
function addEntry!(df, sampler, dim, ess, ess_log_density, runtime, time_horizon, refresh_rate, σ_squared, x_ref_multiplier)
    push!(
        df,
        Dict(
            :sampler => sampler,
            :dimension => dim,
            :ess_median => median(ess),
            :ess_avg => mean(ess),
            :ess_025 => quantile(ess,0.25),
            :ess_075 => quantile(ess,0.75),
            :ess_min => minimum(ess),
            :ess_max => maximum(ess),
            :ess_log_density => ess_log_density,
            :runtime => runtime,
            :time_horizon => time_horizon,
            :refresh_rate => refresh_rate,
            :σ_squared => σ_squared,
            :x_ref_multiplier => x_ref_multiplier
        ),
    )
end

function runall(numb_of_exp, over_dimension, over_xref, time_horizon)
    df = DataFrame(
        sampler = String[],
        dimension = Int[],
        time_horizon = Float64[],
        refresh_rate = Float64[],
        ess_median = Float64[],
        ess_avg = Float64[],
        ess_025 = Float64[],
        ess_075 = Float64[],
        ess_min = Float64[],
        ess_max = Float64[],
        ess_log_density = Float64[],
        runtime = Float64[],
        σ_squared = Float64[],
        x_ref_multiplier = Float64[]
    );
    bps_time_horizon = time_horizon #TOCHANGE
    E(x) = norm(x)^2/2
    ∇E(x) = x
    refresh_rate = 0.1
    bps_refresh_rate = 1.0
    n_batches = 50 # for batch means
    σ_squared = 1.0
    U_hessian_bound = abs(1-1/σ_squared)
    for dim in over_dimension
        Q = LinearAlgebra.symmetric(Matrix{Float64}(I, dim, dim),:U)
        for _ in 1:numb_of_exp
            for x_ref_multiplier in over_xref
                x_ref = x_ref_multiplier .* ones(Float64,dim)
                ∇E_ref = ∇E(x_ref)
                Σ_inv = 1/σ_squared * Matrix{Float64}(I, dim, dim)
                runtime = @elapsed (t_skeleton, x_skeleton,v_skeleton) = Boomerang(
                        ∇E,
                        U_hessian_bound,
                        time_horizon,
                        refresh_rate,
                        x_ref,
                        Σ_inv)

                ess = ess_pdmp_components(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        ellipticQ = true,
                        x_ref = x_ref,
                        n_batches = n_batches
                        )
                        ess_log_density = ess_pdmp_squared_radius(t_skeleton, x_skeleton, v_skeleton, n_batches = n_batches, ellipticQ = true, x_ref= x_ref, true_mean = Float64(dim), true_variance = Float64(2 * dim))

                addEntry!(df, "Boomerang", dim, ess, ess_log_density, runtime, time_horizon, refresh_rate, σ_squared, x_ref_multiplier)
                # plot_pdmp(t_skeleton,x_skeleton,v_skeleton, name=string("Boomerang-σ^2-",σ_squared, "-dim-", dim),ellipticQ=true, x_ref= x_ref, n_samples=100000)
            end
            runtime = @elapsed (t_skeleton, x_skeleton,v_skeleton) = BPS(∇E, Q, bps_time_horizon, refresh_rate = bps_refresh_rate)
            ess = ess_pdmp_components(
                t_skeleton,
                x_skeleton,
                v_skeleton,
                ellipticQ = false, #TOCHANGE
                n_batches = n_batches
            )
            ess_log_density = ess_pdmp_squared_radius(t_skeleton, x_skeleton, v_skeleton, n_batches = n_batches, ellipticQ = false, true_mean = Float64(dim), true_variance = Float64(2 * dim))
            addEntry!(df, "BPS", dim, ess, ess_log_density, runtime, time_horizon, refresh_rate, NaN, NaN)
        end
    end
    df
end

numb_of_exp = 20
over_dimension = [1,10,100]
over_xref = 0.0:0.2:2.0
time_horizon = 10000.0
df = runall(numb_of_exp, over_dimension, over_xref, time_horizon)
CSV.write("data-by-xref.csv", df)

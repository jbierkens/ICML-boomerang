include("logistic.jl")
include("boomerang.jl")
include("zigzag.jl")
include("bps.jl")
include("pdmp.jl")
include("mala.jl")
include("asvar.jl")

using StatsBase # for autocor in plot_pdmp
using DataFrames, StatsPlots # to collect and plot results
using Statistics

n_batches = 50 # number of batches in batch means both for ess_pdmp as ess.


## SETTINGS FOR LONG, SUBSAMPLING FOCUSED RUN
time_horizon = 10000.0; # 10000.0
refresh_rate = 0.1;
reresh_rate_bps = 1.0;
n_experiments = 20;
observations = 10 .^ (2:5); # 10 .^ (2:5)
dims = 2;


function addEntry!(df, sampler, dim, n_observations, ess, ess_squared_radius, runtime, preprocessing_time, time_horizon, refresh_rate)

    push!(
        df,
        Dict(
            :sampler => sampler,
            :dimension => dim,
            :observations => n_observations,
            :ess_median => median(ess),
            :ess_avg => mean(ess),
            :ess_025 => if(any(isnan.(ess))) NaN else quantile(ess,0.25) end,
            :ess_075 => if(any(isnan.(ess))) NaN else quantile(ess,0.75) end,
            :ess_min => minimum(ess),
            :ess_max => maximum(ess),
            :ess_squared_radius => ess_squared_radius,
            :runtime => runtime,
            :preprocessing_time => preprocessing_time,
            :time_horizon => time_horizon,
            :refresh_rate => refresh_rate,
        ),
    )
end



function experiment!(
    df::DataFrame,
    dim::Int,
    n_observations::Int,
    time_horizon::Float64,
    refresh_rate::Float64;
    n_experiments = 1,
    plot = false,
    verbose = true,
    by_dimension = false
)

    if verbose
        println("number of observations = ", n_observations)
        println("dimension = ", dim)
    end

    parameter = randn(dim)

    (Y, Z) = generateLogisticData(parameter, n_observations)

    Q = LogisticDominatingMatrix(Y)
    M1 = opnorm(Q)
    M1_factorized = vec([norm(Q[:,j], 2) for j in 1:size(Q)[1]])
    Q_zz = LogisticEntrywiseDominatingMatrix(Y)

    (E, ∇E, h_E, ∂E) = constructLogisticEnergy(Y, Z)

    preprocessing_time = @elapsed((x_ref, Σ_inv) = preprocess(E, ∇E, h_E, dim))

    (
     ∂E_ss,
     Q_zz_ss,
     ∇E_ss,
     h_E_ss,
     hessian_bound_ss
    ) = LogisticSubsamplingTools(Y, Z)

    Σ = inv(Σ_inv)
    Σ_diag_inv = inv(Diagonal(Σ)) # we want to take the inverse variances (but could instead consider Diagonal(Σ_inv) TODO)
    ∇E_ref = ∇E(x_ref)
    # in order not to have outrageous runtimes we rescale the time horizon of algorithms for which velocity does not depend on Σ
    # to Tr(Σ)^(1/2)* T. This is e.g. for Bouncy, ZigZag
    time_horizon_rescaled = time_horizon * sqrt(sum(diag(Σ)))

    for i = 1:n_experiments
        if (verbose && n_experiments > 1)
            println("experiment number ", i)
        end

        if !by_dimension # in which case running subsampled algorithms is feasible
            # run subsampled algorithms
            if verbose
                println("running Boomerang w/subsampling...")
            end
            runtime = @elapsed ((
                t_skeleton,
                x_skeleton,
                v_skeleton,
            ) = BoomerangSubsampling(
                ∇E_ss,
                h_E_ss,
                hessian_bound_ss,
                n_observations,
                time_horizon,
                refresh_rate_boomerang,
                x_ref,
                ∇E_ref,
                Σ_inv,
            ))

            ess = ess_pdmp_components(
                t_skeleton,
                x_skeleton,
                v_skeleton,
                ellipticQ = true,
                x_ref = x_ref,
                n_batches = n_batches
            )
            ess_squared_radius = ess_pdmp_squared_radius(t_skeleton,x_skeleton,v_skeleton,n_batches = n_batches, ellipticQ=true,x_ref=x_ref)

            if (plot && i == 1)
                plot_pdmp(
                    t_skeleton,
                    x_skeleton,
                    v_skeleton,
                    ellipticQ = true,
                    x_ref = x_ref,
                    name = "boomerang-subsampling",
                )
            end
            addEntry!(df, "Boomerang w/subsampling", dim, n_observations, ess, ess_squared_radius, runtime, preprocessing_time, time_horizon, refresh_rate_boomerang)

            if verbose
                println("running ZigZag w/subsampling...")
            end
            runtime = @elapsed ((
                t_skeleton,
                x_skeleton,
                v_skeleton,
            ) = ZigZagSubsampling(
                ∂E_ss,
                Q_zz_ss,
                n_observations,
                time_horizon_rescaled,
                x_ref,
                ∇E_ref,
            ))
            ess = ess_pdmp_components(
                t_skeleton,
                x_skeleton,
                v_skeleton,
                n_batches = n_batches
            )
            ess_squared_radius = ess_pdmp_squared_radius(t_skeleton,x_skeleton,v_skeleton, n_batches = n_batches, ellipticQ=false)

            if (plot && i == 1)
                plot_pdmp(
                    t_skeleton,
                    x_skeleton,
                    v_skeleton,
                    name = "zigzag-subsampling",
                )
            end
            addEntry!(df, "ZigZag w/subsampling", dim, n_observations, ess, ess_squared_radius, runtime, preprocessing_time, time_horizon, 0.0)

            if verbose
                println("running BPS w/subsampling...")
            end
            runtime = @elapsed ((
                t_skeleton,
                x_skeleton,
                v_skeleton,
            ) = BPS_subsampling(
                ∇E_ss,
                hessian_bound_ss,
                n_observations,
                time_horizon_rescaled,
                x_ref,
                ∇E_ref,
                refresh_rate = refresh_rate_bps,
            ))
            ess = ess_pdmp_components(
                t_skeleton,
                x_skeleton,
                v_skeleton,
                n_batches = n_batches
            )
            ess_squared_radius = ess_pdmp_squared_radius(t_skeleton,x_skeleton,v_skeleton, n_batches = n_batches, ellipticQ=false)

            if (plot && i == 1)
                plot_pdmp(
                    t_skeleton,
                    x_skeleton,
                    v_skeleton,
                    name = "BPS w/subsampling",
                )
            end
            addEntry!(df, "BPS w/subsampling", dim, n_observations, ess, ess_squared_radius, runtime, preprocessing_time, time_horizon, refresh_rate_bps)

        end

        if verbose
            println("running Boomerang...")
        end
        runtime = @elapsed ((
            t_skeleton,
            x_skeleton,
            v_skeleton,
        ) = Boomerang(∇E, M1, time_horizon, refresh_rate_boomerang, x_ref, Σ_inv))

        ess = ess_pdmp_components(
            t_skeleton,
            x_skeleton,
            v_skeleton,
            ellipticQ = true,
            x_ref = x_ref,
            n_batches = n_batches
        )
        ess_squared_radius = ess_pdmp_squared_radius(t_skeleton,x_skeleton,v_skeleton, n_batches = n_batches, ellipticQ=true, x_ref=x_ref)

        if (plot && i == 1)
            plot_pdmp(
                t_skeleton,
                x_skeleton,
                v_skeleton,
                ellipticQ = true,
                x_ref = x_ref,
                name = "boomerang",
            )
        end

        addEntry!(df, "Boomerang", dim, n_observations, ess, ess_squared_radius, runtime, preprocessing_time, time_horizon, refresh_rate_boomerang)

        # if verbose
        #     println("running Boomerang (diagonal)...")
        # end
        # runtime = @elapsed ((
        #     t_skeleton_boomerang_diag,
        #     x_skeleton_boomerang_diag,
        #     v_skeleton_boomerang_diag,
        # ) = Boomerang(∇E, M1, time_horizon, refresh_rate, x_ref, Σ_diag_inv))
        #
        # ess = ess_pdmp_components(
        #     t_skeleton_boomerang_diag,
        #     x_skeleton_boomerang_diag,
        #     v_skeleton_boomerang_diag,
        #     ellipticQ = true,
        #     x_ref = x_ref,
        #     n_batches = n_batches
        # )
        # if (plot && i == 1)
        #     plot_pdmp(
        #         t_skeleton_boomerang_diag,
        #         x_skeleton_boomerang_diag,
        #         v_skeleton_boomerang_diag,
        #         ellipticQ = true,
        #         x_ref = x_ref,
        #         name = "boomerang-diagonal",
        #     )
        # end
        # addEntry!(df, "Boomerang (diagonal)", dim, n_observations, ess, runtime, preprocessing_time, time_horizon, refresh_rate)

        if verbose
            println("running ZigZag...")
        end
        runtime = @elapsed ((
            t_skeleton,
            x_skeleton,
            v_skeleton,
        ) = ZigZag(∂E, Q_zz, time_horizon_rescaled))

        ess = ess_pdmp_components(t_skeleton, x_skeleton, v_skeleton,n_batches = n_batches)

        ess_squared_radius = ess_pdmp_squared_radius(t_skeleton,x_skeleton,v_skeleton, n_batches = n_batches, ellipticQ=false)

        if (plot && i == 1)
            plot_pdmp(
                t_skeleton,
                x_skeleton,
                v_skeleton,
                name = "zigzag",
            )
        end
        addEntry!(df, "ZigZag", dim, n_observations, ess, ess_squared_radius, runtime, 0.0, time_horizon, 0.0)

        if verbose
            println("running BPS...")
        end
        runtime = @elapsed ((
            t_skeleton,
            x_skeleton,
            v_skeleton,
        ) = BPS(∇E, Q, time_horizon_rescaled, refresh_rate = refresh_rate_bps))
        ess = ess_pdmp_components(t_skeleton, x_skeleton, v_skeleton,n_batches = n_batches)
        ess_squared_radius = ess_pdmp_squared_radius(t_skeleton,x_skeleton,v_skeleton, n_batches = n_batches, ellipticQ=false)

        if (plot && i == 1)
            plot_pdmp(
                t_skeleton,
                x_skeleton,
                v_skeleton,
                name = "BPS",
            )
        end

        addEntry!(df, "BPS", dim, n_observations, ess, ess_squared_radius, runtime, 0.0, time_horizon, refresh_rate_bps)

        if verbose
            println("running MALA...")
        end
        mala_stepsize = 1.5 * sum(diag(Σ))/dim^(4/3) # typical size is (tr Σ/dim), optimal scaling is dim^(-1/3)
        mala_iterations = Int(round(time_horizon))
        runtime = @elapsed (x_mala = MALA(E, ∇E, mala_stepsize, dim, mala_iterations))
        ess = ESS(x_mala,n_batches = n_batches)
        ess_squared_radius = ESS(sum(x_mala.^2,dims=1),n_batches=n_batches)[1]
        addEntry!(df, "MALA", dim, n_observations, ess, ess_squared_radius, runtime, 0.0, time_horizon, NaN)
    end
end

function postprocess!(df::DataFrame)
    df[!, :avg_ess_per_sec] = df[!, :ess_avg] ./ (df[!, :runtime] + df[!,:preprocessing_time])
    df[!, :min_ess_per_sec] = df[!, :ess_min] ./ (df[!, :runtime] + df[!,:preprocessing_time])
    df[!, :squared_radius_ess_per_sec] = df[!, :ess_squared_radius] ./ (df[!, :runtime] + df[!,:preprocessing_time])

end

## SETTINGS FOR LONG, DIMENSION DEPENDENT RUN
# time_horizon = 10000.0;
# refresh_rate = 0.1;
# n_experiments = 20;
# observations = 10 .^ 3;
# dims = 2 .^ (1:5);


function run_test!(
    df,
    observations,
    dims,
    n_experiments,
    refresh_rate,
    time_horizon,
    verbose = true,
)
    if length(dims) > 1
        by_dimension = true
    else
        by_dimension = false
    end
    for n_observations in observations
        println("n_observations = ", n_observations)
        for dim in dims
            println("..dimension = ", dim)
            for i = 1:n_experiments
                println("....experiment ", i)
                experiment!(
                    df,
                    dim,
                    n_observations,
                    time_horizon,
                    refresh_rate,
                    plot = false,
                    n_experiments = 1,
                    verbose = true,
                    by_dimension = by_dimension
                )
            end
        end
    end
end

# using ProfileView

# a quick run to compile everything

df = DataFrame(
    sampler = String[],
    dimension = Int[],
    observations = Int[],
    time_horizon = Float64[],
    refresh_rate = Float64[],
    ess_median = Float64[],
    ess_avg = Float64[],
    ess_025 = Float64[],
    ess_075 = Float64[],
    ess_min = Float64[],
    ess_max = Float64[],
    ess_squared_radius = Float64[],
    runtime = Float64[],
    preprocessing_time = Float64[],
);

run_test!(df, 100, 2, 1, refresh_rate, time_horizon)

# Profile.clear()

df = DataFrame(
    sampler = String[],
    dimension = Int[],
    observations = Int[],
    time_horizon = Float64[],
    refresh_rate = Float64[],
    ess_median = Float64[],
    ess_avg = Float64[],
    ess_025 = Float64[],
    ess_075 = Float64[],
    ess_min = Float64[],
    ess_max = Float64[],
    ess_squared_radius = Float64[],
    runtime = Float64[],
    preprocessing_time = Float64[],
);

run_test!(df, observations, dims, n_experiments, refresh_rate, time_horizon)
# @profview run_test!(df, observations, dims, n_experiments, refresh_rate, time_horizon)
postprocess!(df)

# @df df dotplot(:observations, :avg_ess_per_sec,group=:sampler, xaxis=:log, yaxis=:log)
# savefig("boxplot-by-observations")


# write results to disk
using CSV
CSV.write(string("results-by-observations-in-", dims, "-dimensions-refresh-", refresh_rate, ".csv"), df)

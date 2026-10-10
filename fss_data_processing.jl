using CSV
using DataFrames
using Statistics


function binder_cumulant(m4, m2)
    return 1 - m4/(3*m2^2)
end

sim_results = CSV.read("simulation_results/binder_sim_data.csv", DataFrame)


result = combine(
    groupby(sim_results, [:L, :T]),
    :m2 => mean => :m2_mean,
    :m4 => mean => :m4_mean
)

result.U = binder_cumulant.(result.m4_mean, result.m2_mean)

sort!(result, [:L, :T])

CSV.write("simulation_results/binder_data.csv", result)
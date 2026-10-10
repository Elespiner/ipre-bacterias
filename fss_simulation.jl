using Plots
using Plots.PlotMeasures
using Statistics
using StatsBase
using Base.Threads
using DataFrames
using CSV

function get_energy(lattice, J=1)
    N, M = size(lattice)
    energy = 0.0

    @inbounds @simd for i in 1:N
        for j in 1:M 
            neighbors = 0

            #vecino de arriba
            neighbors += (i > 1) ? lattice[i-1, j] : lattice[N, j]
            # vecino de abajo
            neighbors += (i < N) ? lattice[i+1, j] : lattice[1, j]
            # vecino de la izquierda
            neighbors += (j > 1) ? lattice[i, j-1] : lattice[i, M]
            # vecino de la derecha
            neighbors += (j < M) ? lattice[i, j+1] : lattice[i, 1]

            energy -= J * lattice[i, j] * neighbors
        end
    end

    return energy / 2.0
end

function metropolis(spin_arr::Matrix{T}, iter::Int, T_temp::Real, J=1.0, K=1.0) where {T<:Real}
    spins = copy(spin_arr)
    N, M = size(spins)
    beta = 1.0 / (K * T_temp)
    n_spins = N * M
    total_steps = n_spins * iter

    mag_time_series = zeros(Float64, iter)
    current_net_spin = Float64(sum(spins))

    prob_4J = exp(-beta * 4 * J)
    prob_8J = exp(-beta * 8 * J)

    sample_idx = 1

    @inbounds for t in 1:total_steps
        x = rand(1:N)
        y = rand(1:M)
        s_i = spins[x, y]

        # Condiciones de borde periódicas
        x_prev = (x == 1) ? N : x - 1
        x_next = (x == N) ? 1 : x + 1
        y_prev = (y == 1) ? M : y - 1
        y_next = (y == M) ? 1 : y + 1

        vecinos = spins[x_prev, y] + spins[x_next, y] + spins[x, y_prev] + spins[x, y_next]
        dE = 2 * J * s_i * vecinos

        if dE <= 0
            spins[x, y] = -s_i
            current_net_spin += 2 * (-s_i)
        elseif dE == 4 * J
            if rand() < prob_4J
                spins[x, y] = -s_i
                current_net_spin += 2 * (-s_i)
            end
        elseif dE == 8 * J
            if rand() < prob_8J
                spins[x, y] = -s_i
                current_net_spin += 2 * (-s_i)
            end
        else
            if rand() < exp(-beta * dE)
                spins[x, y] = -s_i
                current_net_spin += 2 * (-s_i)
            end
        end

        # Guardamos solo un dato por MCS
        if t % n_spins == 0
            mag_time_series[sample_idx] = current_net_spin / n_spins
            sample_idx += 1
        end
    end

    return mag_time_series
end

"""
    Realiza varias simulaciones del algoritmo metropolis para el modelo de Ising
    para distintos tamaños L. Genera DataFrame con información sobre m^2 y m^4 para cada
    simulación, tamaño y temperatura
"""
function get_magnetization_sim( Ls::AbstractVector{<:Int},
                                Ts::AbstractVector{<:Real},
                                iter::Int,
                                sims::Int,
                                crit_temp::Real)

    n_rows = length(Ls) * length(Ts) * sims

    results = DataFrame(
        L = zeros(Int, n_rows),
        T = zeros(Float64, n_rows),
        sim = zeros(Int, n_rows),
        m2 = zeros(Float64, n_rows),
        m4 = zeros(Float64, n_rows)
    )

    @assert iter > 1000

    for (j, L) in enumerate(Ls)

        lattice_unif = ones(Int, L, L)
        lattice_rand = rand((-1, 1), L, L)

        Threads.@threads for i in eachindex(Ts) # Recorremos cada temperatura
            T = Ts[i]

            for s in 1:sims # para cada temperatura repetimos sims veces

                # Elegimos la lattice dependiendo de la temperatura
                current_lattice = (T < crit_temp) ? lattice_unif : lattice_rand

                mag_time_series = metropolis(current_lattice, iter, T)

                # Descartamos las primeras 1000 iteraciones
                @views m = mag_time_series[(1000 + 1):end]  

                #idx = LinearIndices((sims, length(Ts), length(Ls)))
                #row = idx[s, i, j]

                row = ((j - 1) * length(Ts) + (i- 1)) * sims + s

                results.L[row] = L
                results.T[row] = T
                results.sim[row] = s
                results.m2[row] = mean(x -> x^2, m)
                results.m4[row] = mean(x -> x^4, m)

            end

        end
    end
    return results
end


Ls = [16, 32, 48, 64]
Ts = range(2.10, 2.44, step=0.005)

results = get_magnetization_sim(Ls , Ts, 8000, 30, 2.269185)

CSV.write("simulation_results/binder_sim_data.csv", results)
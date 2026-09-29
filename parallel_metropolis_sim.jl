using Plots
using Plots.PlotMeasures
using Statistics
using StatsBase
using Base.Threads

#=
    Este scrip ejecuta una simulación del algoritmo metropolis
    y genera un plot con los resultados estadísticos de la magnetización
    del sistema.

    La simulación usa paralelismo por lo que debe ser ejecutado con 
    julia -t auto parallel_metropolis_sim.jl
    puede tardar entre 30 y 60 mins en ejecutarse completo.
=#

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

function metropolis_f(spin_arr::Matrix{<:Real}, times::Int, T::Real, energy::Real, J=1.0, K=1.0)
    spin_arr = copy(spin_arr)
    net_spins = zeros(Float64, times)
    N, M = size(spin_arr)
    beta = 1/(K*T)
    current_net_spin = Float64(sum(spin_arr))

    @inbounds for t in 1:times
        x = rand(1:N)
        y = rand(1:M)
        spin_i = spin_arr[x, y]
        spin_f = -1*spin_i
        E_i = 0
        E_f = 0

        if x > 1
            E_i += -J * spin_i * spin_arr[x-1, y]
            E_f += -J * spin_f * spin_arr[x-1, y]
        elseif x == 1
            E_i += -J * spin_i * spin_arr[N, y]
            E_f += -J * spin_f * spin_arr[N, y]
        end
        if x < N
            E_i += -J * spin_i * spin_arr[x+1, y]
            E_f += -J * spin_f * spin_arr[x+1, y]
        elseif x == N
            E_i += -J * spin_i * spin_arr[1, y]
            E_f += -J * spin_f * spin_arr[1, y]
        end
        if y > 1
            E_i += -J * spin_i * spin_arr[x, y-1]
            E_f += -J * spin_f * spin_arr[x, y-1]
        elseif y == 1
            E_i += -J * spin_i * spin_arr[x, M]
            E_f += -J * spin_f * spin_arr[x, M]
        end
        if y < M
            E_i += -J * spin_i * spin_arr[x, y+1]
            E_f += -J * spin_f * spin_arr[x, y+1]
        elseif y == M
            E_i += -J * spin_i * spin_arr[x, 1]
            E_f += -J * spin_f * spin_arr[x, 1]
        end

        dE = E_f - E_i
        if (dE > 0) && (rand() < exp(-beta*dE))
            spin_arr[x, y] = spin_f
            current_net_spin += 2 * spin_f
        elseif (dE <= 0)
            spin_arr[x, y] = spin_f
            current_net_spin += 2 * spin_f
        end
        
        net_spins[t] = current_net_spin
    end
    return net_spins
end

function metropolis_f_opt(spin_arr::Matrix{T}, iter::Int, T_temp::Real, J=1.0, K=1.0) where {T<:Real}
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

function get_magnetization_sim(lattice_unif::Matrix{<:Real}, 
                                lattice_rand::Matrix{<:Real},
                                Ts::AbstractVector{<:Real},
                                iter::Int,
                                sims::Int,
                                crit_temp::Real)
    N, M = size(lattice_unif)
    num_Ts = length(Ts)
    mag_means = zeros(Float64, num_Ts)
    mag_stds  = zeros(Float64, num_Ts)
    mag_skew  = zeros(Float64, num_Ts)
    mag_kurt  = zeros(Float64, num_Ts)

    n_spins = N * M

    Threads.@threads for i in 1:num_Ts # Recorremos cada temperatura
        T = Ts[i]
        ens_means = zeros(Float64, sims)
        ens_vars  = zeros(Float64, sims)
        ens_skew  = zeros(Float64, sims)
        ens_kurt  = zeros(Float64, sims)

        for s in 1:sims # para cada temperatura repetimos sims veces

            # Elegimos la lattice dependiendo de la temperatura
            current_lattice = (T < crit_temp) ? copy(lattice_unif) : copy(lattice_rand)

            mag_time_series = metropolis_f_opt(current_lattice, iter, T)
            
            
            # Descartamos las primeras 1000 iteraciones
            @views stationary_series = mag_time_series[(1000 + 1):end]

            # obtenemos las métricas para esta simulación
            ens_means[s] = mean(stationary_series)
            ens_vars[s]  = var(stationary_series)
            ens_skew[s]  = skewness(stationary_series)
            ens_kurt[s]  = kurtosis(stationary_series)
        end

        # Sacamos el promedio de las métricas para esta temperatura sobre todas las simulaciones
        mag_means[i] = mean(ens_means)
        mag_stds[i]  = mean(ens_vars)
        mag_skew[i]  = mean(ens_skew)
        mag_kurt[i]  = mean(ens_kurt)
    end

    return mag_means, mag_stds, mag_skew, mag_kurt
end

N = 100
init_random = rand(N, N)

lattice_unif = ones(Int, N, N)

lattice_rand = zeros(Int, N, N)
lattice_rand[init_random .>=0.5] .= 1
lattice_rand[init_random .< 0.5] .= -1

Ts = range(1.42, 3.12, step=0.05)
mag_means, mag_stds, mag_skew, mag_kurt = get_magnetization_sim(lattice_unif, lattice_rand, Ts, 4000, 500, 2.27)


p1 = plot(Ts, mag_means, xlabel="T", ylabel="Mean", marker=:circle)
vline!([2.27], label="T_c", linewidth=2, linestyle=:dash, opacity=0.5)
vspan!([2.22, 2.32], opacity=0.2, label="Critical Region")
p2 = plot(Ts, mag_stds, xlabel="T", ylabel="Variance", marker=:circle)
vline!([2.27], label="T_c", linewidth=2, linestyle=:dash, opacity=0.5)
vspan!([2.22, 2.32], opacity=0.2, label="Critical Region")
p3 = plot(Ts, abs.(mag_skew), xlabel="T", ylabel="Skewness", marker=:circle)
vline!([2.27], label="T_c", linewidth=2, linestyle=:dash, opacity=0.5)
vspan!([2.22, 2.32], opacity=0.2, label="Critical Region")
p4 = plot(Ts, mag_kurt, xlabel="T", ylabel="Kurtosis", marker=:circle)
vline!([2.27], label="T_c", linewidth=2, linestyle=:dash, opacity=0.5)
vspan!([2.22, 2.32], opacity=0.2, label="Critical Region")

p = plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 800), margin=5mm, dpi=300)
savefig(p, "test_results.png")
gui(p)
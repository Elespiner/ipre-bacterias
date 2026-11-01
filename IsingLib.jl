module IsingLib

using Statistics
using StatsBase
using Base.Threads

export get_energy, metropolis_ising_mag, metropolis_spins_energy, wolff_update!, wolff

"""
Obtiene la energía de un sistema del modelo de Ising representado
por una matriz lattice. 
"""
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

"""
Función que ejecuta una simulación de Metropolis con iter Monte Carlo Sweeps
y retorna una serie de tiempo de la magnetizacion
"""
function metropolis_ising_mag(spin_arr::Matrix{T}, iter::Int, T_temp::Real, J=1.0, K=1.0) where {T<:Real}
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
Función que ejecuta una simulacion del metropolis del modelo de Ising
y retorna una seria de tiempo de los spins netos y la energía. 
"""
function metropolis_spins_energy(spin_arr, times, T, energy, J=1, K=1)
    spin_arr = copy(spin_arr)
    net_spins = zeros(times)
    net_energy = zeros(times)
    N, M = size(spin_arr)
    beta = 1/(K*T)
    n_spins = N * M
    total_steps = n_spins * times

    current_net_spin = Float64(sum(spin_arr))

    sample_idx = 1

    @inbounds for t in 1:total_steps
        # Elegimos un punto random en el array
        x = rand(1:N)
        y = rand(1:M)
        spin_i = spin_arr[x, y]
        spin_f = -1*spin_i

        E_i = 0
        E_f = 0

        #aplicamos los cambios en la energía
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

        #decidimos si volteamos el spin
        if (dE > 0) && (rand() < exp(-beta*dE))
            spin_arr[x, y] = spin_f
            energy += dE
            current_net_spin += 2 * spin_f
        elseif (dE <= 0)
            spin_arr[x, y] = spin_f
            energy += dE
            current_net_spin += 2 * spin_f
        end
        

        #guardamos cada MCS
        if t % n_spins == 0
            net_spins[sample_idx] = current_net_spin
            net_energy[sample_idx] = energy 
            sample_idx += 1
        end

    end
    return net_spins, net_energy
end

"""
Construye e invierte un cluster de Wolff.

Devuelve:
- cluster_size: cantidad de espines invertidos
- spin_i: signo original del cluster

marks, stack y cluster_num permiten reutilizar memoria.
"""
function wolff_update!(spins, p, marks, cluster_num, stack)

    N, M = size(spins)

    x = rand(1:N)
    y = rand(1:M)

    spin_i = spins[x, y]

    empty!(stack)
    push!(stack, (x, y))

    marks[x, y] = cluster_num
    spins[x, y] = -spin_i

    cluster_size = 1

    while !isempty(stack)
        x, y = pop!(stack)

        neighbors = (
            (mod1(x - 1, N), y),
            (mod1(x + 1, N), y),
            (x, mod1(y - 1, M)),
            (x, mod1(y + 1, M))
        )

        for (nx, ny) in neighbors
            if marks[nx, ny] != cluster_num &&
               spins[nx, ny] == spin_i &&
               rand() < p

                marks[nx, ny] = cluster_num

                # Invertimos el spin al tiro
                spins[nx, ny] = -spin_i

                push!(stack, (nx, ny))
                cluster_size += 1
            end
        end
    end

    return cluster_size, spin_i
end

"""
Ejecuta 'iter' iteraciones del algoritmo de Wolff
que no es exactamente igual a un MCS

Retorna la magnetizacion normalizada en cada iter.
"""
function wolff(spin_arr, iter::Int, T::Real, J=1.0, K=1.0)
    spins = copy(spin_arr)
    N, M = size(spins)

    p = -expm1((-2 * J)/(K * T))

    mag_time_series = zeros(Float64, iter)
    current_net_spin = sum(spins)

    marks = zeros(Int, N, M)
    stack = Tuple{Int,Int}[]
    cluster_num = 0

    for t in 1:iter
        cluster_num += 1

        cluster_size, spin_i = wolff_update!(
            spins, p, marks, cluster_num, stack
        )

        current_net_spin -= 2 * spin_i * cluster_size
        mag_time_series[t] = current_net_spin / (N*M)
    end

    return mag_time_series
end

end #module end

using Plots
using Statistics
using Base.Threads
using DataFrames
using CSV
using Random

# ==============================================
# 1. ACTUALIZACIÓN DE WOLFF
# ==============================================

"""
Construye e invierte un cluster de Wolff.

Devuelve:
- cluster_size: cantidad de espines invertidos
- seed_spin: signo original del cluster

marks, stack y stamp permiten reutilizar memoria.
"""
function wolff_update!(spins, p, marks, stamp, stack)

    N, M = size(spins)

    x = rand(1:N)
    y = rand(1:M)

    seed_spin = spins[x, y]

    empty!(stack)
    push!(stack, (x, y))

    marks[x, y] = stamp
    spins[x, y] = -seed_spin

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
            if marks[nx, ny] != stamp &&
               spins[nx, ny] == seed_spin &&
               rand() < p

                marks[nx, ny] = stamp

                # Invertimos el espín al incorporarlo
                spins[nx, ny] = -seed_spin

                push!(stack, (nx, ny))
                cluster_size += 1
            end
        end
    end

    return cluster_size, seed_spin
end


# ==============================================
# 2. SIMULACIÓN DE WOLFF
# ==============================================

"""
Ejecuta 'iter' barridos equivalentes de Wolff.

Un barrido equivalente corresponde aproximadamente
a L² espines invertidos, acumulando clusters.

Devuelve un vector de magnetización normalizada,
con una medición por barrido equivalente.
"""
function wolff(spin_arr, iter::Int, T::Real)
    spins = copy(spin_arr)
    N, M = size(spins)

    p = -expm1(-2 / T)

    mag_time_series = zeros(Float64, iter)
    current_net_spin = sum(spins)

    marks = zeros(Int, N, M)
    stack = Tuple{Int,Int}[]
    stamp = 0

    for t in 1:iter
        stamp += 1

        cluster_size, seed_spin = wolff_update!(
            spins, p, marks, stamp, stack
        )

        current_net_spin -= 2 * seed_spin * cluster_size
        mag_time_series[t] = current_net_spin / (N*M)
    end

    return mag_time_series
end


# ==============================================
# 3. SIMULACIONES PARA FSS
# ==============================================

function get_magnetization_sim(
    Ls::AbstractVector{<:Int},
    Ts::AbstractVector{<:Real},
    iter::Int,
    sims::Int,
    crit_temp::Real;
    burn_in::Int=3000
)

    @assert 0 <= burn_in < iter

    n_rows = length(Ls) * length(Ts) * sims

    results = DataFrame(
        L = zeros(Int, n_rows),
        T = zeros(Float64, n_rows),
        sim = zeros(Int, n_rows),
        m = zeros(Float64, n_rows),
        mAbs = zeros(Float64, n_rows),
        m2 = zeros(Float64, n_rows),
        m4 = zeros(Float64, n_rows)
    )

    for (j, L) in enumerate(Ls)

        lattice_unif = ones(Int, L, L)

        Threads.@threads for i in eachindex(Ts)

            T = Ts[i]

            for s in 1:sims

                # Condición inicial
                current_lattice = T < crit_temp ?
                    lattice_unif :
                    rand((-1, 1), L, L)

                # Simulación Wolff
                mag_time_series = wolff(
                    current_lattice, iter, T
                )

                # Descartar termalización
                m = @view mag_time_series[burn_in+1:end]

                # Fila única para cada combinación
                row = ((j - 1) * length(Ts) + (i - 1)) * sims + s

                results.L[row] = L
                results.T[row] = T
                results.sim[row] = s
                results.m[row] = mean(m)
                results.mAbs[row] = mean(abs, m)
                results.m2[row] = mean(x -> x^2, m)
                results.m4[row] = mean(x -> x^4, m)

            end
        end
    end

    return results
end


# ==============================================
# 4. EJECUTAR SIMULACIONES
# ==============================================

Ls = [16, 24, 32, 48, 64, 96, 128]
Ts = range(2.20, 2.34, step=0.005)

iter = 20_000
sims = 80
burn_in = 2000

Tc = 2 / log(1 + sqrt(2))

@time sim_results = get_magnetization_sim(
    Ls, Ts, iter, sims, Tc;
    burn_in=burn_in
)


# ==============================================
# 5. GUARDAR DATOS INDIVIDUALES
# ==============================================

mkpath("simulation_results")

CSV.write(
    "simulation_results/wolff_sim_data.csv",
    sim_results
)


# ==============================================
# 6. CALCULAR OBSERVABLES PROMEDIADOS
# ==============================================

function binder_cumulant(m4, m2)
    return 1 - m4 / (3 * m2^2)
end

result = combine(
    groupby(sim_results, [:L, :T]),
    :m2 => mean => :m2_mean,
    :m4 => mean => :m4_mean,
    :m => mean => :m_mean,
    :mAbs => mean => :mAbs_mean
)

result.U = binder_cumulant.(
    result.m4_mean,
    result.m2_mean
)

# Susceptibilidad magnética
result.chi = result.L.^2 ./ result.T .*
    (result.m2_mean .- result.m_mean.^2)

# Susceptibilidad modificada
result.chi_abs = result.L.^2 ./ result.T .*
    (result.m2_mean .- result.mAbs_mean.^2)

sort!(result, [:L, :T])

CSV.write(
    "simulation_results/wolff_binder_data.csv",
    result
)

println("Simulación Wolff terminada.")
println("Filas guardadas: ", nrow(sim_results))

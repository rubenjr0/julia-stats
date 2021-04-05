# using Plots
# theme(:wong2)

function help()
    printstyled("== Medidas estadisticas ==\n", color=:reverse)
    println("= media(x)")
    println("= σ2(x) Varianza")
    println("= σ(x) Desviación típica")
    println("= dm(x) Desviación media")
    println("= mediana(x)")
    println("= moda(x)")
    println("= z(x) Tipificador de x: z(x).(x)\n")
    printstyled("== Frecuencias ==\n", color=:reverse)
    println("= ni(x) Frecuencia absoluta")
    println("= fi(x) Frecuencia relativa")
    println("= Ni(x) Frecuencia absoluta acumulada")
    println("= Fi(x) Frecuencia relativa acumulada\n")
    printstyled("== Momentos ==\n", color=:reverse)
    println("= mrc(x,r,c) Momento de orden R en el punto C")
    println("= mr(x) Momento ordinario de orden R")
    println("= μr(x) Momento central de orden R\n")
    printstyled("== Asimetría ==\n", color=:reverse)
    println("= ap(x) Coeficiente de asimetría de Pearson")
    println("= g1(x) Coeficiente de asimetría de Fisher\n")
    printstyled("== Apuntamiento ==\n", color=:reverse)
    println("= g2(x) Coeficiente de aplastamiento de Fisher\n")
end

# Base
media(x) = sum(x) / length(x)
cov(x, y) = sum(x .* y) / length(x) - media(x) * media(y)
σ2(x) = cov(x, x)
σ(x) = sqrt(σ2(x))
dm(x) = media(x .- media(x))
mediana(x) = quantile(x, 0.5)
moda(x) = unique(x)[findmax(ni(x))[2]]
z(x) = k -> (k - media(x)) / σ(x)

# Freqs
ni(x) = [count(==(xi), x) for xi in unique(x)]
fi(x) = ni(x) ./ length(x)
Ni(x) = x |> ni |> cumsum
Fi(x) = x |> fi |> cumsum
#= 
function ni(x)
    d = Dict()
    for xi in x
        if xi in keys(d)
            d[xi] += 1
        else
            d[xi] = 1
        end
    end
    return d
end =#

# Momentos
mrc(x, c, r) = sum(((unique(x) .- c).^r) .* fi(x))
mr(x, r) = mrc(x, 0, r)
μr(x, r) = mrc(x, media(x), r)

# Asimetria
ap(x) = (media(x) - moda(x)) / σ(x)
g1(x) = μr(x, 3) / (σ(x)^3)

# Apuntamiento
g2(x) = μr(x, 4) / (σ2(x)^2) - 3

# Outliers
function outliers(x, LB, HB)
    RQ = HB - LB
    II = LB - 1.5 * RQ
    IS = HB + 1.5 * RQ
    EI = LB - 3 * RQ
    ES = HB + 3 * RQ
    return Dict{String,Any}(
        "st" => filter(xi -> xi < EI || xi > ES, x), 
        "lt" => filter(xi -> (xi > EI && xi < II) || (xi < ES && xi > IS), x)
    )
end

# Cuantiles
function quantile(x, c)
    cN = c * length(x)
    D = cN % 1
    E = Int(cN - D)
    xs = sort(x)
    if D != 0
        return xs[E + 1]
    else
        return (xs[E] + xs[E + 1]) / 2
    end
end

# Regression
function reg_lin(x, y) 
    a = cov(x, y) / σ2(x)
    b = media(y) - a * media(x)
    return [a, b]
end

function reg_exp(x, y)
    logy = log.(y)
    a = cov(x, logy) / σ2(x)
    b = media(logy) - a * media(x)
    return [exp(a), exp(b)]
end

# Predictors
function predictor_lin(x, y)
    a, b = reg_lin(x, y)
    return x -> a * x + b
end

function predictor_exp(x, y)
    a, b = reg_exp(x, y)
    return x -> a^x * b
end

# Errors
err(x,y,f) = sum((f.(x) .- y).^2)

# Plots
function plot_reg(x, y, p)
    scatter(x, y, legend=false)
    plot!(x, p(x, y))
end
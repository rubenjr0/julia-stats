using Plots
theme(:wong2)

function help()
    printstyled("== Tema 1 ==\n", color=:reverse)
    printstyled("== Medidas estadisticas ==\n", color=:reverse)
    println("= media(x)")
    println("= var(x) Varianza")
    println("= dt(x) Desviación típica")
    println("= dm(x) Desviación media")
    println("= mediana(x)")
    println("= moda(x)")
    println("= z(x) Tipificador de x: z(x).(x)")
    println("= outliers(x, LB, HB) Outliers con limites inferior LB y superior HB (cuantiles)\n")
    printstyled("== Frecuencias ==\n", color=:reverse)
    println("= ni(x) Frecuencias absolutas")
    println("= fi(x) Frecuencias relativas")
    println("= Ni(x) Frecuencias absolutas acumuladas")
    println("= Fi(x) Frecuencias relativas acumuladas")
    println("= var_from_ni(u,n) Genera una variable a partir de sus frecuencias absolutas\n")
    printstyled("== Momentos ==\n", color=:reverse)
    println("= mrc(x,r,c) Momento de orden R en el punto C")
    println("= mr(x) Momento ordinario de orden R")
    println("= μr(x) Momento central de orden R\n")
    printstyled("== Asimetría ==\n", color=:reverse)
    println("= ap(x) Coeficiente de asimetría de Pearson")
    println("= g1(x) Coeficiente de asimetría de Fisher\n")
    printstyled("== Apuntamiento ==\n", color=:reverse)
    println("= g2(x) Coeficiente de aplastamiento de Fisher\n")
    printstyled("== Tema 2 ==\n", color=:reverse)
    printstyled("== Regresion ==\n", color=:reverse)
    println("= reg_lin(x, y) Regresion lineal ax+b")
    println("= reg_exp(x, y) Regresion exponencial a*b^x")
    printstyled("== Predictores ==\n", color=:reverse)
    println("= pred_lin(x, y) Construye un predictor lineal")
    println("= pred_exp(x, y) Construye un predictor exponencial")
    printstyled("== Error ==\n", color=:reverse)
    println("= ecm(x, y, p) Calcula el error cuadratico medio del predictor")
    printstyled("== Visualización ==\n", color=:reverse)
    println("= plot_reg(x, y, p) Visualiza la regresion con el predictor p\n")
end

# Base
media(x) = sum(x) / length(x)
cov(x, y) = sum(x .* y) / length(x) - media(x) * media(y)
var(x) = cov(x, x)
dt(x) = sqrt(var(x))
dm(x) = media(x .- media(x))
mediana(x) = quantile(x, 0.5)
moda(x) = unique(x)[findmax(ni(x))[2]]
z(x) = k -> (k - media(x)) / dt(x)

# Freqs
ni(x) = [count(==(xi), x) for xi in unique(x)]
fi(x) = ni(x) ./ length(x)
Ni(x) = x |> ni |> cumsum
Fi(x) = x |> fi |> cumsum

var_from_ni(u, n) = vcat([repeat([uj], nj) for (uj, nj) in zip(u,n)]...)

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
mrc(x, r, c) = sum(((unique(x) .- c).^r) .* fi(x))
mr(x, r) = mrc(x, 0, r)
μr(x, r) = mrc(x, media(x), r)

# Asimetria
ap(x) = (media(x) - moda(x)) / σ(x)
g1(x) = μr(x, 3) / (dt(x)^3)

# Apuntamiento
g2(x) = μr(x, 4) / (var(x)^2) - 3

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
    a = cov(x, y) / var(x)
    b = media(y) - a * media(x)
    return [a, b]
end

function reg_exp(x, y)
    logy = log.(y)
    a = cov(x, logy) / var(x)
    b = media(logy) - a * media(x)
    return [exp(a), exp(b)]
end

# Predictors
function pred_lin(x, y)
    a, b = reg_lin(x, y)
    return x -> a * x + b
end

function pred_exp(x, y)
    a, b = reg_exp(x, y)
    return x -> a^x * b
end

# Errors
ecm(x,y,p) = sum((p(x,y).(x) .- y).^2) / length(x)

# Plots
function plot_reg(x, y, p)
    scatter(x, y, legend=false)
    plot!(x, p(x, y))
end
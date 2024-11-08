
# Create the bnlearn test Gaussian network and write to a .rda file

a = data.frame(A = rnorm(5000, 1, 1),
               B = rnorm(5000, 2, 3),
               C = rep(0, 5000),
               D = rep(0, 5000),
               E = rnorm(5000, 3.5, 2),
               F = rep(0, 5000),
               G = rnorm(5000, 5, 2))
a$C = 2 * (a$A + a$B) + rnorm(5000, 2, 0.5)
a$D = 1.5 * a$B + rnorm(5000, 6, 0.33)
a$F = 2 * a$A + a$D + a$E + 1.5 * a$G + rnorm(5000, 0, 1)

# network specification.
dag = model2network("[A][B][E][G][C|A:B][D|B][F|A:D:E:G]")

bn = custom.fit(dag, list(
  A = list(coef = c("(Intercept)" = 1), sd = 1),
  B = list(coef = c("(Intercept)" = 2), sd = 3),
  C = list(coef = c("(Intercept)" = 2, "A" = 2, B = 2), sd = 0.5),
  D = list(coef = c("(Intercept)" = 6, "B" = 1.5), sd = 0.33),
  E = list(coef = c("(Intercept)" = 3.5), sd = 2),
  F = list(coef = c("(Intercept)" = 0, "A" = 2, "D" = 1, "E" = 1, "G" = 1.5), sd = 1),
  G = list(coef = c("(Intercept)" = 5), sd = 2)
))

save(bn, file='rda/gauss.rda')
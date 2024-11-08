#
# Test d-separation in small network used in BN Review paper

library(kkbnlearn)

dag = model2network("[A][C][B|A:C]")

print(dag)

res = cpdag(dag)
print(res)

graphviz.plot(res)


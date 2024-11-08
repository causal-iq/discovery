#
# Test d-separation in small network used in BN Review paper

library(bnlearn)

dag = model2network("[X][D|X][F|X][E|D:Y][Y|F][G|E]")

print(dag)

# graphviz.plot(dag)

message(sprintf('X and Y d-separated: %s', dsep(dag, 'X', 'Y')))
message(sprintf('X and Y given F d-separated: %s', dsep(dag, 'X', 'Y', 'F')))
message(sprintf('X and Y given F, G d-separated: %s', dsep(dag, 'X', 'Y', c('F', 'G'))))
message(sprintf('X and Y given D, E, F d-separated: %s', dsep(dag, 'X', 'Y', c('D', 'E', 'F'))))
message(sprintf('X and Y given D, F d-separated: %s', dsep(dag, 'X', 'Y', c('D', 'F'))))
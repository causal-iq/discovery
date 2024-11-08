
# Demonstrating converting learnt bnlearn graphs to dagLearned.csv format which Bayesys software
# can read in.

# Load the bnlearn package (needs to be installed - see bnlearn.com)

library(bnlearn)

# use sample data provided in bnlearn - creates dataset "asia"

data(asia)
print(asia)

# use hc (hill-climbing) bnlearn function to learn a graph from this data

graph = hc(asia)
print(graph)

# generate a representation of graph in Bayesys format in lines variable

lines = character(0)  # array which contains lines in file
lines[1] = "ID,Variable 1,Dependency,Variable 2"  # header row

# loop over directed edges in graph, constructing appropriate line in dagLearned format

edges = arcs(graph)
if (nrow(edges) > 0) {
  for (i in 1:nrow(edges)) {
    lines[i + 1] = sprintf("%s,%s", i, paste(edges[i, ], collapse=",->,"))
  }
}

# write the lines out to file 'dagLearned,csv'

writeLines(lines, 'dagLearned.csv')

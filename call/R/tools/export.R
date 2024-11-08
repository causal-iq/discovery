#
# Export data frame to CSV, or graph to Tetrad or Bayesys compliant graph definition
#
# data {data frame or BN object} - data frame or graph to export
# fileName {character} - name of file to write to
# format {character} - "tetrad", "bayesys", otherwise data frame export
#
export = function(data, fileName, format="tetrad") {

  #
  # Export BN graph in Tetrad format
  #
  # graph {bn} - bnlearn graph object
  # fileName {character} - name of output file
  #
  exportTetrad = function(graph, fileName) {
    
    # Header lines and list of node names
    lines = character(0)
    lines[1] = "Graph Nodes:"
    lines[2] = paste(nodes(graph), collapse=",")
    lines[3] = ""
    lines[4] = "Graph Edges:"
    
    # write out each directed edge
    
    edges = arcs(graph)
    for (i in 1:nrow(edges)) {
      lines[i + 4] = sprintf("%s. %s", i, paste(edges[i, ], collapse=" --> "))
    }
    writeLines(lines, fileName)
    message(lines)
  }

  #
  # Export BN graph in Bayesys format
  #
  # graph {bn} - bnlearn graph object
  #
  exportBayesys = function(graph, fileName) {
    if (class(graph) != 'bn') {
      return(NULL)
    }
    lines = character(0)
    lines[1] = "ID,Variable 1,Dependency,Variable 2"
    
    # loop over directed edges 
    
    edges = arcs(graph)
    if (nrow(edges) > 0) {
      for (i in 1:nrow(edges)) {
        lines[i + 1] = sprintf("%s,%s", i, paste(edges[i, ], collapse=",->,"))
      }
    }
    writeLines(lines, fileName)
  }
    
    
    
    
  if (format == "tetrad") {
    exportTetrad(data, fileName)
  } else if (format == "bayesys") {
    exportBayesys(data, fileName)
  } else {
    
    # format required by Tetrad and Bayesys
    
    write.table(data, file = fileName, append = FALSE, quote = TRUE, sep = ",",
                eol = "\n", na = "NA", dec = ".", row.names = FALSE,
                col.names = TRUE, qmethod = c("escape", "double"),
                fileEncoding = "")
  }
}
# compute the score of a network.
debug_score = function(x, data, type = NULL, ..., by.node = FALSE, debug = FALSE) {

  # check x's class.
  bnlearn:::check.bn(x)
  # the original data set is needed.
  bnlearn:::check.data(data)
  # check the network against the data.
  bnlearn:::check.bn.vs.data(x, data)
  # check debug and by.node.
  bnlearn:::check.logical(by.node)
  bnlearn:::check.logical(debug)
  # no score if the graph is partially directed.
  if (bnlearn:::is.pdag(x$arcs, names(x$nodes)))
    stop("the graph is only partially directed.")
  # check the score label.
  type = bnlearn:::check.score(type, data)

  # expand and sanitize score-specific arguments.
  extra.args = bnlearn:::check.score.args(score = type, network = x,
                 data = data, extra.args = list(...), learning = FALSE)
  # check that the score is decomposable when returning node contributions.
  if (by.node && !bnlearn:::is.score.decomposable(type, extra.args))
    stop("the score is not decomposable, node terms are not defined.")

  # compute the node contributions to the network score.
  local = bnlearn:::per.node.score(network = x, data = data, score = type,
            targets = names(x$nodes), extra.args = extra.args, debug = debug)

  if (by.node)
    return(local)
  else
    return(sum(local))

}#SCORE

# from frontend-learning.R
# TABU list greedy search frontend.
kk.tabu = function(x, start = NULL, whitelist = NULL, blacklist = NULL,
    score = NULL, ..., debug = FALSE, tabu = 10, max.tabu = tabu,
    max.iter = Inf, maxp = Inf, optimized = TRUE) {

  kk.greedy.search(x = x, start = start, whitelist = whitelist,
    blacklist = blacklist, score = score, heuristic = "tabu", debug = debug,
    ..., max.iter = max.iter, tabu = tabu, max.tabu = max.tabu,
    maxp = maxp, optimized = optimized)

}#TABU

# from frontend-learning.R
# Hill Climbing greedy search frontend.
kk.hc = function(x, start = NULL, whitelist = NULL, blacklist = NULL,
    score = NULL, ..., debug = FALSE, restart = 0, perturb = 1,
    max.iter = Inf, maxp = Inf, optimized = TRUE) {

  kk.greedy.search(x = x, start = start, whitelist = whitelist,
    blacklist = blacklist, score = score, heuristic = "hc", debug = debug,
    ..., restart = restart, perturb = perturb,
    max.iter = max.iter, maxp = maxp, optimized = optimized)

}#HC

# from learning-algorithms.R
# score-based learning algorithms.
kk.greedy.search = function(x, start = NULL, whitelist = NULL, blacklist = NULL,
    score = NULL, heuristic = "hc", ..., optimized = TRUE, debug = FALSE) {

  # check the data are there.
  bnlearn:::check.data(x)
  # check the algorithm.
  bnlearn:::check.learning.algorithm(heuristic, class = "score")
  # check the score label.
  score = bnlearn:::check.score(score, x)
  # check debug.
  bnlearn:::check.logical(debug)

  # check unused arguments in misc.args.
  extra = list(...)
  misc.args = extra[names(extra) %in% bnlearn:::method.extra.args[[heuristic]]]
  extra.args = extra[names(extra) %in% bnlearn:::score.extra.args[[score]]]
  bnlearn:::check.unused.args(extra, c(bnlearn:::method.extra.args[[heuristic]], bnlearn:::score.extra.args[[score]]))

  # expand and check the max.iter parameter (common to all algorithms).
  max.iter = bnlearn:::check.max.iter(misc.args$max.iter)
  # expand and check the maxp parameter (common to all algorithms).
  maxp = bnlearn:::check.maxp(misc.args$maxp, data = x)

  if (heuristic == "hc") {

    # expand and check the number of random restarts.
    restart = bnlearn:::check.restart(misc.args$restart)
    # expand and check the magnitude of the perturbation when random restarts
    # are effectively used.
    perturb = ifelse((restart > 0), bnlearn:::check.perturb(misc.args$perturb), 0)

  }#THEN
  else if (heuristic == "tabu") {

    # expand and check the arguments related to the tabu list.
    tabu = bnlearn:::check.tabu(misc.args$tabu)
    max.tabu = bnlearn:::check.max.tabu(misc.args$max.tabu, tabu)

  }#THEN

  # sanitize whitelist and blacklist, if any.
  whitelist = bnlearn:::build.whitelist(whitelist, nodes = names(x), data = x,
                algo = heuristic, criterion = score)
  blacklist = bnlearn:::build.blacklist(blacklist, whitelist, names(x), algo = heuristic)
  # if there is no preseeded network, use an empty one.
  if (is.null(start))
    start = bnlearn:::empty.graph(nodes = names(x))
  else {

    # check start's class.
    bnlearn:::check.bn(start)
    # check the preseeded network against the data set.
    bnlearn:::check.bn.vs.data(start, x)
    # check the preseeded network against the maximum number of parents.
    nparents = sapply(start$nodes, function(x) length(x$parents))
    if (any(nparents > maxp))
      stop("nodes ", paste(names(which(nparents > maxp)), collapse = " "),
        " have more than 'maxp' parents.")
    # check the preseeded network is valid for the model assumptions.
    bnlearn:::check.arcs.against.assumptions(start$arcs, x, score)

  }#ELSE

  if (debug == TRUE && heuristic == "tabu") {
    print(sprintf(paste("Options: score='%s', max.iter=%f, maxp=%f, ",
                        "tabu=%i, tabu.max=%i", sep=""),
                  score, max.iter, maxp, tabu, max.tabu))
  }

  # apply the whitelist to the preseeded network; undirected arcs are allowed
  # but applied as directed to interoperate with bnlearn() in hybrid.search().
  if (!is.null(whitelist)) {

    for (i in 1:nrow(whitelist))
      start$arcs = bnlearn:::set.arc.direction(whitelist[i, "from"],
                       whitelist[i, "to"], start$arcs)

  }#THEN

  # apply the blacklist to the preseeded network.
  if (!is.null(blacklist)) {

    blacklisted = apply(start$arcs, 1, function(x){ bnlearn:::is.blacklisted(blacklist, x) })
    start$arcs = start$arcs[!blacklisted, , drop = FALSE]

  }#THEN

  # be sure the graph structure is up to date.
  start$nodes = bnlearn:::cache.structure(names(start$nodes), arcs = start$arcs)
  # no party if the graph is partially directed.
  if (bnlearn:::is.pdag(start$arcs, names(start$nodes)))
    stop("the graph is only partially directed.")
  # check whether the graph is acyclic.
  if (!bnlearn:::is.acyclic(arcs = start$arcs, nodes = names(start$nodes)))
    stop("the preseeded graph contains cycles.")

  # sanitize score-specific arguments.
  extra.args = bnlearn:::check.score.args(score = score, network = start,
                 data = x, extra.args = extra.args, learning = TRUE)

  # reset the test counter.
  bnlearn:::reset.test.counter()

  # call the right backend.
  if (heuristic == "hc") {

    res = kk.hill.climbing(x = x, start = start, whitelist = whitelist,
      blacklist = blacklist, score = score, extra.args = extra.args,
      restart = restart, perturb = perturb, max.iter = max.iter,
      maxp = maxp, optimized = optimized, debug = debug)

  }#THEN
  else if (heuristic == "tabu"){

    res = kk.tabu.search(x = x, start = start, whitelist = whitelist,
      blacklist = blacklist, score = score, extra.args = extra.args,
      max.iter = max.iter, optimized = optimized, tabu = tabu,
      maxp = maxp, debug = debug)

  }#THEN

  # set the metadata of the network in one stroke.
  res$learning = list(whitelist = whitelist, blacklist = blacklist,
    test = score, ntests = test.counter(),
    algo = heuristic, args = extra.args, optimized = optimized,
    illegal = bnlearn:::check.arcs.against.assumptions(NULL, x, score))

  invisible(res)

}


# from hill-climbing.R
# unified hill climbing implementation (both optimized and by spec).
kk.hill.climbing = function(x, start, whitelist, blacklist, score, extra.args,
    restart, perturb, max.iter, maxp, optimized, debug = FALSE) {

  # cache nodes' labels.
  nodes = names(x)
  # cache the number of nodes.
  n.nodes = length(nodes)
  # set the iteration counter.
  iter = 1
  # check whether the score is score-equivalent.
  score.equivalence = bnlearn:::is.score.equivalent(score, nodes, extra.args)
  # check whether the score is decomposable.
  score.decomposability = bnlearn:::is.score.decomposable(score, extra.args)
  # allocate the cache matrix.
  cache = matrix(0, nrow = n.nodes, ncol = n.nodes)
  # nodes to be updated (all of them in the first iteration).
  updated = seq_len(n.nodes) - 1L
  # set the number of random restarts.
  restart.counter = restart

  # set the reference score.
  reference.score = bnlearn:::per.node.score(network = start, score = score,
                      targets = nodes, extra.args = extra.args, data = x)

  # convert the blacklist to an adjacency matrix for easy use.
  if (!is.null(blacklist))
    blmat = bnlearn:::arcs2amat(blacklist, nodes)
  else
    blmat = matrix(0L, nrow = n.nodes, ncol = n.nodes)

  # convert the whitelist to an adjacency matrix for easy use.
  if (!is.null(whitelist))
    wlmat = bnlearn:::arcs2amat(whitelist, nodes)
  else
    wlmat = matrix(0L, nrow = n.nodes, ncol = n.nodes)

  if (debug) {

    cat("----------------------------------------------------------------\n")
    cat("* starting from the following network:\n")
    print(start)
    cat("* current score:", sum(reference.score), "\n")
    cat("* whitelisted arcs are:\n")
    if (!is.null(whitelist)) print(whitelist)
    cat("* blacklisted arcs are:\n")
    if (!is.null(blacklist)) print(blacklist)

    # set the metadata of the network; othewise the debugging output is
    # confusing and not nearly as informative.
    start$learning$algo = "hc"
    start$learning$ntests = 0
    start$learning$test = score
    start$learning$args = extra.args
    start$learning$optimized = optimized

  }#THEN

  repeat {

    # build the adjacency matrix of the current network structure.
    amat = bnlearn:::arcs2amat(start$arcs, nodes)
    # compute the number of parents of each node.
    nparents = colSums(amat)

    # set up the score cache (BEWARE: in place modification!).
    .Call(bnlearn:::call_score_cache_fill,
          nodes = nodes,
          data = x,
          network = start,
          score = score,
          extra = extra.args,
          reference = reference.score,
          equivalence = score.equivalence && optimized,
          decomposability = score.decomposability,
          updated = (if (optimized) updated else seq(length(nodes)) - 1L),
          amat = amat,
          cache = cache,
          blmat = blmat,
          debug = debug)

    # select which arcs should be tested for inclusion in the graph (hybrid
    # learning algorithms should hook the restrict phase here).
    to.be.added = bnlearn:::arcs.to.be.added(amat = amat, nodes = nodes,
                    blacklist = blmat, whitelist = NULL, nparents = nparents,
                    maxp = maxp, arcs = FALSE)

    # get the best arc addition/removal/reversal.
    bestop = .Call(bnlearn:::call_hc_opt_step,
                   amat = amat,
                   nodes = nodes,
                   added = to.be.added,
                   cache = cache,
                   reference = reference.score,
                   wlmat = wlmat,
                   blmat = blmat,
                   nparents = nparents,
                   maxp = maxp,
                   debug = debug)

    # the value FALSE is the canary value in bestop$op meaning "no operation
    # improved the network score"; break the loop.
    if (bestop$op == FALSE) {

      if ((restart.counter >= 0) && (restart > 0)) {

        if (restart.counter == restart) {

          # store away the learned network at the first restart.
          best.network = start
          best.network.score = sum(reference.score)

        }#THEN
        else {

          # if the network found by the algorithm is better, use that one as
          # starting network for the next random restart; use the old one
          # otherwise (and keep amat in sync).
          if (best.network.score > sum(reference.score)) {

            start = best.network
            amat = bnlearn:::arcs2amat(best.network$arcs, nodes)

            if (debug) {

              cat("----------------------------------------------------------------\n")
              cat("* the old network was better, discarding the current one.\n")
              cat("* now the current network is :\n")
              print(start)
              cat("* current score:", best.network.score, "\n")

            }#THEN

          }#THEN
          else {

            # store away the learned network at the first restart.
            best.network = start
            best.network.score = sum(reference.score)

          }#ELSE

        }#ELSE

        # this is the end; if the network found by the algorithm is the best one
        # it's now the 'end' one once more.
        if (restart.counter == 0) break

        # don't try to do anything if there are no more iterations left.
        if (iter >= max.iter) {

          if (debug)
            cat("@ stopping at iteration", max.iter, "ignoring random restart.\n")

          break

        }#THEN
        # increment the iteration counter.
        iter = iter + 1

        # decrement the number of random restart we still have to do.
        restart.counter = restart.counter - 1

        if (debug) {

          cat("----------------------------------------------------------------\n")
          cat("* doing a random restart,", restart.counter, "of", restart, "left.\n")

        }#THEN

        # perturb the network.
        start = bnlearn:::perturb.backend(network = start, iter = perturb, nodes = nodes,
                  amat = amat, whitelist = whitelist, blacklist = blacklist,
                  maxp = maxp, debug = debug)

        # update the cached values of the end network.
        start$nodes = bnlearn:::cache.structure(nodes, arcs = start$arcs)

        # update the scores of the nodes as needed.
        if (best.network.score > sum(reference.score)) {

          # all scores must be updated; this happens when both the network
          # resulted from the random restart is discarded and the old network
          # is perturbed.
          reference.score = bnlearn:::per.node.score(network = start, score = score,
                          targets = nodes, extra.args = extra.args, data = x)

          updated = which(nodes %in% nodes) - 1L

        }#THEN
        else {

          # the scores whose nodes' parents changed must be updated.
          reference.score[names(start$updates)] =
            bnlearn:::per.node.score(network = start, score = score,
                targets = names(start$updates),
               extra.args = extra.args, data = x)

          updated = which(nodes %in% names(start$updates)) - 1L

        }#THEN

        # nuke start$updates from orbit.
        start$updates = NULL

        next

      }#THEN
      else
        break

    }#THEN

    # update the network structure.
    start = bnlearn:::arc.operations(start, from = bestop$from, to = bestop$to,
              op = bestop$op, check.cycles = FALSE, check.illegal = FALSE,
              update = TRUE, debug = FALSE)

    if ("prior" %in% names(extra.args) && extra.args$prior %in% c("cs", "marginal"))
      bnlearn:::reference.score[c(bestop$from, bestop$to)] =
        bnlearn:::per.node.score(network = start, score = score,
          targets = c(bestop$from, bestop$to), extra.args = extra.args, data = x)

    # set the nodes whose cached score deltas are to be updated.
    if (bestop$op == "reverse")
      updated = which(nodes %in% c(bestop$from, bestop$to)) - 1L
    else
      updated = which(nodes %in% bestop$to) - 1L

    if (debug) {

      # update the test counter of the network; very useful to check how many
      # score comparison has been done up to now.
      start$learning$ntests = test.counter()

      cat("----------------------------------------------------------------\n")
      cat("* best operation was: ")
      if (bestop$op == "set")
        cat("adding", bestop$from, "->", bestop$to, ".\n")
      else if (bestop$op == "drop")
        cat("removing", bestop$from, "->", bestop$to, ".\n")
      else
        cat("reversing", bestop$from, "->", bestop$to, ".\n")
      cat("* current network is :\n")
      print(start)
      cat("* current score:", sum(reference.score), "\n")

    }#THEN

    # check the current iteration index against the max.iter parameter.
    if (iter >= max.iter) {

      if (debug)
        cat("@ stopping at iteration", max.iter, ".\n")

      break

    }#THEN
    else iter = iter + 1

  }#REPEAT

  return(start)

}#HILL.CLIMBING


# from tabu.R
# unified tabu search implementation (both optimized and by spec).
kk.tabu.search = function(x, start, whitelist, blacklist, score, extra.args,
    max.iter, maxp, optimized, tabu, debug = FALSE) {

  # cache nodes' labels.
  nodes = names(x)
  # cache the number of nodes.
  n.nodes = length(nodes)
  # set the iteration counter.
  iter = 1
  # check whether the score is score-equivalent.
  score.equivalence = bnlearn:::is.score.equivalent(score, nodes, extra.args)
  # check whether the score is decomposable.
  score.decomposability = bnlearn:::is.score.decomposable(score, extra.args)

  if (debug == TRUE) {
    print(sprintf("Score equivalence is %i, decomposability is %i and penalty coeff is %f",
                  score.equivalence, score.decomposability, extra.args$k))
  }
  # allocate the cache matrix.
  cache = matrix(0, nrow = n.nodes, ncol = n.nodes)
  # nodes to be updated (all of them in the first iteration).
  updated = seq_len(n.nodes) - 1L
  # allocate the tabu list.
  tabu.list = vector("list", tabu)
  # maximum number of iteration the algorithm can go on without
  # improving the best network score.
  max.loss.iter = tabu
  # set the counter for suc iterations.
  loss.iter = 0
  # keep track of the best score value.
  best.score = -Inf

  # set the reference score.
  reference.score = bnlearn:::per.node.score(network = start, score = score,
                      targets = nodes, extra.args = extra.args, data = x,
                      debug=debug)

  # convert the blacklist to an adjacency matrix for easy use.
  if (!is.null(blacklist))
    blmat = bnlearn:::arcs2amat(blacklist, nodes)
  else
    blmat = matrix(0L, nrow = n.nodes, ncol = n.nodes)

  # convert the whitelist to an adjacency matrix for easy use.
  if (!is.null(whitelist))
    wlmat = bnlearn:::arcs2amat(whitelist, nodes)
  else
    wlmat = matrix(0L, nrow = n.nodes, ncol = n.nodes)

  if (debug) {

    cat("----------------------------------------------------------------\n")
    cat("* starting from the following network:\n")
    print(start)
    cat("* current score:", sum(reference.score), "\n")
    cat("* whitelisted arcs are:\n")
    if (!is.null(whitelist)) print(whitelist)
    cat("* blacklisted arcs are:\n")
    if (!is.null(blacklist)) print(blacklist)

    # set the metadata of the network; othewise the debugging output is
    # confusing and not nearly as informative.
    start$learning$algo = "tabu"
    start$learning$ntests = 0
    start$learning$test = score
    start$learning$args = extra.args
    start$learning$optimized = optimized

  }#THEN

  repeat {

    current = as.integer((iter - 1) %% tabu)

    # keep the best network seen so far and its score value for the
    # evaluation of the stopping rule; but always create a "best network" in
    # the first iteration using the starting network.
    if ((sum(reference.score) > best.score) || (iter == 1)) {

      best.network = start
      best.score = sum(reference.score)

    }#THEN

    if (debug)
      cat("* iteration", iter, "using element", current, "of the tabu list.\n")

    # build the adjacency matrix of the current network structure.
    amat = bnlearn:::arcs2amat(start$arcs, nodes)
    # compute the number of parents of each node.
    nparents = colSums(amat)

    # add the hash of the network into the tabu list for future reference.
    # (BEWARE: in place modification of tabu.list!)
    .Call(bnlearn:::call_tabu_hash,
          amat = amat,
          nodes = nodes,
          tabu.list = tabu.list,
          current = current)

    # set up the score cache (BEWARE: in place modification!).
    .Call(bnlearn:::call_score_cache_fill,
          nodes = nodes,
          data = x,
          network = start,
          score = score,
          extra = extra.args,
          reference = reference.score,
          equivalence = score.equivalence && optimized,
          decomposability = score.decomposability,
          updated = (if (optimized) updated else seq(length(nodes)) - 1L),
          amat = amat,
          cache = cache,
          blmat = blmat,
          debug = debug)
    print(cache)

    # select which arcs should be tested for inclusion in the graph (hybrid
    # learning algorithms should hook the restrict phase here).
    to.be.added = bnlearn:::arcs.to.be.added(amat = amat, nodes = nodes,
                    blacklist = blmat, whitelist = NULL, nparents = nparents,
                    maxp = maxp, arcs = FALSE)
    print(to.be.added)
    # stop('reached here')

    # get the best arc addition/removal/reversal.
    bestop = .Call(call_tabu_step,
                   amat = amat,
                   nodes = nodes,
                   added = to.be.added,
                   cache = cache,
                   reference = reference.score,
                   wlmat = wlmat,
                   blmat = blmat,
                   tabu.list = tabu.list,
                   current = current,
                   baseline = 0,
                   nparents = nparents,
                   maxp = maxp,
                   debug = debug)

    # the value FALSE is the canary value in bestop$op meaning "no operation
    # improved the network score"; reconsider prevously discarded solutions
    # and find the one that causes the minimum decrease in the network score.
    if (bestop$op == FALSE) {

      if (loss.iter >= max.loss.iter) {

        # reset the return value to the best network ever found.
        start = best.network

        if (debug) {

          cat("----------------------------------------------------------------\n")
          cat("* maximum number of iterations without improvements reached, stopping.\n")
          cat("* best network ever seen is:\n")
          print(best.network)

        }#THEN

        break

      }#THEN
      else {

        # increase the counter of the iterations without improvements.
        loss.iter = loss.iter + 1

      }#ELSE

      if (debug) {

        cat("----------------------------------------------------------------\n")
        cat("* network score did not increase (for", loss.iter,
              "times), looking for a minimal decrease :\n")

      }#THEN

      bestop = .Call(call_tabu_step,
                     amat = amat,
                     nodes = nodes,
                     added = to.be.added,
                     cache = cache,
                     reference = reference.score,
                     wlmat = wlmat,
                     blmat = blmat,
                     tabu.list = tabu.list,
                     current = current,
                     baseline = -Inf,
                     nparents = nparents,
                     maxp = maxp,
                     debug = debug)

      # it might be that there are no more legal operations.
      if (bestop$op == FALSE) {

       if (debug) {

         cat("----------------------------------------------------------------\n")
         cat("* no more possible operations.\n")
         cat("@ stopping at iteration", iter, ".\n")

       }#THEN

       # reset the return value to the best network ever found.
       if (loss.iter > 0)
         start = best.network

       break

     }#THEN

    }#THEN
    else {

      if (sum(reference.score) > best.score)
        loss.iter = 0

    }#ELSE

    # update the network structure.
    start = arc.operations(start, from = bestop$from, to = bestop$to,
              op = bestop$op, check.cycles = FALSE, check.illegal = FALSE,
              update = TRUE, debug = FALSE)

    # set the nodes whose cached score deltas are to be updated.
    if (bestop$op == "reverse")
      updated = which(nodes %in% c(bestop$from, bestop$to)) - 1L
    else
      updated = which(nodes %in% bestop$to) - 1L

    if (debug) {

      # update the test counter of the network; very useful to check how many
      # score comparison has been done up to now.
      start$learning$ntests = test.counter()

      cat("----------------------------------------------------------------\n")
      cat("* best operation was: ")
      if (bestop$op == "set")
        cat("adding", bestop$from, "->", bestop$to, ".\n")
      else if (bestop$op == "drop")
        cat("removing", bestop$from, "->", bestop$to, ".\n")
      else
        cat("reversing", bestop$from, "->", bestop$to, ".\n")
      cat("* current network is :\n")
      print(start)
      cat("* current score:", sum(reference.score), "\n")
      cat(sprintf("* best score up to now: %s (delta: %s)\n", format(best.score),
        format(sum(reference.score) - best.score)))

    }#THEN

    # check the current iteration index against the max.iter parameter.
    if (iter >= max.iter) {

      if (debug)
        cat("@ stopping at iteration", max.iter, ".\n")

      # reset the return value to the best network ever found.
      if (loss.iter > 0)
        start = best.network

      break

    }#THEN
    else iter = iter + 1

  }#REPEAT

  return(start)


}

# from tabu.R
# unified tabu search implementation (both optimized and by spec).
orig.tabu.search = function(x, start, whitelist, blacklist, score, extra.args,
    max.iter, maxp, optimized, tabu, debug = FALSE) {

  # cache nodes' labels.
  nodes = names(x)
  # cache the number of nodes.
  n.nodes = length(nodes)
  # set the iteration counter.
  iter = 1
  # check whether the score is score-equivalent.
  score.equivalence = is.score.equivalent(score, nodes, extra.args)
  # check whether the score is decomposable.
  score.decomposability = is.score.decomposable(score, extra.args)
  # allocate the cache matrix.
  cache = matrix(0, nrow = n.nodes, ncol = n.nodes)
  # nodes to be updated (all of them in the first iteration).
  updated = seq_len(n.nodes) - 1L
  # allocate the tabu list.
  tabu.list = vector("list", tabu)
  # maximum number of iteration the algorithm can go on without
  # improving the best network score.
  max.loss.iter = tabu
  # set the counter for suc iterations.
  loss.iter = 0
  # keep track of the best score value.
  best.score = -Inf

  # set the reference score.
  reference.score = per.node.score(network = start, score = score,
                      targets = nodes, extra.args = extra.args, data = x)

  # convert the blacklist to an adjacency matrix for easy use.
  if (!is.null(blacklist))
    blmat = arcs2amat(blacklist, nodes)
  else
    blmat = matrix(0L, nrow = n.nodes, ncol = n.nodes)

  # convert the whitelist to an adjacency matrix for easy use.
  if (!is.null(whitelist))
    wlmat = arcs2amat(whitelist, nodes)
  else
    wlmat = matrix(0L, nrow = n.nodes, ncol = n.nodes)

  if (debug) {

    cat("----------------------------------------------------------------\n")
    cat("* starting from the following network:\n")
    print(start)
    cat("* current score:", sum(reference.score), "\n")
    cat("* whitelisted arcs are:\n")
    if (!is.null(whitelist)) print(whitelist)
    cat("* blacklisted arcs are:\n")
    if (!is.null(blacklist)) print(blacklist)

    # set the metadata of the network; othewise the debugging output is
    # confusing and not nearly as informative.
    start$learning$algo = "tabu"
    start$learning$ntests = 0
    start$learning$test = score
    start$learning$args = extra.args
    start$learning$optimized = optimized

  }#THEN

  repeat {

    current = as.integer((iter - 1) %% tabu)

    # keep the best network seen so far and its score value for the
    # evaluation of the stopping rule; but always create a "best network" in
    # the first iteration using the starting network.
    if ((sum(reference.score) > best.score) || (iter == 1)) {

      best.network = start
      best.score = sum(reference.score)

    }#THEN

    if (debug)
      cat("* iteration", iter, "using element", current, "of the tabu list.\n")

    # build the adjacency matrix of the current network structure.
    amat = arcs2amat(start$arcs, nodes)
    # compute the number of parents of each node.
    nparents = colSums(amat)

    # add the hash of the network into the tabu list for future reference.
    # (BEWARE: in place modification of tabu.list!)
    .Call(call_tabu_hash,
          amat = amat,
          nodes = nodes,
          tabu.list = tabu.list,
          current = current)

    # set up the score cache (BEWARE: in place modification!).
    .Call(call_score_cache_fill,
          nodes = nodes,
          data = x,
          network = start,
          score = score,
          extra = extra.args,
          reference = reference.score,
          equivalence = score.equivalence && optimized,
          decomposability = score.decomposability,
          updated = (if (optimized) updated else seq(length(nodes)) - 1L),
          amat = amat,
          cache = cache,
          blmat = blmat,
          debug = debug)

    # select which arcs should be tested for inclusion in the graph (hybrid
    # learning algorithms should hook the restrict phase here).
    to.be.added = arcs.to.be.added(amat = amat, nodes = nodes,
                    blacklist = blmat, whitelist = NULL, nparents = nparents,
                    maxp = maxp, arcs = FALSE)

    # get the best arc addition/removal/reversal.
    bestop = .Call(call_tabu_step,
                   amat = amat,
                   nodes = nodes,
                   added = to.be.added,
                   cache = cache,
                   reference = reference.score,
                   wlmat = wlmat,
                   blmat = blmat,
                   tabu.list = tabu.list,
                   current = current,
                   baseline = 0,
                   nparents = nparents,
                   maxp = maxp,
                   debug = debug)

    # the value FALSE is the canary value in bestop$op meaning "no operation
    # improved the network score"; reconsider prevously discarded solutions
    # and find the one that causes the minimum decrease in the network score.
    if (bestop$op == FALSE) {

      if (loss.iter >= max.loss.iter) {

        # reset the return value to the best network ever found.
        start = best.network

        if (debug) {

          cat("----------------------------------------------------------------\n")
          cat("* maximum number of iterations without improvements reached, stopping.\n")
          cat("* best network ever seen is:\n")
          print(best.network)

        }#THEN

        break

      }#THEN
      else {

        # increase the counter of the iterations without improvements.
        loss.iter = loss.iter + 1

      }#ELSE

      if (debug) {

        cat("----------------------------------------------------------------\n")
        cat("* network score did not increase (for", loss.iter,
              "times), looking for a minimal decrease :\n")

      }#THEN

      bestop = .Call(call_tabu_step,
                     amat = amat,
                     nodes = nodes,
                     added = to.be.added,
                     cache = cache,
                     reference = reference.score,
                     wlmat = wlmat,
                     blmat = blmat,
                     tabu.list = tabu.list,
                     current = current,
                     baseline = -Inf,
                     nparents = nparents,
                     maxp = maxp,
                     debug = debug)

      # it might be that there are no more legal operations.
      if (bestop$op == FALSE) {

       if (debug) {

         cat("----------------------------------------------------------------\n")
         cat("* no more possible operations.\n")
         cat("@ stopping at iteration", iter, ".\n")

       }#THEN

       # reset the return value to the best network ever found.
       if (loss.iter > 0)
         start = best.network

       break

     }#THEN

    }#THEN
    else {

      if (sum(reference.score) > best.score)
        loss.iter = 0

    }#ELSE

    # update the network structure.
    start = arc.operations(start, from = bestop$from, to = bestop$to,
              op = bestop$op, check.cycles = FALSE, check.illegal = FALSE,
              update = TRUE, debug = FALSE)

    # set the nodes whose cached score deltas are to be updated.
    if (bestop$op == "reverse")
      updated = which(nodes %in% c(bestop$from, bestop$to)) - 1L
    else
      updated = which(nodes %in% bestop$to) - 1L

    if (debug) {

      # update the test counter of the network; very useful to check how many
      # score comparison has been done up to now.
      start$learning$ntests = test.counter()

      cat("----------------------------------------------------------------\n")
      cat("* best operation was: ")
      if (bestop$op == "set")
        cat("adding", bestop$from, "->", bestop$to, ".\n")
      else if (bestop$op == "drop")
        cat("removing", bestop$from, "->", bestop$to, ".\n")
      else
        cat("reversing", bestop$from, "->", bestop$to, ".\n")
      cat("* current network is :\n")
      print(start)
      cat("* current score:", sum(reference.score), "\n")
      cat(sprintf("* best score up to now: %s (delta: %s)\n", format(best.score),
        format(sum(reference.score) - best.score)))

    }#THEN

    # check the current iteration index against the max.iter parameter.
    if (iter >= max.iter) {

      if (debug)
        cat("@ stopping at iteration", max.iter, ".\n")

      # reset the return value to the best network ever found.
      if (loss.iter > 0)
        start = best.network

      break

    }#THEN
    else iter = iter + 1

  }#REPEAT

  return(start)

}#TABU.SEARCH

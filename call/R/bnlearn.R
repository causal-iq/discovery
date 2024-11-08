#
#   Interface to R bnlearn package
#
bnlearn = function(method, params) {
    detach("package:stats", unload = TRUE)  # prevent name clash with bnlearn
    library(bnlearn)

    # switch control into method specified by method parameter

    return (switch(method,
                   "dagscore" = dagscore(params),
                   "citest" = citest(params),
                   "pdag2cpdag" = pdag2cpdag(params),
                   "learn" = bnlearn_learn(params),
                   "compare" = bnlearn_compare(params),
                   "import" = bnlearn_import(params),
            ))
}

#
#   Returns data as an R data data frame whether data provided as a Python
#   DataFrame or is name of file containing data.
#
get_data = function(params) {
    
    if (!is.null(params[['data']])) {  # must use [[]] otherwise will match datafile !!
        data = data.frame(params$data, stringsAsFactors=TRUE);
    
    } else if (params$dstype == 'categorical') {
    
        data = read.csv(file=params$datafile, header=TRUE, colClasses="factor",
                        na.strings='*');
    } else {
        data = read.csv(file=params$datafile, header=TRUE, na.strings='*');
        
    }
    return(data)
}

#
#   Scores DAG against data
#
dagscore = function(params) {

    dag = model2network(params$dag);  # instantiate the DAG object
    data = get_data(params)           # get data as R data frame

    scores = list();
    # print(params)
    for (type in params$types) {
        # print(type)
        if (type == 'bde' || type == 'bds') {
            
            # iss and prior score parameters supported for bde & bds

            scores[type] = score(dag, data, type=type, iss=params$iss,
                                 prior=params$prior);
        } else if ((type == 'aic' || type == 'bic') && params$k != 1) {
            
            # k complexity parameter multiplier supported for bic and aic

            options(warn = -1)  # suppress warnings for non-standard k
            k = params$k
            if (type == 'bic') {
                k = 0.5 * k * log(nrow(data))
            }
            scores[type] = score(dag, data, type=type, k=k);

        } else {
            scores[type] = score(dag, data, type=type, debug=TRUE);
        }
    }
    return (scores);
}

#
#   Perform multiple CI tests on data
#
citest = function(params) {
    data = get_data(params);  # get data as R data frame
    
    results = list();  # results will be returned as list of lists
    for (type in params$types) {  # loop over test types requested
        if (length(params$z) == 0) {
            result = ci.test(params$x, params$y, data = data, test = type);
        } else {
            result = ci.test(params$x, params$y, params$z, data = data, 
                             test = type);            
        }
        results[[type]] = list(statistic=as.numeric(result$statistic),
                               df=as.numeric(result$parameter),
                               p_value=as.numeric(result$p.value));
    }
    return (results);
}

#
#   Return CPDAG for supplied PDAG
#
#   params$nodes: [str] nodes names in format "[n1][n2]...."
#   params$edges: [list] of edges in format [n1, '->', n2] or [n1, '-', n2]
#
pdag2cpdag = function(params) {
    
    #   Set up bn object with no edges

    pdag = model2network(params$nodes)
    
    #   Add in arcs or undirected edges

    for (edge in params$edges) {
        if (edge[2] == '->') {
            pdag = set.arc(pdag, edge[1], edge[3])
        } else {
            pdag = set.edge(pdag, edge[1], edge[3])            
        }
    }
    
    return (arcs(cpdag(pdag)))
}


#
#   
#
#   Compares graph against a reference one. Note bnlearn converts graphs
#   to CPDAGs before doing the comparison
#
#   params$nodes: [str] nodes names in format "[n1][n2]...."
#   params$edges: [list] of edges in format [n1, '->', n2] or [n1, '-', n2]
#   params$ref: [list] of ref edges in format [n1, '->', n2] or [n1, '-', n2]
#
bnlearn_compare = function(params) {
    
    #   Set up bn objects with no edges

    pdag = model2network(params$nodes)
    ref = model2network(params$nodes)
    
    #   Add in arcs or undirected edges

    for (edge in params$edges) {
        if (edge[2] == '->') {
            pdag = set.arc(pdag, edge[1], edge[3])
        } else {
            pdag = set.edge(pdag, edge[1], edge[3])            
        }
    }
    for (edge in params$ref) {
        if (edge[2] == '->') {
            ref = set.arc(ref, edge[1], edge[3])
        } else {
            ref = set.edge(ref, edge[1], edge[3])            
        }
    }
    
    metrics = compare(ref, pdag)
    metrics['shd'] = shd(ref, pdag)
    
    return (metrics)
}

bnlearn_import = function(params) {
    cat(sprintf('\n* Importing BN ...\n'))
    load(params$rda)
    return (bn)
}

#
#   Learns a graph using the specified algorithm
#
#   params$algorithm: [str] learning algorithm to use
#   params$datafile: [str] data file name
#   params$score: [str] score used
#   params$whitelist: [dict] from, to defining required arcs
#   params$blacklist: [dict] from, to defining prohibited arcs
#
bnlearn_learn = function(params) {
    data = get_data(params);  # get data as R data frame
    
    cat(sprintf('\n* Running bnlearn algorithm %s ...\n', params$algorithm))

    # Set up any whitelist and blacklist

    if (length(params$whitelist$from) == 0) {
        whitelist = NULL        
    } else {
        whitelist = as.data.frame(params$whitelist)
    }
    if (length(params$blacklist$from) == 0) {
        blacklist = NULL        
    } else {
        blacklist = as.data.frame(params$blacklist)
    }
    cat('\nWhitelist param is:\n')
    print(whitelist)
    cat('\nBlacklist param is:\n')
    print(blacklist)
    cat('\n')
    
    start = proc.time()
    options(warn = -1)  # suppress warnings - specifically those in PC learning
    if (exists('test', params)) {
        test = params$test
        alpha = params$alpha
        cat(sprintf('    test = %s\n    alpha = %f\n', test, alpha))        
    }
    score = params$score
    cat(sprintf('    score = %s\n', score))

    if (params$algorithm == 'pc.stable') {
        graph = pc.stable(data, alpha=params$alpha, debug=TRUE,
                          whitelist=whitelist, blacklist=blacklist)
    } else if (params$algorithm == 'gs') {
        graph = gs(data, alpha=params$alpha, debug=TRUE,
                   whitelist=whitelist, blacklist=blacklist)        
    } else if (params$algorithm == 'inter.iamb') {
        graph = inter.iamb(data, alpha=params$alpha, debug=TRUE,
                           whitelist=whitelist, blacklist=blacklist)        
    } else if (exists('k', params)) {
        k = params$k
        cat(sprintf('    k = %s\n', k))
        if (params$score == 'bic') {
            k = 0.5 * k * log(nrow(data))
        }
        graph = switch(params$algorithm,
                       "hc" = hc(data, score=score, k=k, debug=TRUE,
                                 whitelist=whitelist, blacklist=blacklist),
                       "tabu" = tabu(data, score=score, k=k, debug=TRUE,
                                     whitelist=whitelist, blacklist=blacklist),
                       "h2pc" = h2pc(data, 
                                     maximize.args=list(score=score, k=k), 
                                     restrict.args=list(alpha=params$alpha),
                                     whitelist=whitelist, blacklist=blacklist,
                                     debug=TRUE),
                       "mmhc" = mmhc(data, 
                                     maximize.args=list(score=score, k=k), 
                                     restrict.args=list(alpha=params$alpha),
                                     whitelist=whitelist, blacklist=blacklist,
                                     debug=TRUE))
        
    } else if (exists('iss', params)){
        iss = params$iss
        prior = params$prior
        cat(sprintf('    iss = %f\n    prior = %s\n', iss, prior))        
        graph = switch(params$algorithm,
                       "hc" = hc(data, score=score, iss=iss, prior=prior,
                                 whitelist=whitelist, blacklist=blacklist,
                                 debug=TRUE),
                       "tabu" = tabu(data, score=score, iss=iss, prior=prior,
                                     whitelist=whitelist, blacklist=blacklist,
                                     debug=TRUE),
                       "h2pc" = h2pc(data, 
                                     maximize.args=list(score=score, iss=iss,
                                                        prior=prior), 
                                     restrict.args=list(alpha=params$alpha),
                                     whitelist=whitelist, blacklist=blacklist,
                                     debug=TRUE),
                       "mmhc" = mmhc(data, 
                                     maximize.args=list(score=score, iss=iss,
                                                        prior=prior), 
                                     restrict.args=list(alpha=params$alpha),
                                     whitelist=whitelist, blacklist=blacklist,
                                     debug=TRUE))
        
    } else {
        graph = switch(params$algorithm,
                       "hc" = hc(data, score=score, debug=TRUE,
                                 whitelist=whitelist, blacklist=blacklist),
                       "tabu" = tabu(data, score=score, debug=TRUE,
                                     whitelist=whitelist, blacklist=blacklist),
                       "h2pc" = h2pc(data, debug=TRUE,
                                     maximize.args=list(score=score), 
                                     restrict.args=list(alpha=params$alpha),
                                     whitelist=whitelist, blacklist=blacklist),
                       "mmhc" = mmhc(data, debug=TRUE,
                                     maximize.args=list(score=score), 
                                     restrict.args=list(alpha=params$alpha),
                                     whitelist=whitelist, blacklist=blacklist))
    }
    time = proc.time() - start
    cat('\n\n')

    print(time)

    return (list(arcs=arcs(graph), nodes=nodes(graph), elapsed=time[[3]]))
}

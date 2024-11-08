#
#   test package supporting testing of call/R
#
test = function(method, params) {
    return (switch(method,
                   "echo" = echo(params),
                   "error" = error(params)
            ))
}

#
#   Just echoes back the parameters for testing
#
echo = function(params) {
    print(params)
    return(params)
}

#
#   Calls stop to simulate abnormal termination for testing
#
error = function(params) {
    stop('Simulated abnormal termination')
}

#
#   Entry point into R code
#
library(rjson);

# Get and check command line argument which specifies unique call id which
# is uded in a names of input and outut JSON files

args = commandArgs(trailingOnly = TRUE);
if (length(args) != 1) {
    stop(sprintf('Bad args passed to main.R: "%s"', args));
}
id = args[1];

# open and check input file created by Python dispatch_r method

input = fromJSON(file=sprintf('call/R/tmp/%s.in.json', id))
if (is.null(input$package) || is.null(input$method)) {
    stop('package and/or method not specified');
}

# load the appropriate "package" and call its entry point to execute method

result = switch(input$package,
                'test' = {source('call/R/test.R');
                          test(input$method, input$params)},
                'bnlearn' = {source('call/R/bnlearn.R');
                             bnlearn(input$method, input$params)});
if (is.null(result)) {
    stop(sprintf('Unknown package:method specified: %s:%s',
                 input$package, input$method))
}

# write result as JSON file which will be read by Python dispatch_r method

write(toJSON(result), file=sprintf('call/R/tmp/%s.out.json', id))

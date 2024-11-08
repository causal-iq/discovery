
# Variable Order Experiments - Simple Linear Example with Asia

library(bnlearn)

# Get bnlearn's built in Asia dataset with 5K rows

data(asia)
cat("\nDefault ordering of Asia dataset is optimal:\n")
str(asia)

# Specify and plot the reference Asia network

ref <- model2network("[A][S][T|A][L|S][B|S][E|T:L][X|E][D|E:B]")
graphviz.plot(ref, layout="dot", shape="ellipse", main="Asia reference graph")

# Learn graphs with HC and pc.stable for this optimally ordered graph

hc_opt <- hc(asia)
hc_opt_diffs <- compare(ref, hc_opt)
hc_opt_shd <- shd(ref, hc_opt)
sub = sprintf("TP=%i, FP=%i, FN=%i, SHD=%i\nwith variable ordering: %s", 
              hc_opt_diffs$tp, hc_opt_diffs$fp, 
              hc_opt_diffs$fn, hc_opt_shd,
              paste(colnames(asia), collapse=", "))
graphviz.plot(hc_opt, layout="dot", shape="ellipse", 
              main="Asia learnt using HC with optimal ordering",
              sub=sub)

pc_opt <- pc.stable(asia)
pc_opt_diffs <- compare(ref, pc_opt)
pc_opt_shd <- shd(ref, pc_opt)
sub = sprintf("TP=%i, FP=%i, FN=%i, SHD=%i\nwith variable ordering: %s", 
              pc_opt_diffs$tp, pc_opt_diffs$fp, 
              pc_opt_diffs$fn, pc_opt_shd,
              paste(colnames(asia), collapse=", "))
graphviz.plot(pc_opt, layout="dot", shape="ellipse", 
              main="Asia learnt using PC.Stable with optimal ordering",
              sub=sub)

# Change variable ordering to be worst - i.e. anti-topological

asia <- asia[, c("D", "X", "E", "B", "L", "T", "S", "A")]
str(asia)

# Learn graphs with HC and pc.stable for this worst ordered graph

hc_bad <- hc(asia)
hc_bad_diffs <- compare(ref, hc_bad)
hc_bad_shd <- shd(ref, hc_bad)
sub = sprintf("TP=%i, FP=%i, FN=%i, SHD=%i\nwith variable ordering: %s", 
              hc_bad_diffs$tp, hc_bad_diffs$fp, 
              hc_bad_diffs$fn, hc_bad_shd,
              paste(colnames(asia), collapse=", "))
graphviz.plot(hc_bad, layout="dot", shape="ellipse", 
              main="Asia learnt using HC with worst ordering",
              sub=sub)

pc_bad <- pc.stable(asia)
pc_bad_diffs <- compare(ref, pc_bad)
pc_bad_shd <- shd(ref, pc_bad)
sub = sprintf("TP=%i, FP=%i, FN=%i, SHD=%i\nwith variable ordering: %s", 
              pc_bad_diffs$tp, pc_bad_diffs$fp, 
              pc_bad_diffs$fn, pc_bad_shd,
              paste(colnames(asia), collapse=", "))
graphviz.plot(pc_bad, layout="dot", shape="ellipse", 
              main="Asia learnt using PC.Stable with worst ordering",
              sub=sub)
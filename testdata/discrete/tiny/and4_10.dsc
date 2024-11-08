belief network "and4_10"
node X1 {
  type : discrete [ 2 ] = { "0", "1" };
}
node X2 {
  type : discrete [ 2 ] = { "0", "1" };
}
node X3 {
  type : discrete [ 2 ] = { "0", "1" };
}
node X4 {
  type : discrete [ 2 ] = { "0", "1" };
}
probability ( X1 ) {
   0.6666667, 0.3333333;
}
probability ( X3 ) {
   0.3333333, 0.6666667;
}
probability ( X2 | X1, X3 ) {
  (0, 0) : 0.6666667, 0.3333333;
  (1, 0) : 0.5555556, 0.4444444;
  (0, 1) : 0.4444444, 0.5555556;
  (1, 1) : 0.3333333, 0.6666667;
}
probability ( X4 | X2 ) {
  (0) : 0.6666667, 0.3333333;
  (1) : 0.3333333, 0.6666667;
}

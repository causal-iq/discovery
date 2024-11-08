belief network "ab_ac"
node A {
  type : discrete [ 2 ] = { "0", "1" };
}
node B {
  type : discrete [ 2 ] = { "0", "1" };
}
node C {
  type : discrete [ 2 ] = { "0", "1" };
}
probability ( A ) {
   0.3, 0.7;
}
probability ( B | A ) {
  (0) : 0.1, 0.9;
  (1) : 0.8, 0.2;
}
probability ( C | A ) {
  (0) : 0.6, 0.4;
  (1) : 0.3, 0.7;
}

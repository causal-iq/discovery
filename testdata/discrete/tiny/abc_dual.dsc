belief network "abc_dual"
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
   0.8, 0.2;
}
probability ( B | A ) {
  (0) : 0.5, 0.5;
  (1) : 0.6, 0.4;
}
probability ( C | A, B ) {
  (0, 0) : 0.6, 0.4;
  (0, 1) : 0.3, 0.7;
  (1, 0) : 0.2, 0.8;
  (1, 1) : 0.1, 0.9;
}

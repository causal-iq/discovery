belief network "ab_cb"
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
   0.7, 0.3;
}
probability ( C ) {
   0.2, 0.8;
}
probability ( B | A, C ) {
  (0, 0) : 0.1, 0.9;
  (1, 0) : 0.5, 0.5;
  (0, 1) : 0.9, 0.1;
  (1, 1) : 0.7, 0.3;
}

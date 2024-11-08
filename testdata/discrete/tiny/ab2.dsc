belief network "ab"
node A {
  type : discrete [ 2 ] = { "0", "1" };
}
node B {
  type : discrete [ 3 ] = { "0", "1", "2" };
}
probability ( A ) {
   0.75, 0.25;
}
probability ( B | A ) {
  (0) : 0.5, 0.3, 0.2;
  (1) : 0.25, 0.50, 0.25;
}

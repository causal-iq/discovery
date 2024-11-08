belief network "ba"
node A {
  type : discrete [ 2 ] = { "0", "1" };
}
node B {
  type : discrete [ 2 ] = { "0", "1" };
}
probability ( B ) {
   0.75, 0.25;
}
probability ( A | B ) {
  (0) : 0.5, 0.5;
  (1) : 0.25, 0.75;
}

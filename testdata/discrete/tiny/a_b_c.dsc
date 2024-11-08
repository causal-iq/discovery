belief network "a_b_c"
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
   0.5, 0.5;
}
probability ( B ) {
   0.2, 0.8;
}
probability ( C ) {
   0.4, 0.6;
}

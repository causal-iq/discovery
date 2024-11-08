/*
 *  Example 1 from operators module
 */

/*  
    C variables start with letter or _, then any cobo of letters, numbers and _
    Variables should be decared before you use it
*/

#include <stdio.h>

int main (int argc, char *argv[]) {
    int a = 10;
    printf("a is initially %d", a);
    printf("\na with pre-fix ++a is %d", ++a);
    printf("\na with pre-fix a++ is %d", a++);
    printf("\nFinally a is %d", a);

    /* Bitwise operators */

    unsigned char ten = 10;   // 0000 1010
    unsigned char seven = 7;  // 0000 0101
    printf("\n1010 & 0111 is %d (0010)", ten & seven);
    printf("\n1010 | 0111 is %d (1111)", ten | seven);
    printf("\n1010 ^ 0111 is %d (1101)", ten ^ seven);
    printf("\n1010 ^ 0111 is %d (1101)", ten ^ seven);
    printf("\n1010 ^ 0111 is %d (1101)", ten ^ seven);
    printf("\n1010 >> 2 is %d (0010)", ten >> 2);
    printf("\n1010 << 2 is %d (10100)", ten << 2);
    
    /* Cast and sizeof operators */

    float f = 12.7;
    printf("\n%f cast to int is %i", f, (int)f);
    printf("\n%d cast to float is %f", seven, (float)seven);
    printf("\nsizeof(ten) is %d bytes", sizeof(ten));
    printf("\nsizeof(short) is %d bytes", sizeof(short));
    printf("\nsizeof(int) is %d bytes", sizeof(int));
    printf("\nsizeof(long) is %d bytes", sizeof(long));
    printf("\nsizeof(float) is %d bytes", sizeof(float));
    printf("\nsizeof(double) is %d bytes", sizeof(double));
}
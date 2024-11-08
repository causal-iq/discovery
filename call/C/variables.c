/*
 *  Example 1 from variables module
 */

/*  
    C variables start with letter or _, then any cobo of letters, numbers and _
    Variables should be decared before you use it
*/

#include <stdio.h>
#include <stdbool.h>                // can use bool datatype & true, false

int main(int argc, char *argv[]) {
    int a = 3;                      // 4 or 8 bytes
    int rgbColor = 0xFFEF0D;        // hexadecimal literal
    float f = 12.1;                 // 4 bytes generally
    float g = 1.7e4;                // 4 bytes generally, literal is double
    float f2 = -2.0f;               // literal can be explicitly set a float
    double d = -9.7;                // 8 bytes generally
    _Bool b1 = 0;                   // _Bool is basic type, 0 indicates false
    _Bool b2 = 1;                   // 1 indicates true
    bool b3 = true;                 // from stdbool.h
    
    short s1 = 0;                   // int, uses less memory
    long l1 = 22836309609L;         // long int
    unsigned u = 2;                 // unsigned integer
    long double ld = 2.7e99L;       // long double
    
    enum primaryColor { red, yellow, blue };  // define enum, ints from 0
    enum assign { right=4, wrong=3, maybe};
    
    enum primaryColor myColor = blue, AnotherColor;  // declare enum variables
    enum assign e = maybe;
    
    char chr = 'X';                 // single quotes for char
    char letter = 65;             // can assign using numbers but poor style
    
    char nl ='\n';                   // newline
    char cr = '\r';                 // carriage return
    
    printf("\na is %d", a);
    printf("\nb3 is %d", b3);
    printf("\nmyColor is %d", myColor);  // blue is pos 2 in enum list
    printf("\ne is %d", e);  // maybe is 1 after wrong - NOTE ClASH
    printf("\nletter is %c", letter);
    printf("\nNumber of arguments: %d", argc);
    printf("\nProgram name is %s", argv[0]);
}
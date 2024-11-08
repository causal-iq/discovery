/*
 *  First example C program from Udemy course
 */

/*  Include header file - angle brackets mean it is a system header
    otherwise use double quotes. Header files typically just type
    and function definitions */

#include <stdio.h>  

int main() {
    char str[100];                      // string 
    int i;                              // integer
    double d;                           // double precision
    
    printf("Enter an integer, word and float: ");
    scanf("%d %s %lf", &i, str, &d);            // int requires &, str does not
    
    printf("\nInteger was %d, string was '%s', double %lf", i, str, d);  
    return 0;
}

#include <stdio.h>

int main(int argc, char *argv[]) {
    int array[3];
    int i;
    
    printf("\nFirst element is %d", array[0]);
    printf("\nSecond element is %d", array[1]);
    printf("\nThird element is %d", array[2]);
    printf("\nNon-existent fourth element is %d", array[3]);
    printf("\ni is %d\n", i);
    
    int multi[2][3] = {{1, 2}, {3}};  // not initialising all elements
    
    for (int row = 0; row < 2; row++) {
        printf("\nRow %d is ", row);
        for (int col = 0; col < 3; col++) {
            printf("%d ", multi[row][col]);
        }
    }
}
#define _USE_MATH_DEFINES // Necessary on some systems for math constants
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>


/* column-major coordinates for an arbitrary matrix. */
#define CMC(i, j, nrow) ((i) + (j) * (nrow))

/* Structures in data.structures.h  */

/* columns (variables) metadata common to all types of data tables. */
typedef struct {

  int nobs;             /* number of observations. */
  int ncols;            /* number of columns. */
  const char **names;   /* column names (optional). */
  /* flags *flag;          not used in cwpost */

} meta;

/* data table for gaussian data. */
typedef struct {

  meta m;               /* metadata. */
  double **col;         /* pointers to the continuous columns. */
  double *mean;         /* means of the continuous columns (optional). */

} gdata;

/* structures in covariance.h */

/* covariance matrix, with additional fields to carry around its own SVD
 * decomposition and dimension. */
typedef struct {

  int dim;     /* dimension of the covariance matrix. */
  double *mat; /* pointer to the matrix. */
  double *u;   /* SVD decomposition, U left matrix, for c_svd(). */
  double *d;   /* SVD decomposition, vector for U's diagonal, for c_svd(). */
  double *vt;  /* SVD decomposition, V^t right matrix, for c_svd(). */

} covariance;

float cwpost(double *xx, double **z, int p, int data_ncols, int nobs, 
             double alpha_mu, double alpha_w);
gdata gdata_from_SEXP(double **df, int offset, int nobs, int ncols);
gdata empty_gdata(int nobs, int ncols);
void gdata_cache_means(gdata *dt, int offset);
void c_meanvec(double **data, double *mean, int nrow, int ncol, int first);
double c_det(double *matrix, int rows);
void *Calloc1D(size_t R, size_t size);
void print_vector(double *vector, size_t size, int width, int precision);
void print_array(double **array, int nrows, int ncols);


/**
 * Print out a vector in readable format - from ChatGPT
 * 
 * @param *vector pointer to start of vector
 * @param size number of values in vector
 * @param width width of each printed value
 * @param precision precision of each printed value
 */
void print_vector(double *vector, size_t size, int width, int precision) {
    printf("["); // Start of the vector
    for (size_t i = 0; i < size; i++) {
        // Print each value with fixed width and precision
        printf("%*.*f", width, precision, vector[i]);
        if (i < size - 1)
            printf(", "); // Add a comma and space for all but the last element
    }
    printf("]"); // End of the vector
}


/**
 * Print out a 2D array row by row - Ken Kitson
 * 
 * @param arr array to print out, pointers to COLUMNS
 */
void print_array(double **array, int nrows, int ncols) {
  double *row = malloc(ncols * sizeof(double));
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      row[j] = array[j][i];
    }
    if (i == 0) {
      printf(" [");
    } else {
      printf(",\n  ");
    }
    print_vector(row, ncols, 7, 4);
  }
  printf("]\n");
  free(row);
}

/**
 * Allocates memory for 1-D array - from allocations.c
 * 
 * @param R number of items in array
 * @param size size of individual item in bytes
 * 
 * @return void * pointer to start of array
 * 
 */
void *Calloc1D(size_t R, size_t size) {

void *p = NULL;

  if (R == 0)
    return NULL;

  p = calloc(R, size);

  if (!p)
    printf("***unable to allocate a %d array.", R);

  return(p);

}/*CALLOC1D*/


/**
 * Initialises an empty gdata struct for Gaussian data - from data.table.c
 * 
 * @param nobs number of observations (rows)
 * @param ncols number of columns (variables)
 * 
 * @return gdata initialised, empty gdata structure
 */
gdata empty_gdata(int nobs, int ncols) {

gdata dt = { 0 };

  dt.m.nobs = nobs;
  dt.m.ncols = ncols;
  dt.col = Calloc1D(ncols, sizeof(double *));

  return dt;

}/*EMPTY_GDATA*/

/**
 * Create a data table from Gaussian data. - from data.table.c
 * 
 * @param df pointer to array of data
 * @param offset number of empty columns to insert before data itself
 * @param nobs number of observations (rows)
 * @param ncols number of columns (variables) in df
 */

gdata gdata_from_SEXP(double **df, int offset, int nobs, int ncols) {

int i = 0;
gdata dt = { 0 };

  /* initialize the object, taking care of the corner case of zero-columns data
   * frames. */
  if (ncols == 0)
    dt = empty_gdata(0, ncols + offset);
  else
    dt = empty_gdata(nobs, ncols + offset);

  for (i = 0; i < ncols; i++)
    dt.col[i + offset] = df[i];

  return dt;

}/*GDATA_FROM_SEXP*/


/**
 * Compute and cache column means in gdata structure - from data.table.c
 * 
 * @param dt Gaussuan data where means will be updated
 * @param offset start column to generate means for
 */
void gdata_cache_means(gdata *dt, int offset) {

  (*dt).mean = Calloc1D((*dt).m.ncols, sizeof(double));
  c_meanvec((*dt).col, (*dt).mean, (*dt).m.nobs, (*dt).m.ncols, offset);

}/*CACHE_MEANS*/

/**
 * Compute means for a set columns - from variance.c
 * 
 * @param data array of data column pointers)
 * @param mean vector where means will be placed
 * @param nrow number of rows
 * @param ncol number of columns
 * @param first starting column index
 */
void c_meanvec(double **data, double *mean, int nrow, int ncol, int first) {

int i = 0, j = 0;
long double sum = 0;

  for (i = first; i < ncol; i++) {

    for (j = 0, sum = 0; j < nrow; j++)
      sum += data[i][j];

    mean[i] = sum / nrow;

  }/*FOR*/

}/*C_MEANVEC*/

/**
 * Creae an empty covariance matrix structure - from covariance.c
 */
covariance new_covariance(int dim, bool decomp) {

covariance cov = { 0 };

  cov.mat = Calloc1D(dim * dim, sizeof(double));
  cov.dim = dim;

  if (decomp) {

    cov.u = Calloc1D(dim * dim, sizeof(double));
    cov.d = Calloc1D(dim, sizeof(double));
    cov.vt = Calloc1D(dim * dim, sizeof(double));

  }/*THEN*/

  return cov;

}/*NEW_COVARIANCE*/

/* fill a covariance matrix. */
void c_covmat(double **data, double *mean, int nrow, int ncol,
    covariance cov, int first) {

int i = 0, j = 0, k = 0;
long double temp = 0;

  /* special case: with zero and one observations, the covariance
   * matrix is all zeroes. */
  if (nrow <= 1) {

    memset(cov.mat, '\0', ncol * ncol * sizeof(double));

    return ;

  }/*THEN*/

  /* compute the actual covariance. */
  for (i = first; i < ncol; i++) {

    for (j = i; j < ncol; j++) {

      for (k = 0, temp = 0; k < nrow; k++)
        temp += (data[i][k] - mean[i]) * (data[j][k] - mean[j]);

      /* fill in the symmetric element of the matrix. */
      cov.mat[CMC(j, i, ncol)] = cov.mat[CMC(i, j, ncol)] =
        (double)(temp / (nrow - 1));

    }/*FOR*/

  }/*FOR*/

}/*C_COVMAT*/

void covariance_drop_variable(covariance *full, covariance *sub, int to_drop) {

  for (int j = 0, k = 0; j < (*full).dim; j++)
    for (int i = 0; i < (*full).dim; i++)
      if ((i != to_drop) && (j != to_drop))
        (*sub).mat[k++] = (*full).mat[CMC(i, j, (*full).dim)];

  (*sub).dim = (*full).dim - 1;

}/*COVARIANCE_DROP_COLUMN*/

/**
 * Compute determinant of a flattened column-major-order covariance matrix. 
 * This version just supports 1x1, 2x2 and 3x3 matrices. From Chat-GPT.
 * 
 * @param matrix pointer to flattened covariance matrix
 * @param rows number of rows (and columns) in unflattened matrix
 */
double c_det(double *matrix, int rows) {
  if (rows == 1) {
    return matrix[0];
  } else if (rows == 2) {
    return matrix[0] * matrix[3] - matrix[1] * matrix[2];
  } else if (rows == 3) {
    return matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
           matrix[3] * (matrix[1] * matrix[8] - matrix[2] * matrix[7]) +
           matrix[6] * (matrix[1] * matrix[5] - matrix[2] * matrix[4]);
  } else {
    printf("\n*** c_det does not support > 2 parents\n");
    return 0.0;
  }
}

/**
 * Compute condtional posterior probability for Gaussian data using Wishart
 * priors - from wishart.posterior.c
 * 
 * @param xx vector of child data
 * @param z array of parent (columns) data
 * @param p number of parents
 * @param data_ncols number of columns (variables) in complete dataset
 * @param nobs number of observations (rows)
 * @param alpha_mu weight assigned to prior means
 * @param alpha_w weight assigned to prior variances
 * 
 */
float cwpost(double *xx, double **z, int p, int data_ncols, int nobs,
             double alpha_mu, double alpha_w) {

  double logprob = 0.0;
  double t = 0;
 
  printf("\ncwpost arguments:\n");
  printf("          xx: ");
  print_vector(xx, nobs, 7, 4);
  printf("\n          z:\n");
  print_array(z, nobs, p);
  printf("  data_ncols: %d\n", data_ncols);
  printf("        nobs: %d\n", nobs);
  printf("    alpha_mu: %.4f\n", alpha_mu);
  printf("     alpha_w: %.4f\n", alpha_w);

  /* first term. */
  logprob = 0.5 * (log(alpha_mu) - log(nobs + alpha_mu));
  printf("\nAfter 1st term: %.6f\n", logprob);

  /* Gamma_l ratio in the second term. */
  logprob += lgamma(0.5 * (nobs + alpha_w - data_ncols + p + 1)) -
             lgamma(0.5 * (alpha_w - data_ncols + p + 1));
  printf("After 2nd term: %.6f\n", logprob);

  /* leftover from the second term. */
  logprob -= 0.5 * nobs * log(M_PI);
  printf("After lef term: %.6f\n", logprob);

  /* third term, ratio of the determinants of the prior T matrices. */
  t = alpha_mu * (alpha_w - data_ncols - 1) / (alpha_mu + 1);

  logprob += 0.5 * (alpha_w - data_ncols + p + 1) * (p + 1) * log(t) -
             0.5 * (alpha_w - data_ncols + p) * (p) * log(t);
  printf("After 3rd term: %.6f\n", logprob);

  /* third term, ratio of the determinants of the posterior R matrices. */
  // gdata dtx = gdata_from_SEXP(z, 1);

  gdata dtx = gdata_from_SEXP(z, 1, nobs, p);
  dtx.col[0] = xx;
  printf("dtx (child and parent data) is:\n");
  print_array(dtx.col, nobs, p + 1);
  gdata_cache_means(&dtx, 0);
  printf("Column means are: ");
  print_vector(dtx.mean, p + 1, 7, 3);
  covariance R = new_covariance(dtx.m.ncols, false);
  printf("\nEmpty covariance matrix R[%dx%d] created", R.dim, R.dim);
  covariance Rtilde = new_covariance(dtx.m.ncols - 1, false);
  printf("\nEmpty covariance matrix R~[%dx%d] created", Rtilde.dim, Rtilde.dim);

  /* compute the rescaled covariance matrix. */
  c_covmat(dtx.col, dtx.mean, dtx.m.nobs, dtx.m.ncols, R, 0);
  for (int i = 0; i < R.dim * R.dim; i++)
    R.mat[i] *= nobs - 1;
  printf("\n\nFlattened re-scaled covariance matrix R is:\n");
  print_vector(R.mat, R.dim * R.dim, 7, 4);

  /* add the prior matrix, diagonal with elements t. */
  for (int i = 0; i < R.dim; i++)
    R.mat[CMC(i, i, R.dim)] += t;
  printf("\n\nR with prior added in:\n");
  print_vector(R.mat, R.dim * R.dim, 7, 4);
  printf("\nDeterminant is %.3f", c_det(R.mat, R.dim));

  /* add the outer product of the difference in observed and prior means. */
  // for (int i = 0; i < R.dim; i++)
  //  for (int j = 0; j < R.dim; j++) {
  //
  //    R.mat[CMC(i, j, R.dim)] += (nobs * alpha_mu) / (nobs + alpha_mu) *
  //                                (dtx.mean[i] - nu[i]) * (dtx.mean[j] - nu[j]);
  // }/*FOR*/

  /* subset the rescaled covariance matrix to include only the parents. */
  covariance_drop_variable(&R, &Rtilde, 0);
  printf("\n\nR~ parents-only covariance matrix:\n");
  print_vector(Rtilde.mat, Rtilde.dim * Rtilde.dim, 7, 4);
  printf("\nDeterminant is %.3f", c_det(Rtilde.mat, Rtilde.dim));

  logprob += 0.5 * (nobs + alpha_w - data_ncols + p) *
               log(c_det(Rtilde.mat, Rtilde.dim));
  logprob -= 0.5 * (nobs + alpha_w - data_ncols + p + 1) *
               log(c_det(R.mat, R.dim));
  printf("\nBGE is %.6f", logprob);

}


int main() {

  /* Set up the test data for X and Y. Follow bnlearn convention of pointers 
     to start of each COLUMN unlike standard C practice. */

  int ncols = 3;
  int nobs = 3;

  double **data = malloc(ncols * sizeof(double *));  // ncols columns
  for (int j = 0; j < ncols; j++){
    data[j] = malloc(nobs * sizeof(double));
  }
  data[0][0] = 0.0;
  data[0][1] = 0.2;
  data[0][2] = 1.1;
  data[1][0] = 3.1;
  data[1][1] = 5.4;
  data[1][2] = 0.3;
  data[2][0] = 4.0;
  data[2][1] = 1.7;
  data[2][2] = 0.3;
  printf("\nData is:\n");
  print_array(data, nobs, ncols);

  /* Set child (xx) to col 1, parents (z) to cols 0 and 2 */

  double **z = malloc(2 * sizeof(double *));
  z[0] = data[0];
  z[1] = data[2];
  double *xx = data[1];

  /* compute conditional posterior probability */

  cwpost(xx, z, 2, ncols, nobs, 1.0, 5.0);

  /* free memory */

  free(z);
  for (int j; j < ncols; j++) {
    free(data[j]);
  }
  free(data);

  return 0;
}

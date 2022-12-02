/*
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>
#include "cblock.h"
#include <mpi.h>
using namespace std;

void printMat(const char mesg[], double *E, int m, int n);



//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//


extern control_block cb;


int nprocs = 1;
int current_rank = 0;

// Each processor only works on its portion of the matrix
int sub_m, sub_n;

int rankx, ranky;

int E_ROOT_TO_PERIPH = 4;
int R_ROOT_TO_PERIPH = 5;

double *recv_left, *recv_right, *send_left, *send_right;

void set_processor_gemoetry(int m, int n) {

  // currently assume that processor gemoetry is evenly divisible

  // current_rank will hold the rank number of the current process
  #ifdef _MPI_
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&current_rank);
  #endif

  
  sub_m = m / cb.py;
  sub_n = n / cb.px;

  rankx = current_rank / cb.px;
  ranky = current_rank % cb.px;

  recv_left = new double[sub_m];
	recv_right = new double[sub_m];
	send_left = new double[sub_m];
	send_right = new double[sub_m];
}


void init (double *E,double *E_prev,double *R,int m,int n){

  
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&current_rank);
  
  MPI_Request send_req_E[cb.px*cb.py];
  MPI_Request recieve_req_E[cb.px*cb.py];
 
  MPI_Request send_req_R[cb.px*cb.py];
  MPI_Request recieve_req_R[cb.px*cb.py]; 
  

  

  int nx=n, ny=m;
  int i;
  double *E_init;
  double *R_init;
  double *sub_E_init;
  double *sub_R_init;
  double *send_E_init;
  double *send_R_init;
  
  assert(E_init= (double*) memalign(16, sizeof(double)*nx*ny) );
  assert(R_init= (double*) memalign(16, sizeof(double)*nx*ny) );
  assert(sub_E_init= (double*) memalign(16, sizeof(double)*sub_m*sub_n) );
  assert(sub_R_init= (double*) memalign(16, sizeof(double)*sub_m*sub_n) );
  assert(send_E_init= (double*) memalign(16, sizeof(double)*(sub_m+2)*(sub_n+2)) );
  assert(send_R_init= (double*) memalign(16, sizeof(double)*(sub_m+2)*(sub_n+2)) );
  
  for (i=0; i<(sub_m+2)*(sub_n+2); i++){
    send_E_init[i] = 0;
    send_R_init[i] = 0;
  }
  
  if(current_rank==0)
  {
    for (i=0; i < (m)*(n); i++)
        E_init[i] = R_init[i] = 0;

    for (i = 0; i < m*n; i++) {
	    int colIndex = i % n;		// gives the base index (first row's) of the current index

        // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
	    if(colIndex < (n/2+1))
	    continue;

        E_init[i] = 1.0;
    }

    for (i = 0; i < m*n; i++) {
  	  int rowIndex = i / n;		// gives the current row number in 2D array representation
	    int colIndex = i % n;		// gives the base index (first row's) of the current index

        // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
	    if(rowIndex < (m/2+1))
	    continue;

        R_init[i] = 1.0;
    }
    
  
      for(int j=0; j<sub_m*sub_n; j++) {
        i=0;
        sub_E_init[j] = E_init[(i%cb.px)*sub_n + (i/cb.px)*n*sub_m + (j/sub_n)*n + (j%sub_n)];
        sub_R_init[j] = R_init[(i%cb.px)*sub_n + (i/cb.px)*n*sub_m + (j/sub_n)*n + (j%sub_n)];
      }
      for(int j =0; j<sub_m*sub_n; j++) {
        int sub_rowIndex = j/sub_n;
        int sub_colIndex = j%sub_n;
        i=0;
        send_E_init[(sub_rowIndex+1) *sub_n + sub_colIndex+1] = sub_E_init[j]; 
        send_R_init[(sub_rowIndex+1) *sub_n + sub_colIndex+1] = sub_R_init[j];
      }
    E_prev = send_E_init;
    R = send_R_init;

        
    for(i=1; i< cb.px*cb.py; i++) {
      for(int j=0; j<sub_m*sub_n; j++) {
        sub_E_init[j] = E_init[(i%cb.px)*sub_n + (i/cb.px)*n*sub_m + (j/sub_n)*n + (j%sub_n)];
        sub_R_init[j] = R_init[(i%cb.px)*sub_n + (i/cb.px)*n*sub_m + (j/sub_n)*n + (j%sub_n)];
      }
      for(int j =0; j<sub_m*sub_n; j++) {
        int sub_rowIndex = j/sub_n;
        int sub_colIndex = j%sub_n;
        send_E_init[(sub_rowIndex+1) *sub_n + sub_colIndex+1] = sub_E_init[j]; 
        send_R_init[(sub_rowIndex+1) *sub_n + sub_colIndex+1] = sub_R_init[j];
      }
      
      MPI_Isend(send_E_init, (sub_m+2)*(sub_n+2), MPI_DOUBLE, i, E_ROOT_TO_PERIPH, MPI_COMM_WORLD, send_req_E+i-1);
      MPI_Isend(send_R_init, (sub_m+2)*(sub_n+2), MPI_DOUBLE, i, R_ROOT_TO_PERIPH, MPI_COMM_WORLD, send_req_R+i-1);
    }          
  
  }//if ends      
        
  
  else
  {
    //MPI Receive Inital Conditions  
    MPI_Irecv(E_prev, (sub_m+2)*(sub_n+2), MPI_DOUBLE, 0, E_ROOT_TO_PERIPH, MPI_COMM_WORLD, recieve_req_E+current_rank-1);
    MPI_Irecv(R, (sub_m+2)*(sub_n+2), MPI_DOUBLE, 0, R_ROOT_TO_PERIPH, MPI_COMM_WORLD, recieve_req_R+current_rank-1);
  } 
  
  




#if 1
    // printMat("E_prev",E_prev,m,n);
    // printMat("R",R,m,n);
#endif
}//init ends

double *alloc1D(int m,int n){

    // m and n are padded with +2 in the calling function in apf.cpp
    set_processor_gemoetry(m-2, n-2);

    double *E;

    // Assign 2 extra rows and 2 extra columns for each sub-matrix portion
    // to hold ghost rows and/or padding.
    assert(E= (double*) memalign(16, sizeof(double)* (sub_m + 2) * (sub_n + 2)) );
    return(E);
}

void printMat(const char mesg[], double *E, int m, int n){
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m>34)
      return;
#endif
    printf("%s\n",mesg);
    for (i=0; i < (m+2)*(n+2); i++){
       int rowIndex = i / (n+2);
       int colIndex = i % (n+2);
       if ((colIndex>0) && (colIndex<n+1))
          if ((rowIndex > 0) && (rowIndex < m+1))
            printf("%6.3f ", E[i]);
       if (colIndex == n+1)
	    printf("\n");
    }
}

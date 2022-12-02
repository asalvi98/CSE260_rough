/*
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 *
 * Modified and  restructured by Scott B. Baden, UCSD
 *
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>
#include <mpi.h>
using namespace std;

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);

extern control_block cb;

// #ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
// __attribute__((optimize("no-tree-vectorize")))
// #endif

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//

extern int sub_m, sub_n;
extern int current_rank;
extern int rankx, ranky;
extern double *recv_left, *recv_right, *send_left, *send_right;

int NORTH_TO_SOUTH = 0;
int SOUTH_TO_NORTH = 1;
int EAST_TO_WEST = 2;
int WEST_TO_EAST = 3;


double L2Norm(double sumSq){
    double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

void copy_ghost_cells(double *E_prev) {
  int i, j;
  int num_msgs_to_be_recieved = 0;

  MPI_Request send[4];
  MPI_Request recieve[4];

  if (rankx == 0) {
    // Top most cells

    // Fill the topmost row of cell with values from 2 rows below it
    for (i = 0; i < (sub_n+2); i++) {
        E_prev[i] = E_prev[i + (sub_n+2)*2];
    }
  } else {

    // recieve the ghost cells for north border from the cell above
    MPI_Irecv(&E_prev[1], sub_n, MPI_DOUBLE, current_rank - cb.px, NORTH_TO_SOUTH, MPI_COMM_WORLD, recieve + num_msgs_to_be_recieved);
    num_msgs_to_be_recieved+=1;

    // send the cell's actual top row as ghost cells to the cell above
		MPI_Isend(&E_prev[(sub_n + 2) + 1], sub_n, MPI_DOUBLE, current_rank - cb.px, SOUTH_TO_NORTH, MPI_COMM_WORLD, send);

  }



  if (rankx == cb.py - 1) {
    // bottom most process

    // Fill the bottom-most row of cell with values from 2 rows above it
    for (i = ((sub_m+2)*(sub_n+2)-(sub_n+2)); i < (sub_m+2)*(sub_n+2); i++) {
        E_prev[i] = E_prev[i - (sub_n+2)*2];
    }
  } else {

    // recieve the ghost cells for south border from the cell below
    MPI_Irecv(&E_prev[(sub_m + 1)*(sub_n + 2) + 1], sub_n, MPI_DOUBLE, current_rank + cb.px, SOUTH_TO_NORTH, MPI_COMM_WORLD, recieve + num_msgs_to_be_recieved);
    num_msgs_to_be_recieved+=1;

    // Send the cell's actual bottom row as ghost cells to the cell below
		MPI_Isend(&E_prev[sub_m*(sub_n + 2) + 1], sub_n, MPI_DOUBLE, current_rank + cb.px, NORTH_TO_SOUTH, MPI_COMM_WORLD, send + 1);

  }

  if (ranky == 0) {
    //left most process

    // Fills the left most column with values 2 columns right of it
    for (i = 0; i < (sub_m+2)*(sub_n+2); i+=(sub_n+2)) {
        E_prev[i] = E_prev[i+2];
    }

  } else {

    // pack actual west-most column into contiguos array
    for (i = sub_n + 2 + 1, j = 0; j < sub_m; i += sub_n + 2, ++j)
		{
			send_left[j] = E_prev[i];
		}

    // recieve the ghost cells for west border from the cell on the left
    MPI_Irecv(recv_left, sub_m, MPI_DOUBLE, current_rank - 1, WEST_TO_EAST, MPI_COMM_WORLD, recieve + num_msgs_to_be_recieved);
    num_msgs_to_be_recieved+=1;

    // Send the cell's actual west-most column as ghost cells to the cell on the left
		MPI_Isend(send_left, sub_m, MPI_DOUBLE, current_rank - 1, EAST_TO_WEST, MPI_COMM_WORLD, send + 2);



    // MPI_Datatype west_column;
    // // west column: sub_m number of elements, each sub_n + 2 distance apart
    // MPI_Type_vector(sub_m, 1, sub_n + 2, MPI_DOUBLE, &west_column);
    // MPI_Type_commit(&west_column);
    //
    // // recieve the ghost cells for west border from the cell on the left
    // MPI_Irecv(&E_prev[sub_n + 2], 1, west_column, current_rank - 1, WEST_TO_EAST, MPI_COMM_WORLD, recieve + num_msgs_to_be_recieved);
    // num_msgs_to_be_recieved++;
    //
    // // Send the cell's actual west-most column as ghost cells to the cell on the left
		// MPI_Isend(&E_prev[(sub_n + 2) + 1], 1, west_column, current_rank - 1, EAST_TO_WEST, MPI_COMM_WORLD, send + 2);

  }


  if (ranky == cb.px - 1) {
    //right most process

    // Fills the right most column with values 2 columns left of it
    for (i = (sub_n+1); i < (sub_m+2)*(sub_n+2); i+=(sub_n+2)) {
        E_prev[i] = E_prev[i-2];
    }

  } else {

    // pack cell's actual east-most column into contiguos array
    for (i = sub_n + (sub_n + 2), j = 0; j < sub_m; i += (sub_n + 2), ++j)
		{
			send_right[j] = E_prev[i];
		}

    MPI_Irecv(recv_right, sub_m, MPI_DOUBLE, current_rank + 1, EAST_TO_WEST, MPI_COMM_WORLD, recieve + num_msgs_to_be_recieved);
    num_msgs_to_be_recieved+=1;

    // Send the cell's actual east-most column as ghost cells to the cell on the right
    MPI_Isend(send_right, sub_m, MPI_DOUBLE, current_rank + 1, WEST_TO_EAST, MPI_COMM_WORLD, send + 3);



    // MPI_Datatype east_column;
    // // east column: sub_m number of elements, each sub_n + 2 distance apart
    // MPI_Type_vector(sub_m, 1, sub_n + 2, MPI_DOUBLE, &east_column);
    // MPI_Type_commit(&east_column);
    //
    // // recieve the ghost cells for east border from the cell on the right
    // MPI_Irecv(&E_prev[sub_n + 2 + sub_n + 1], 1, east_column, current_rank + 1, EAST_TO_WEST, MPI_COMM_WORLD, recieve + num_msgs_to_be_recieved);
    // num_msgs_to_be_recieved++;
    //
    // // Send the cell's actual east-most column as ghost cells to the cell on the right
    // MPI_Isend(&E_prev[(sub_n + 2) + sub_n], 1, east_column, current_rank + 1, WEST_TO_EAST, MPI_COMM_WORLD, send + 3);

  }

  // proceed with differential eqn solving only when ghost cells have been recieved
  MPI_Status status[4];
  MPI_Waitall(num_msgs_to_be_recieved, recieve, status);

  // unpack left column packed arrays back into the original sub-matrix
  if (ranky != 0)
	{
		for (i = sub_n + 2, j = 0; j < sub_m; i += sub_n + 2, ++j)
			E_prev[i] = recv_left[j];
	}

  // unpack right column packed arrays back into the original sub-matrix
	if (ranky != cb.px - 1)
	{
		for (i = (sub_n + 1) + (sub_n + 2), j = 0; j < sub_m; i += sub_n + 2, ++j)
			E_prev[i] = recv_right[j];
	}

return;

}

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){

 // Simulated time is different from the integer timestep number
 double t = 0.0;

 int i, j;

 double *E = *_E, *E_prev = *_E_prev;
 double *R_tmp = R;
 double *E_tmp = *_E;
 double *E_prev_tmp = *_E_prev;
 double mx, sumSq;
 int niter;

 int innerBlockRowStartIndex = (sub_n+2)+1;
 int innerBlockRowEndIndex = (((sub_m+2)*(sub_n+2) - 1) - (sub_n)) - (sub_n+2);


 // We continue to sweep over the mesh until the simulation has reached
 // the desired number of iterations
  for (niter = 0; niter < cb.niters; niter++){

      if  (cb.debug && (niter==0)){
	        stats(E_prev,sub_m,sub_n,&mx,&sumSq);
          double l2norm = L2Norm(sumSq);
	        repNorms(l2norm,mx,dt,sub_m,sub_n,-1, cb.stats_freq);
      	  // if (cb.plot_freq)
      	      // plotter->updatePlot(E,  -1, m+1, n+1);
      }



    copy_ghost_cells(E_prev);

//////////////////////////////////////////////////////////////////////////////

#define FUSED 1

#ifdef FUSED
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(sub_n+2)) {
        E_tmp = E + j;
	      E_prev_tmp = E_prev + j;
        R_tmp = R + j;
	      for(i = 0; i < sub_n; i++) {
	          E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(sub_n+2)]+E_prev_tmp[i-(sub_n+2)]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#else
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(sub_n+2)) {
        E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for(i = 0; i < sub_n; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(sub_n+2)]+E_prev_tmp[i-(sub_n+2)]);
            }
    }

    /*
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(sub_n+2)) {
        E_tmp = E + j;
        R_tmp = R + j;
	E_prev_tmp = E_prev + j;
        for(i = 0; i < sub_n; i++) {
	  E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
	  R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#endif
     /////////////////////////////////////////////////////////////////////////////////

   if (cb.stats_freq){
     if ( !(niter % cb.stats_freq)){
        stats(E,sub_m,sub_n,&mx,&sumSq);
        double l2norm = L2Norm(sumSq);
        repNorms(l2norm,mx,dt,sub_m,sub_n,niter, cb.stats_freq);
    }
   }

   if (cb.plot_freq){
          if (!(niter % cb.plot_freq)){
	             // plotter->updatePlot(E,  niter, m, n);
        }
    }

   // Swap current and previous meshes
   double *tmp = E; E = E_prev; E_prev = tmp;

 } //end of 'niter' loop at the beginning

  //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters

  stats(E_prev,sub_m,sub_n,&mx,&sumSq);

  double globalSq = 0.0;

  // reduce the L2 and Linf errors to process 0
  MPI_Reduce(&sumSq, &globalSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&mx, &Linf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  L2 = L2Norm(globalSq);

  // Swap pointers so we can re-use the arrays
  *_E = E;
  *_E_prev = E_prev;
}

void printMat2(const char mesg[], double *E, int m, int n){
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

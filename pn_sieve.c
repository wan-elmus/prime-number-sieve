
/* PROGRAM Sieve_parallel */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#define n 100000000
#define BLOCK_SIZE 100000
#define ROOT 0

int main(int argc, char *argv[]) {
  int i, j, k, num, first, loc, remainder, starting_point;
  int id, p, blocks_per_process, num_blocks, global_count = 0, local_count = 0;
  int start_block_id, end_block_id;
  double elapsed_time;
  char* prime;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Barrier(MPI_COMM_WORLD);
  elapsed_time = -MPI_Wtime();
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if (n % BLOCK_SIZE != 0) {
    if (id == ROOT) printf("Error: n is not divisible by BLOCK_SIZE\n");
    MPI_Finalize();
    exit(1);
  }

  blocks_per_process = n / BLOCK_SIZE / p;
  num_blocks = blocks_per_process * p;

  prime = (char*) malloc(sizeof(char) * (n + 1));
  if (prime == NULL) {
    printf("Error: Memory allocation failed\n");
    MPI_Finalize();
    exit(1);
  }

  // Initialize all elements to TRUE
  for (i = 0; i <= n; i++) {
    prime[i] = 1;
  }

  // Partition the array into equal size portions
  start_block_id = id * blocks_per_process;
  end_block_id = (id + 1) * blocks_per_process - 1;

  // The first process handles the first portion of the array
  if (id == ROOT) {
    for (num = 2; num <= sqrt(n); num++) {
      if (prime[num]) {
        first = num * num;
        if (first % 2 == 0) {
          first += num;
        }
        loc = first - 1;
        while (loc < n) {
          loc += num * 2;
          prime[loc] = 0;
        }
      }
    }
  }

  // Broadcast the step numbers through a pipeline
  for (k = 3; k <= sqrt(n); k += 2) {
    if (id == ROOT) {
      num = k;
      for (i = 1; i < p; i++) {
        MPI_Send(&num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      }
      first = BLOCK_SIZE;
    } else {
      MPI_Recv(&num, 1, MPI_INT, id - 1, 0, MPI_COMM_WORLD, &status);
      first = start_block_id * BLOCK_SIZE;
    }

    // Determine the starting point for stepping
    remainder = first % num;
    if (remainder == 0) {
      starting_point = first;
    } else {
      starting_point = (first / num + 1) * num - 1;
    }

    // Step through the portion of the array
    for (j = starting_point; j < (end_block_id + 1) * BLOCK_SIZE; j += num * 2) {
      prime[j] = 0;
    }

    // Count the number of primes in this portion
    for (j = start_block_id * BLOCK_SIZE; j < (end_block_id + 1) * BLOCK_SIZE; j++) {
        if (prime[j]) {
            local_count++;
            }
        }
    // Reduce the counts from all processes to the root process
MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);

// Reset the local count
local_count = 0;

}

MPI_Barrier(MPI_COMM_WORLD);
elapsed_time += MPI_Wtime();

if (id == ROOT) {
printf("Number of primes found = %d\n", global_count - 1); // Subtract 1 to exclude the number 1
printf("Total elapsed time: %10.6f\n", elapsed_time);
}

free(prime);
MPI_Finalize();
return 0;
}

// Compile and run using mpicc and mpirun respectively.

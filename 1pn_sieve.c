
/* PROGRAM Sieve_Parallel */
#include <stdio.h>
#include <math.h>
#include "mpi.h"

#define n 100

int main(int argc, char **argv)
{
  int size, rank, i, j, num, loc, first, remainder, step, prime_count = 0;
  double elapsed_time;
  char *marked;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Barrier(MPI_COMM_WORLD);
  elapsed_time = -MPI_Wtime();
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Compute the size of each portion of the array
  int chunk_size = n / size;
  int chunk_size_sqrt = sqrt(chunk_size);

  // Allocate memory for the marked array
  marked = (char *)calloc(chunk_size, sizeof(char));

  // Set all elements of the marked array to TRUE
  for (i = 0; i < chunk_size; i++)
  {
    marked[i] = 1;
  }

  // Find all primes in the first portion of the array
  if (rank == 0)
  {
    for (num = 2; num <= chunk_size_sqrt; num++)
    {
      if (marked[num])
      {
        // Calculate the starting point for this process
        first = num * num;
        remainder = first % num;
        if (remainder == 0)
        {
          loc = first;
        }
        else
        {
          loc = first - remainder + num;
        }

        // Eliminate all multiples of this prime in the first portion of the array
        while (loc < first + chunk_size)
        {
          marked[loc - first] = 0;
          loc += num;
        }
      }
    }

    // Broadcast each prime number to all processes and eliminate its multiples
    for (i = 1; i < size; i++)
    {
      // Find the next prime number
      j = 0;
      while (!marked[j])
      {
        j++;
      }
      num = j + 2;

      // Send the prime number to the next process
      MPI_Send(&num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

      // Calculate the starting point for this process
      first = num * ((rank * chunk_size + chunk_size_sqrt) / num) + chunk_size_sqrt;
      remainder = first % num;
      if (remainder == 0)
      {
        loc = first;
      }
      else
      {
        loc = first - remainder + num;
      }

      // Eliminate all multiples of this prime in the current portion of the array
      while (loc < rank * chunk_size + chunk_size)
      {
        marked[loc - rank * chunk_size] = 0;
        loc += num;
      }
    }
  }
  else
  {
    // Receive the first prime number from the previous process
    MPI_Recv(&num, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);

    // Calculate the starting point for this process
    first = num * ((rank * chunk_size + chunk_size_sqrt) / num) + chunk_size_sqrt;
    remainder = first % num;
    if (remainder == 0)
    {
      loc = first;
    }
    else
    {
      loc = first - remainder + num;
    }

    // Eliminate all multiples of this prime in the current portion of the array
    while (loc < rank * chunk_size + chunk_size)
    {
      marked[loc - rank * chunk_size] = 0;
      loc += num;
    }
    // Broadcast each prime number to all processes and eliminate its multiples
    while (1)
    {
      // Find the next prime number
      j = 0;
      while (!marked[j])
      {
        j++;
        if (j >= chunk_size)
        {
          break;
        }
      }
      if (j >= chunk_size)
      {
        break;
      }
      num = j + chunk_size * rank + chunk_size_sqrt + 1;

      // Send the prime number to the next process
      MPI_Send(&num, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);

      // Calculate the starting point for this process
      first = num * ((rank * chunk_size + chunk_size_sqrt) / num) + chunk_size_sqrt;
      remainder = first % num;
      if (remainder == 0)
      {
        loc = first;
      }
      else
      {
        loc = first - remainder + num;
      }

      // Eliminate all multiples of this prime in the current portion of the array
      while (loc < rank * chunk_size + chunk_size)
      {
        marked[loc - rank * chunk_size] = 0;
        loc += num;
      }
    }
  }

  // Count the number of primes in the array
  for (i = 0; i < chunk_size; i++)
  {
    if (marked[i])
    {
      prime_count++;
    }
  }

  // Reduce the number of primes from each process to get the total number of primes
  int total_prime_count;
  MPI_Reduce(&prime_count, &total_prime_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  elapsed_time += MPI_Wtime();
  if (rank == 0)
  {
    printf("Total number of primes: %d\n", total_prime_count);
    printf("Elapsed time: %lf seconds\n", elapsed_time);
  }

  // Free the memory allocated for the marked array
  free(marked);

  MPI_Finalize();
  return 0;
}

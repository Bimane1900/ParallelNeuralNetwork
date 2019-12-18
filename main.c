/*
    Neural Network Project by Eibech Barakat & Simon Roysson
*/

#include "functions.h"

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
	int rank, size;
    int workers = 4;
    int n = 20;
    float max = 0;
    float min = INT32_MAX;
    float one = -1.0f;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    float *inp = (float*)aligned_alloc(32, n*sizeof(float));
    if(rank == 0){
        printf("init\n");
        for (int i = 0; i < n; i++)
        {
            inp[i] = (float)i;
            MPI_Send(inp+i, 1, MPI_FLOAT, (i%workers)+1, i, MPI_COMM_WORLD);
            // if(inp[i] < min)
            //     min = inp[i];
            // if(inp[i] > max)
            //     max = inp[i];
            // printf("%f\n",inp[i]);
        }
        MPI_Send(&one, 1, MPI_FLOAT, 1, 100, MPI_COMM_WORLD);
        MPI_Send(&one, 1, MPI_FLOAT, 2, 100, MPI_COMM_WORLD);
        MPI_Send(&one, 1, MPI_FLOAT, 3, 100, MPI_COMM_WORLD);
        MPI_Send(&one, 1, MPI_FLOAT, 4, 100, MPI_COMM_WORLD);
    }
    else if(rank > 0 && rank < 5){
        float data;
        while(data >= 0){
            MPI_Recv(&data, 1, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            data *= 2;
            MPI_Send(&data, 1, MPI_FLOAT, 5, 1, MPI_COMM_WORLD);
            printf("rank : %d, recv data: %f\n", rank, data);
        }
    }
    else if (rank == 5){
        float data;
        int j = 0;
        int term = 0;
        while (term != workers)
        {
            MPI_Recv(&data, 1, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(data < 0)
                term++;
            else{
                inp[j] = data;
                j++;
            }

        }
        
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    //printf("min: %f\n", min);
    //printf("max: %f\n", max);
    // featureScaling(inp, max, min, n);
    // MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 5){
     printf("scaled\n");
    for (int i = 0; i < n; i++)
    {
        printf("%f\n",inp[i]);
    }
    }
    
    return 0;
}
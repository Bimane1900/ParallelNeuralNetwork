/*
    Neural Network Project by Eibech Barakat & Simon Roysson
*/

#include "functions.h"



int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
	int rank, size;
    int workers = 4;
    int n = ROWS*(COLUMNS-1);
    float *empty = NULL;
    float *data;// = (float*)aligned_alloc(32,ROWS*COLUMNS-1*sizeof(float));
    float *testData = (float*)aligned_alloc(32,ROWS*sizeof(float));
    float max = 0, min = 20000.0f;
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    if(rank == EMITTER){
        data = (float*)aligned_alloc(32,n*sizeof(float));
        readInputData((char*)"data.txt", data, testData);
        find_min_max(data, ROWS, COLUMNS-1, &max, &min);
        //printData(data, ROWS, COLUMNS-1);
        

        //send min/max
        for (int i = 2; i < size; i++)
        {
            MPI_Send(&min, 1, MPI_FLOAT, i, n, MPI_COMM_WORLD);
            MPI_Send(&max, 1, MPI_FLOAT, i, n, MPI_COMM_WORLD);
        }

        //send data
        int sendTo = 0;
        for (int i = 0; i < n; i+=AVXLOAD)
        {
            MPI_Send(data+i, AVXLOAD, MPI_FLOAT, (sendTo%workers)+2, i, MPI_COMM_WORLD);
            sendTo++;
            
        }

        //send termination
        for (int i = 2; i < size; i++)
        {
            MPI_Send(&empty, 1, MPI_FLOAT, i, n, MPI_COMM_WORLD);
        }
        
    }
    else if(rank > GATHERER){ //is worker
        float *data = (float*)aligned_alloc(32,sizeof(float)*AVXLOAD);
        
        MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));
        //recv min and max
        MPI_Recv(&min, 1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        MPI_Recv(&max, 1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        while(true){
            MPI_Recv(data, AVXLOAD, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
            if(status->MPI_TAG == n) // if tag == n then its termination time
                break;
            featureScaling(data, max, min, AVXLOAD);
            //forward scaled data to gatherer
            MPI_Send(data, AVXLOAD, MPI_FLOAT, GATHERER, status->MPI_TAG, MPI_COMM_WORLD);
        }
        //send termination to gatherer
        MPI_Send(&empty, 1, MPI_FLOAT, GATHERER, n, MPI_COMM_WORLD);
    }
    else if (rank == GATHERER){
        data = (float*)aligned_alloc(32,n*sizeof(float));
        float *incoming = (float*)aligned_alloc(32,sizeof(float)*AVXLOAD);
        int term = 0; //keep track of workers terminating
        MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));
        while (term != workers)
        {
            //recv from any worker
            MPI_Recv(incoming, AVXLOAD, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, status);
            
            if(status->MPI_TAG == n)
                term++;
            else{
                //store incoming data
                for (int i = 0; i < AVXLOAD; i++)
                {
                    data[i+status->MPI_TAG] = incoming[i];
                }
            }

        }
        
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == GATHERER){
     printf("scaled\n");
     printData(data, ROWS, COLUMNS-1);
    }

    MPI_Finalize();
    
    return 0;
}
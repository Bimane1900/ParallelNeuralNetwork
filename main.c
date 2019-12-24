/*
    Neural Network Project by Eibech Barakat & Simon Roysson
*/

#include "functions.h"


int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
	int rank, size;
    NeuralNetwork NN;
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    if(size != PROCESSES){
        printf("Processes needed: %d\n", PROCESSES);
        return 0;
    }
    int n = ROWS*(COLUMNS-1);
    if(rank == EMITTER){
        NN = initNN(n);
        //data = (float*)aligned_alloc(32,n*sizeof(float));
        setupFeatureScaling(NN.inputLayer, NN.testData, n);
        //awaitFeedforward();
        //setupBackProp();
    }
    else if (rank == GATHERER){
        NN = initNN(n);
        //data = (float*)aligned_alloc(32,n*sizeof(float));
        recieveFeatureScaledData(NN.inputLayer, n);
        //setupFeedforward();
        //recieveBackProp();
    }
    //Is worker
    else{
        handleFeatureScaling(n);
        //handleFeedforward();
        //handleBackProp();
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == GATHERER){
     printf("scaled\n");
     printData(NN.inputLayer, ROWS, COLUMNS-1);
     for (int i = 0; i < n; i++)
     {
         if(*(NN.inputLayer+i) == 1.0f)
            printf("\none: %f",*(NN.inputLayer+i));
     }
     
    }

    MPI_Finalize();
    
    return 0;
}
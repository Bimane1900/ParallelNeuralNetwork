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
    int hiddenlayers = 1;
    int inputsize = ROWS*(COLUMNS-1);
    int HL1Weights = HL1ROWS*HL1COLUMNS;
    int HL1Bias = NODESHL1;
    //int HL2Weights = HL2ROWS*HL2COLUMNS;
    //int HL2Bias = NODESHL2;
    int OLWeights = OLROWS*OLCOLUMNS;
    if(rank == EMITTER){
        NN = initNN(hiddenlayers,inputsize,HL1Weights,HL1Bias,OLWeights);
        //data = (float*)aligned_alloc(32,inputsize*sizeof(float));
        setupFeatureScaling(NN.inputLayer, NN.testData, inputsize);
        //setupFeedforward(NN);
        //setupBackProp();
    }
    else if (rank == GATHERER){
        NN = initNN(hiddenlayers,inputsize,HL1Weights,HL1Bias,OLWeights);
        //data = (float*)aligned_alloc(32,inputsize*sizeof(float));
        recieveFeatureScaledData(NN.inputLayer, inputsize);
        //recieveSetupFeedforward(&NN.hiddenLayers[0].id[0]);
        //setupFeedforward();
        //recieveBackProp();
    }
    //Is worker
    else{
        handleFeatureScaling(inputsize);
        //handleSetupFeedforward();
        //handleFeedforward();
        //handleBackProp();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == EMITTER){
        setupFeedforward();
    }
    else if (rank == GATHERER){
        recieveSetupFeedforward(&NN);
    }
    else{
        handleSetupFeedforward();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == GATHERER){
     printf("scaled\n");
     //printData(NN.hiddenLayers[0].w, HL1ROWS, HL1COLUMNS);
     printf("HIDDEN ID: %f", (NN.learningRate));
     for (int i = 0; i < inputsize; i++)
     {
         if((NN.inputLayer[i]) == 1.0f)
            printf("\none: %f",(NN.inputLayer[i]));
     }
     
    }

    MPI_Finalize();
    
    return 0;
}
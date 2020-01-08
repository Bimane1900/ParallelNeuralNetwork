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
        setupFeatureScaling(NN.inputLayer, NN.testData, inputsize);
    }
    else if (rank == GATHERER){
        NN = initNN(hiddenlayers,inputsize,HL1Weights,HL1Bias,OLWeights);
        recieveFeatureScaledData(NN.inputLayer, inputsize);
    }
    //Is worker
    else{
        handleFeatureScaling(inputsize);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == EMITTER){
        setupFeedforward();//read testdata is here
    }
    else if (rank == GATHERER){
        recieveSetupFeedforward(&NN);    
    }
    else{
        handleSetupFeedforward();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //int epoch = 1;
    //while(epoch){
        if(rank == EMITTER){
           reciveFeedforward(&NN);
        }
        else if (rank == GATHERER){
            feedforward(&NN);
        }
        else{
            handleFeedforward();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    //    --epoch;
    //     if(rank == EMITTER){
    //         //backprop
    //     }
    //     else if (rank == GATHERER){
    //         //recive backrop, assign Weights
    //     }
    //     else{
    //         //handle backprop
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    //}
    if(rank == GATHERER){
     //printf("GATHERER\n");
     //printData(NN.inputLayer, ROWS, HL1ROWS);
     //printData(NN.hiddenLayers[0].w, HL1ROWS, HL1COLUMNS);
     //printf("HIDDEN ID: %f", (NN.learningRate));
    //  for (int i = 0; i < inputsize; i++)
    //  {
    //      if((NN.inputLayer[i]) == 1.0f)
    //         printf("\none: %f",(NN.inputLayer[i]));
    //  }
    }
     if(rank == EMITTER){
        printf("EMITTER\n");
        printData(NN.outputLayer[0].output, 4, 1);
        // for (int i = 0; i < 4; i++)
        // {    
        //     printf("\none: %f",(NN.testData[i]));
        // } 
    }
    MPI_Finalize();
    
    return 0;
}
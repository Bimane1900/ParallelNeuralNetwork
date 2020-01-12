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
    int OLWeights = OLROWS*OLCOLUMNS;
    double totalTime = MPI_Wtime();
    if(rank == EMITTER){
        NN = initNN(hiddenlayers,inputsize,HL1Weights,HL1Bias,OLWeights);
        readInputData((char*)FILENAME, NN.inputLayer, NN.testData);
        //initDummies(NN.inputLayer, NN.testData);
        setupFeatureScaling(NN.inputLayer, inputsize);
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
        startInitiateWeights();//read testdata is here
    }
    else if (rank == GATHERER){
        recieveInitialWeights(&NN);
        printf("Dimensions: %dx%d\n", ROWS, COLUMNS);
        printf("Beginning training...\n");
    }
    else{
        initiateWeights();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    totalTime = MPI_Wtime();
    int epoch = EPOCHS;
    while(epoch){
        if (rank == GATHERER){
            startFeedforward(&NN);
            recieveBackProp(&NN);
        }
        else if(rank == EMITTER){
            recieveFeedforwardOutputs(&NN);
            setupBackProp(&NN);
        }
        else{
            handleFeedforward();
            handleBackProp();
        }
        --epoch;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if(rank == GATHERER){
        //printData(NN.inputLayer, ROWS, HL1ROWS);
        //printData(NN.hiddenLayers[0].w, HL1ROWS, HL1COLUMNS);
        //printData(NN.outputLayer[0].w, NODESHL1, OLCOLUMNS);
        //printData(NN.hiddenLayers[0].output, ROWS, HL1COLUMNS);
        freeNN(NN);
    }
     if(rank == EMITTER){
        totalTime = MPI_Wtime() - totalTime;
        printf("Training complete, took %f\n", totalTime);
        accuracyTest(&NN);
        freeNN(NN); 
    }
    
    MPI_Finalize();
    
    return 0;
}
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
    double totalTime = MPI_Wtime();
    double localTime;
    localTime = MPI_Wtime();
    if(rank == EMITTER){
        NN = initNN(hiddenlayers,inputsize,HL1Weights,HL1Bias,OLWeights);
        readInputData((char*)FILENAME, NN.inputLayer, NN.testData);
        printf("setting up featurescaling...\n");
        setupFeatureScaling(NN.inputLayer, inputsize);
    }
    else if (rank == GATHERER){
        NN = initNN(hiddenlayers,inputsize,HL1Weights,HL1Bias,OLWeights);
        recieveFeatureScaledData(NN.inputLayer, inputsize);
        localTime = MPI_Wtime() -localTime;
        printf("featurescaling complete, took: %f\n",localTime );
    }
    //Is worker
    else{
        handleFeatureScaling(inputsize);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    localTime = MPI_Wtime();
    if(rank == EMITTER){
        printf("Initiating weights...\n");
        setupFeedforward();//read testdata is here
    }
    else if (rank == GATHERER){
        recieveSetupFeedforward(&NN);  
        localTime = MPI_Wtime() - localTime;
        printf("Weights initiaded, took: %f\n", localTime);  
        printf("Beginning training...\n");
    }
    else{
        handleSetupFeedforward();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    localTime = MPI_Wtime();
    totalTime = MPI_Wtime();
    //int epoch = 1;
    //while(epoch){
        if(rank == EMITTER){
            localTime = MPI_Wtime() - localTime;
            printf("Feedforward complete, took %f\n",localTime);
            reciveFeedforward(&NN);
            //printData(NN.hiddenLayers[0].output, ROWS, HL1COLUMNS);
            //printData(NN.outputLayer[0].output, ROWS, OLCOLUMNS);
        }
        else if (rank == GATHERER){
            feedforward(&NN);
            printf("Beginning feedforward...\n");
        }
        else{
            handleFeedforward();
        }
        MPI_Barrier(MPI_COMM_WORLD);
        localTime = MPI_Wtime();
    //    --epoch;
        if(rank == EMITTER){
            printf("Beginning backpropagation...\n");
            setupBackProp(&NN);
        }
        else if (rank == GATHERER){
            recieveBackProp(&NN);
            localTime = MPI_Wtime() - localTime;
            printf("Backpropagation complete, took %f\n", localTime);
            //recive backrop, assign Weights
        }
        else{
            handleBackProp();
        }
        MPI_Barrier(MPI_COMM_WORLD);
        totalTime = MPI_Wtime() - totalTime;
    //}
    if(rank == GATHERER){
        printf("Training complete, took %f\n", totalTime);
     //printf("GATHERER\n");
        //printData(NN.inputLayer, ROWS, HL1ROWS);
        //printData(NN.hiddenLayers[0].w, HL1ROWS, HL1COLUMNS);
        //printData(NN.outputLayer[0].w, NODESHL1, OLCOLUMNS);
        //printf("%f",NN.learningRate);  
        //printData(NN.hiddenLayers[0].output, ROWS, HL1COLUMNS);
        
     //printf("HIDDEN ID: %f", (NN.learningRate));
    //  for (int i = 0; i < inputsize; i++)
    //  {
    //      if((NN.inputLayer[i]) == 1.0f)
    //         printf("\none: %f",(NN.inputLayer[i]));
    //  }
        freeNN(NN);
    }
     if(rank == EMITTER){
        printf("EMITTER\n");
        //printData(NN.hiddenLayers[0].output, ROWS, HL1COLUMNS);
        // for (int i = 0; i < 4; i++)
        // {    
        //     printf("\none: %f",(NN.testData[i]));
        // }
        freeNN(NN); 
    }
    
    MPI_Finalize();
    
    return 0;
}
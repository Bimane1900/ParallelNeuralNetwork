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
    const char* measures[21];
    measures[0] = "ROWS, COLUMNS, TIME\n";
    int iterations = 1;
    if(size != PROCESSES){
        printf("Processes needed: %d\n", PROCESSES);
        return 0;
    }
    while(nOfRows > 128){
        //nOfRows = 1;
    int hiddenlayers = 1;
    int inputsize = ROWS*(COLUMNS-1);
    int HL1Weights = HL1ROWS*HL1COLUMNS;
    int HL1Bias = NODESHL1;
    int OLWeights = OLROWS*OLCOLUMNS;
    double totalTime = MPI_Wtime();
    if(rank == EMITTER){
        NN = initNN(hiddenlayers,inputsize,HL1Weights,HL1Bias,OLWeights);
        printf("Preparing data..\n");
        //readInputData((char*)FILENAME, NN.inputLayer, NN.testData);
        initDummies(NN.inputLayer, NN.testData);
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
    double localTime = MPI_Wtime();
    int epoch = EPOCHS;
    while(epoch){
        if (rank == GATHERER){
            localTime = MPI_Wtime();
            startFeedforward(&NN);
            localTime = MPI_Wtime()-localTime;
            printTime("-----------main, startfeedforward %f\n", localTime);
            localTime = MPI_Wtime();
            recieveBackProp(&NN);
            localTime = MPI_Wtime()-localTime;
            printTime("-----------main, recievBackProp %f\n", localTime);
        }
        else if(rank == EMITTER){
            localTime = MPI_Wtime();
            recieveFeedforwardOutputs(&NN);
            localTime = MPI_Wtime()-localTime;
            printTime("-----------main, reciveFeedforward %f\n", localTime);
            localTime = MPI_Wtime();
            setupBackProp(&NN);
            localTime = MPI_Wtime()-localTime;
            printTime("-----------main, setupbackprop %f\n", localTime);
        }
        else{
            localTime = MPI_Wtime();
            handleFeedforward();
            localTime = MPI_Wtime()-localTime;
            printTime("-----------main, handleFeedforwad %f\n", localTime);
            localTime = MPI_Wtime();
            handleBackProp();
            localTime = MPI_Wtime()-localTime;
            printTime("-----------main, handleBackprop %f\n", localTime);
        }
        --epoch;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if(rank == GATHERER){
        freeNN(NN);
    }
     if(rank == EMITTER){
        totalTime = MPI_Wtime() - totalTime;
        printf("Training complete, took %f\n", totalTime);
        measures[iterations] = saveMeasuredTime(totalTime, nOfRows, nOfColumns);
        iterations++;
        accuracyTest(&NN);
        freeNN(NN); 
    }
    nOfRows /= 2;
    nOfColumns *= 2;
    }
    if(rank == EMITTER){
        for (int i = 0; i < iterations; i++)
        {
            printf("%s",measures[i]);
        }
        
    }
    MPI_Finalize();
    
    return 0;
}
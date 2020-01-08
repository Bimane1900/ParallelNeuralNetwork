/*
    Functions for the Neural Network
*/

#include "math.h"
#include <immintrin.h>
#include <mpi.h>
#include "stdlib.h"
#include "structs.h"
#include "stdio.h"
#include "string.h"
//#include "FF.c"

void printData(float* data, int rows, int columns);
float sigmoid(float x, int derivate);
void setupFeatureScaling(float *data, float* testData, int n);
void setupFeedforward();
void handleSetupFeedforward();
void recieveSetupFeedforward(NeuralNetwork* nn);
void initweights(float* weights);
void featureScale(float *inputData, float max, float min, int n);
void handleFeatureScaling(int n);
void recieveFeatureScaledData(float* data, int n);
void feedforward(NeuralNetwork* nn);
void handleFeedforward();
void calcOutput(float* inputSlice, float* weights, float* bias, float* output);
void reciveFeedforward(NeuralNetwork* nn);

void readInputData(char* textfile, float* data, float* correctRes);
float _mm256_find_max(__m256 vector);
float _mm256_find_min(__m256 vector);
void find_min_max(float* data, int rows, int columns, float* max, float* min);
NeuralNetwork initNN(int HLayers, int InputSize, int HL1W, int HL1B, int OLW);

void printData(float* data, int rows, int columns){
    for (int i = 0; i < rows; i++)
    {
        printf("[");
        for (int j = 0; j < columns; j++)
        {
            printf("%f, ", data[i * columns + j]);
        }
        printf("]\n");   
    }
    
}

float sigmoid(float x, int derivate){
    if(derivate)
        return x*(1-x);
    return 1/(1+expf(-x));
}

//Reads data, finds min/max and forwards data to workers
void setupFeatureScaling(float *data, float* testData, int n){
    float max = 0.0f, min = 20000.0f;
    readInputData((char*)"testdata.txt", data, testData);
    find_min_max(data, ROWS, COLUMNS-1, &max, &min);
    float *empty = NULL;
    //send min/max to workers
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(&min, 1, MPI_FLOAT, i, n, MPI_COMM_WORLD);
        MPI_Send(&max, 1, MPI_FLOAT, i, n, MPI_COMM_WORLD);
    }

    //send data
    int sendTo = 0;
    for (int i = 0; i < n; i+=AVXLOAD)
    {
        MPI_Send(data+i, AVXLOAD, MPI_FLOAT, (sendTo%WORKERS)+2, i, MPI_COMM_WORLD);
        sendTo++;
        
    }

    //send termination
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(&empty, 1, MPI_FLOAT, i, n, MPI_COMM_WORLD);
    }
}

//Recieves data, featureScales it and forwards to a reciever
void handleFeatureScaling(int n){
    float *data = (float*)aligned_alloc(32,sizeof(float)*AVXLOAD);
    float min,max;
    float* empty = NULL;
    float sendSize = AVXLOAD;
        
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));
    //recv min and max
    MPI_Recv(&min, 1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(&max, 1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    while(true){
        MPI_Recv(data, AVXLOAD, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        if(status->MPI_TAG == n) // if tag == n then its termination time
            break;
        if(status->MPI_TAG+AVXLOAD > n){
            sendSize = n-status->MPI_TAG;//(status->MPI_TAG+AVXLOAD)-n;
            featureScale(data, max, min, sendSize);
        }
        else{
            featureScale(data, max, min, AVXLOAD);
            sendSize = AVXLOAD;
        }//forward scaled data to gatherer
        MPI_Send(data, sendSize, MPI_FLOAT, GATHERER, status->MPI_TAG, MPI_COMM_WORLD);
    }
    //send termination to gatherer
    MPI_Send(&empty, 1, MPI_FLOAT, GATHERER, n, MPI_COMM_WORLD);
}

//Normalizes data
void featureScale(float *inputData, float max, float min, int n){
    // __m256 vmax = _mm256_set1_ps(max);
    // __m256 vmin = _mm256_set1_ps(min);
    // __m256 diff = _mm256_sub_ps(vmax, vmin);
    // __m256 x;
    // for (int i = 0; i < n; i+=AVXLOAD)
    // {
    //     x = _mm256_load_ps(inputData+i);
    //     x = _mm256_div_ps(_mm256_sub_ps(x, vmin), diff);
    //     _mm256_store_ps(inputData+i, x);
    // }
    float maskArr[AVXLOAD] __attribute__ ((aligned (32))) = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
    for (int i = n; i < AVXLOAD; i++)
    {
        maskArr[i] = 1.0f;
    }
    
    __m256i mask = _mm256_load_si256((__m256i*)maskArr); 
    __m256 vmax = _mm256_set1_ps(max);
    __m256 vmin = _mm256_set1_ps(min);
    __m256 diff = _mm256_sub_ps(vmax, vmin);
    //__m256 x = _mm256_load_ps(inputData);
    __m256 x = _mm256_maskload_ps(inputData, mask);
    x = _mm256_sub_ps(x, vmin);
    x = _mm256_div_ps(x, diff);
    _mm256_maskstore_ps(inputData, mask, x);
    //_mm256_store_ps(inputData, x);

}

//Wait for featureScaled data and read it
void recieveFeatureScaledData(float* data, int n){
    float *incoming = (float*)aligned_alloc(32,sizeof(float)*AVXLOAD);
    int term = 0; //keep track of workers terminating
    int count = 0;
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));
    while (term != WORKERS)
    {
        //recv from any worker
        MPI_Recv(incoming, AVXLOAD, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        MPI_Get_count(status, MPI_FLOAT, &count);
        if(status->MPI_TAG == n)
            term++;
        else{
            //store incoming data
            for (int i = 0; i < count; i++)
            {
                data[i+status->MPI_TAG] = incoming[i];
            }
        }

    }
    free(incoming);
}

//Reads data from file and stores in float arrays
void readInputData(char* textfile, float* data, float* correctRes){
    FILE *fp;
    char *str = (char*)malloc(200);
    char *temp = NULL;
 
    fp = fopen(textfile, "r");
    if (fp == NULL){
        printf("Could not open file %s",textfile);
    }
    else{
        int lastChar = 0;
        int i=0, j=0;
        while (fgets(str, 200, fp) != NULL){
            lastChar = strlen(str)-5;//-3 in real data -5 test
            temp = str;
            correctRes[j] = float(strtod(str+lastChar, NULL));
            j++;
            temp[lastChar] = '\0';
            while(*temp != '\0'){
                data[i] = (float)strtod(temp, &temp);
                i++;
                temp++;
            }
        }
        fclose(fp);
        
    }
    free(str);
    str = NULL;
    fp = NULL;
    temp = NULL;
}

//Finds max float in a AVX vector
float _mm256_find_max(__m256 vector){
	__m256 tmp = _mm256_permute2f128_ps(vector , vector , 0b10000001 );
	vector = _mm256_max_ps(vector , tmp);
	tmp = _mm256_shuffle_ps(vector , vector , 0b00001110 );
	vector = _mm256_max_ps(vector , tmp);
	tmp = _mm256_shuffle_ps(vector , vector , 0b00000001 );
	vector = _mm256_max_ps(vector , tmp);
	return _mm256_cvtss_f32(vector);
}

//Find min float in a AVX vector
float _mm256_find_min(__m256 vector){
	__m256 tmp = _mm256_permute2f128_ps(vector , vector , 0b10000001 );
	vector = _mm256_min_ps(vector , tmp);
	tmp = _mm256_shuffle_ps(vector , vector , 0b00001110 );
	vector = _mm256_min_ps(vector , tmp);
	tmp = _mm256_shuffle_ps(vector , vector , 0b00000001 );
	vector = _mm256_min_ps(vector , tmp);
	return _mm256_cvtss_f32(vector);
}

//Finds min and max in float* data
void find_min_max(float* data, int rows, int columns, float* max, float* min){
	__m256 min_vector = _mm256_set1_ps(*min);
	__m256 max_vector = _mm256_set1_ps(*max);
	//forloop sets vector with 8 minimum/maximum floats
	for (int i = AVXLOAD; i < rows*columns; i+=AVXLOAD)
	{
        max_vector = _mm256_max_ps(max_vector, _mm256_load_ps(data+i));
		min_vector = _mm256_min_ps(min_vector, _mm256_load_ps(data+i));
	}
	//extract min/max with declared functions
    *max = _mm256_find_max(max_vector);
    *min = _mm256_find_min(min_vector);
}

void initweights(float* weights,int size)
{
    __m256 one_vec = _mm256_set1_ps(1.0f);
    __m256 twos_vec = _mm256_set1_ps(2.0f);
    __m256 RANDMAX_vec = _mm256_set1_ps(RAND_MAX);
    __m256 rand_vector;
    __m256i mask;
    for (int i = 0; i < size; i++)
    {
        weights[i] = (float)rand();
    }
    for (int i = 0; i < size; i+=AVXLOAD)
	{
        if(i+AVXLOAD >= size){
            float maskArr[AVXLOAD] __attribute__ ((aligned (32))) = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
            for (int j = size-i; j < AVXLOAD; j++)
            {
                maskArr[j] = 1.0f;
            }
            
            mask = _mm256_load_si256((__m256i*)maskArr); 
            rand_vector = _mm256_maskload_ps(weights+i, mask);
            rand_vector = _mm256_div_ps(rand_vector,RANDMAX_vec);
            rand_vector = _mm256_mul_ps(rand_vector,twos_vec);
            rand_vector = _mm256_sub_ps(rand_vector,one_vec);
            //_mm256_store_ps(weights+i,rand_vector);
            _mm256_maskstore_ps(weights+i, mask, rand_vector);
            
        }
        else{
            rand_vector = _mm256_div_ps(_mm256_load_ps(weights+i),RANDMAX_vec);
            rand_vector = _mm256_mul_ps(rand_vector,twos_vec);
            rand_vector = _mm256_sub_ps(rand_vector,one_vec);
            _mm256_store_ps(weights+i,rand_vector);
        }
	}
}

void setupFeedforward()
{
    int WeightSize = ((HL1ROWS * HL1COLUMNS) + HL1COLUMNS)/WORKERS;
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(&WeightSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }     
}

void handleSetupFeedforward()
{
    int incoming;
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));
    MPI_Recv(&incoming, 1, MPI_INT, EMITTER, 0, MPI_COMM_WORLD, status);
    float* weights = (float*)aligned_alloc(32, (incoming/*+(incoming%AVXLOAD)*/)*sizeof(float));
    initweights(weights,incoming);
    MPI_Send(weights, incoming, MPI_FLOAT, GATHERER, 1 , MPI_COMM_WORLD);
    free(weights);
}

void recieveSetupFeedforward(NeuralNetwork* nn)
{
    int index = 0;
    int WeightSize = ((HL1ROWS * HL1COLUMNS) + HL1COLUMNS)/WORKERS;
    int WeightSize2;
    int WeightHL1 = (HL1ROWS * HL1COLUMNS);
    float* weights = (float*)aligned_alloc(32, (WeightSize)*sizeof(float));
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));
    for (int i = 2; i < PROCESSES; i++)
    {
    MPI_Recv(weights, WeightSize, MPI_FLOAT, i, 1, MPI_COMM_WORLD, status);
        for (int j = 0; j < WeightSize; j++)
        {
            if (index < WeightHL1)
            {
                nn->hiddenLayers[0].w[index] = weights[j];
            }
            else
            {
                nn->outputLayer[0].w[index-WeightHL1] = weights[j];
            }
            index++;
        }
    }
    for (int i = 0; i < NODESHL1; i++)
    {
        nn->hiddenLayers[0].bias[i] = 0;
    }
    nn->outputLayer[0].bias[0] = 0;
    free(weights);
}

void feedforward(NeuralNetwork* nn){
    float* empty = NULL;
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(nn->hiddenLayers[0].w, (HL1ROWS*HL1COLUMNS), MPI_FLOAT, i, i, MPI_COMM_WORLD);
        MPI_Send(nn->hiddenLayers[0].bias, NODESHL1, MPI_FLOAT, i, i, MPI_COMM_WORLD);
    }   
    for (int i = 0; i < ROWS; i++)
    {
        MPI_Send(nn->inputLayer+(i * HL1ROWS), HL1ROWS, MPI_FLOAT, (i%WORKERS)+2, i, MPI_COMM_WORLD);
    }   
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(&empty, 1, MPI_FLOAT, i, ROWS, MPI_COMM_WORLD);
    }
}

void handleFeedforward(){
    float* inputSlice = (float*)aligned_alloc(32, HL1ROWS*sizeof(float));
    float* weights = (float*)aligned_alloc(32, (HL1ROWS*HL1COLUMNS)*sizeof(float));
    float* bias = (float*)aligned_alloc(32, NODESHL1*sizeof(float));
    float* output = (float*)aligned_alloc(32, NODESHL1*sizeof(float));
    float* empty = NULL;
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));

    MPI_Recv(weights, (HL1ROWS*HL1COLUMNS), MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(bias, NODESHL1, MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);

 
    while(true){    
        MPI_Recv(inputSlice, HL1ROWS, MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        for (int i = 0; i < NODESHL1; i++)
        {
            output[i] = 0.0;
        }
        if (status->MPI_TAG == ROWS)
            break;
        
        calcOutput(inputSlice,weights,bias,output);
        MPI_Send(output, NODESHL1, MPI_FLOAT, EMITTER, status->MPI_TAG, MPI_COMM_WORLD);
        output[0] = 0.0;
        output[1] = 0.0;
    }
    MPI_Send(&empty, 1, MPI_FLOAT, EMITTER, ROWS, MPI_COMM_WORLD);
}

void reciveFeedforward(NeuralNetwork* nn){
    float* output = (float*)aligned_alloc(32, NODESHL1*sizeof(float));
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));
    int term = 0; 
    while(term != WORKERS){    
        MPI_Recv(output, NODESHL1, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        if(status->MPI_TAG == ROWS)
            term++;
        else{
            for (int i = 0; i < NODESHL1; i++)
            {
                nn->hiddenLayers[0].output[(status->MPI_TAG*NODESHL1)+i] = output[i];
            }
        }
    }
}

void calcOutput(float* inputSlice, float* weights,float* bias,float* output){
    for (int j = 0; j < NODESHL1; j++)
    {
        for (int i = 0; i < HL1ROWS; i++)
        {
            output[j] += weights[(i*NODESHL1)+j]*inputSlice[i];
        }
        output[j] += bias[j];
        output[j] = sigmoid(output[j], 0);
    }
}

NeuralNetwork initNN(int HLayers, int InputSize, int HL1W, int HL1B, int OLW){
    NeuralNetwork nn;
    nn.inputLayer = (float*)aligned_alloc(32, InputSize*sizeof(float));
    nn.testData = (float*)aligned_alloc(32, ROWS*sizeof(float));
    
    nn.hiddenLayers = (Layer*)aligned_alloc(32, HLayers*sizeof(Layer));
    nn.hiddenLayers[0].id = (int*)aligned_alloc(32,sizeof(int));
    nn.hiddenLayers[0].w = (float*)aligned_alloc(32, HL1W*sizeof(float));
    nn.hiddenLayers[0].output = (float*)aligned_alloc(32, (ROWS*HL1COLUMNS)*sizeof(float));
    nn.hiddenLayers[0].bias = (float*)aligned_alloc(32, HL1B*sizeof(float));

    nn.outputLayer = (Layer*)aligned_alloc(32, sizeof(Layer));
    nn.outputLayer[0].id = (int*)aligned_alloc(32,sizeof(int));
    nn.outputLayer[0].w = (float*)aligned_alloc(32,OLW*sizeof(float));
    nn.outputLayer[0].output = (float*)aligned_alloc(32,(ROWS*OLCOLUMNS)*sizeof(float));
    nn.outputLayer[0].bias = (float*)aligned_alloc(32,sizeof(float));

    return nn;
}
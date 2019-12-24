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

void printData(float* data, int rows, int columns);
float sigmoid(float x, int derivate);
void setupFeatureScaling(float *data, float* testData, int n);
void featureScale(float *inputData, float max, float min, int n);
void handleFeatureScaling(int n);
void recieveFeatureScaledData(float* data, int n);
void readInputData(char* textfile, float* data, float* correctRes);
float _mm256_find_max(__m256 vector);
float _mm256_find_min(__m256 vector);
void find_min_max(float* data, int rows, int columns, float* max, float* min);
NeuralNetwork initNN(int n);

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
    readInputData((char*)"data.txt", data, testData);
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
        
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));
    //recv min and max
    MPI_Recv(&min, 1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(&max, 1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    while(true){
        MPI_Recv(data, AVXLOAD, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        if(status->MPI_TAG == n) // if tag == n then its termination time
            break;
        featureScale(data, max, min, AVXLOAD);
        //forward scaled data to gatherer
        MPI_Send(data, AVXLOAD, MPI_FLOAT, GATHERER, status->MPI_TAG, MPI_COMM_WORLD);
    }
    //send termination to gatherer
    MPI_Send(&empty, 1, MPI_FLOAT, GATHERER, n, MPI_COMM_WORLD);
}

//Normalizes data
void featureScale(float *inputData, float max, float min, int n){
    __m256 vmax = _mm256_set1_ps(max);
    __m256 vmin = _mm256_set1_ps(min);
    __m256 diff = _mm256_sub_ps(vmax, vmin);
    __m256 x;
    for (int i = 0; i < n; i+=AVXLOAD)
    {
        x = _mm256_load_ps(inputData+i);
        x = _mm256_div_ps(_mm256_sub_ps(x, vmin), diff);
        _mm256_store_ps(inputData+i, x);
    }
}

//Wait for featureScaled data and read it
void recieveFeatureScaledData(float* data, int n){
    float *incoming = (float*)aligned_alloc(32,sizeof(float)*AVXLOAD);
    int term = 0; //keep track of workers terminating
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));
    while (term != WORKERS)
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
            lastChar = strlen(str)-3;
            temp = str;
            correctRes[j] = float(strtod(&str[lastChar], NULL));
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



NeuralNetwork initNN(int n){
    NeuralNetwork nn;
    nn.inputLayer = (float*)aligned_alloc(32, n*sizeof(float));
    nn.testData = (float*)aligned_alloc(32, ROWS*sizeof(float));
    return nn;
}
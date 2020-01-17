/*
    Functions for the Neural Network
*/

#include "math.h"
#include <immintrin.h>
#include <mpi.h>
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
//#include "FF.c"
#include <sys/time.h>
#include "structs.h"

void printData(float* data, int rows, int columns);
void readInputData(char* textfile, float* data, float* correctRes);
NeuralNetwork initNN(int HLayers, int InputSize, int HL1W, int HL1B, int OLW);
void freeNN(NeuralNetwork NN);
float _mm256_find_max(__m256 vector);
float _mm256_find_min(__m256 vector);
void find_min_max(float* data, int rows, int columns, float* max, float* min);
void accuracyTest(NeuralNetwork* nn);
__m256i getAVXVectorMask(int start);
void printTime(const char* str, double time);
float sigmoid(float x, int derivate);
const char* saveMeasuredTime(double totalTime, int nOfRows, int nOfColumns);


#include "featurescaling.h"
#include "feedforward.h"
#include "backpropa.h"

const char* saveMeasuredTime(double totalTime, int nOfRows, int nOfColumns){
    const char* s =(const char*) malloc(30);
    sprintf((char*)s, "%d,%d,%f\n", nOfRows, nOfColumns, totalTime);
    return s;
}

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

//function used to print times, easy to disable by commenting the printf
void printTime(const char* str, double time){
    // if(EPOCHS == 1)
    //     printf(str, time);
}

//neural network activation function
float sigmoid(float x, int derivate){
    if(derivate)
        return x*(1-x);
    return 1/(1+expf(-x));
}

void accuracyTest(NeuralNetwork* nn){
    // printData(nn->outputLayer[0].output, ROWS,1);
    // printf("testdata\n");
    // printData(nn->testData, ROWS, 1);
    int guesses = 0;
    for (int i = 0; i < ROWS; i++)
    {
        if(nn->outputLayer[0].output[i] > 0.5f && nn->testData[i] == 1.0f)
            guesses += 1;
        if(nn->outputLayer[0].output[i] < 0.5f && nn->testData[i] == 0.0f)
            guesses += 1;
    }
    printf("test Accuracy: %f\n",(float)guesses/ROWS);
    
}

void initDummies(float* data, float* correctRes){
    for (int i = 0; i < ROWS*(COLUMNS-1); i++)
    {
        data[i] = 1.1f;
    }
    
    for (int i = 0; i < ROWS; i++)
    {
        correctRes[i] = 1.1f;
    }
    
}

//Reads data from file and stores in float arrays
void readInputData(char* textfile, float* data, float* correctRes){
    FILE *fp;
    char *str = (char*)malloc(2000);
    char *temp = NULL;
 
    fp = fopen(textfile, "r");
    if (fp == NULL){
        printf("Could not open file %s",textfile);
    }
    else{
        int lastCharInd = 0;
        int i=0, j=0;
        while (fgets(str, 2000, fp) != NULL){
            lastCharInd = strlen(str)-LAST_CHAR_OFFSET;//-3 in real data -5 test
            temp = str;
            //correctRes should have last digit per row, which is testing data
            correctRes[j] = float(strtod(str+lastCharInd, NULL));
            j++;
            //moves end of line to not read test data into actual data
            temp[lastCharInd] = '\0';
            //reads every digit on a line 
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
    __m256i mask;
	//forloop sets vector with 8 minimum/maximum floats
	for (int i = AVXLOAD; i < rows*columns; i+=AVXLOAD)
	{
        mask = getAVXVectorMask(rows*columns-i);
        max_vector = _mm256_max_ps(max_vector, _mm256_maskload_ps(data+i,mask));
		min_vector = _mm256_min_ps(min_vector, _mm256_maskload_ps(data+i,mask));
	}
	//extract min/max with declared functions
    *max = _mm256_find_max(max_vector);
    *min = _mm256_find_min(min_vector);
}

//gets a mask for use in maskload or maskstore
//indices between start -> AVXLOAD(8) will be masked
__m256i getAVXVectorMask(int start){
    __m256i mask;
    float maskArr[AVXLOAD] __attribute__ ((aligned (32))) = {
        -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
    for (int i = start; i < AVXLOAD; i++)
    {
        maskArr[i] = 1.0f;
    }
    mask = _mm256_load_si256((__m256i*)maskArr); 
    return mask;
} 

NeuralNetwork initNN(int HLayers, int InputSize, int HL1W, int HL1B, int OLW){
    NeuralNetwork nn;
    nn.inputLayer = (float*)aligned_alloc(32, InputSize*sizeof(float));
    nn.testData = (float*)aligned_alloc(32, ROWS*sizeof(float));
    
    nn.hiddenLayers = (Layer*)aligned_alloc(32, HLayers*sizeof(Layer));
    nn.hiddenLayers[0].w = (float*)aligned_alloc(32, HL1W*sizeof(float));
    nn.hiddenLayers[0].output = (float*)aligned_alloc(32, (ROWS*HL1COLUMNS)*sizeof(float));
    nn.hiddenLayers[0].bias = (float*)aligned_alloc(32, HL1B*sizeof(float));

    nn.outputLayer = (Layer*)aligned_alloc(32, sizeof(Layer));
    nn.outputLayer[0].w = (float*)aligned_alloc(32,OLW*sizeof(float));
    nn.outputLayer[0].output = (float*)aligned_alloc(32,(ROWS*OLCOLUMNS)*sizeof(float));
    nn.outputLayer[0].bias = (float*)aligned_alloc(32,sizeof(float));

    return nn;
}

void freeNN(NeuralNetwork nn){

    free(nn.inputLayer);
    free(nn.testData);
    
    free(nn.hiddenLayers[0].w);
    free(nn.hiddenLayers[0].output);
    free(nn.hiddenLayers[0].bias);
    free(nn.hiddenLayers);
    

    free(nn.outputLayer[0].w);
    free(nn.outputLayer[0].output);
    free(nn.outputLayer[0].bias);
    free(nn.outputLayer);
}
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

void calcLayer(Layer *layer){

}

float sigmoid(float x, int derivate){
    if(derivate)
        return x*(1-x);
    return 1/(1+expf(-x));
}

//normalizes inputs
void featureScaling(float *inputData, float max, float min, int n){
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

float _mm256_find_max(__m256 vector){
	__m256 tmp = _mm256_permute2f128_ps(vector , vector , 0b10000001 );
	vector = _mm256_max_ps(vector , tmp);
	tmp = _mm256_shuffle_ps(vector , vector , 0b00001110 );
	vector = _mm256_max_ps(vector , tmp);
	tmp = _mm256_shuffle_ps(vector , vector , 0b00000001 );
	vector = _mm256_max_ps(vector , tmp);
	return _mm256_cvtss_f32(vector);
}

float _mm256_find_min(__m256 vector){
	__m256 tmp = _mm256_permute2f128_ps(vector , vector , 0b10000001 );
	vector = _mm256_min_ps(vector , tmp);
	tmp = _mm256_shuffle_ps(vector , vector , 0b00001110 );
	vector = _mm256_min_ps(vector , tmp);
	tmp = _mm256_shuffle_ps(vector , vector , 0b00000001 );
	vector = _mm256_min_ps(vector , tmp);
	return _mm256_cvtss_f32(vector);
}

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

NeuralNetwork initNN(){
    NeuralNetwork n;
}
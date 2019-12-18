/*
    Functions for the Neural Network
*/

#include "math.h"
#include <immintrin.h>
#include <mpi.h>
#include "stdlib.h"
#include "structs.h"

void printOutput(){

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
    int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    __m256 vmax = _mm256_set1_ps(max);
    __m256 vmin = _mm256_set1_ps(min);
    __m256 diff = _mm256_sub_ps(vmax, vmin);
    __m256 x;
    for (int i = rank*AVXLOAD; i < n; i+=size*AVXLOAD)
    {
        x = _mm256_load_ps(inputData+i);
        x = _mm256_div_ps(_mm256_sub_ps(x, vmin), diff);
        _mm256_store_ps(inputData+i, x);
    }
}

void readInputData(){
    
}

NeuralNetwork initNN(){
    NeuralNetwork n;
}
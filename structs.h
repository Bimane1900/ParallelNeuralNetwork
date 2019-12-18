/*
    Data structures for Neural Network 
*/

struct Layer{
    int id;
    float *w;
    float *output;
    float *bias;
    int inputColumns;
    int outputColumns;
} typedef Layer;

struct NeuralNetwork{
    Layer *hiddenLayers;
    double learningRate;
    float *inputLayer;
    float *testData;
    int nLayers;
} typedef NeuralNetwork;

#define AVXLOAD 8
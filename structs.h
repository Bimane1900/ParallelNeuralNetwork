/*
    Data structures for Neural Network 
*/

struct Layer{
    float *w;
    float *output;
    float *bias;
} typedef Layer;

struct NeuralNetwork{
    Layer *hiddenLayers;
    Layer *outputLayer;
    double learningRate = 0.5f;
    float *inputLayer;
    float *testData;
} typedef NeuralNetwork;
int nOfRows= 262144;
int nOfColumns = 128;
#define AVXLOAD 8
#define EMITTER 0
#define GATHERER 1
#define WORKERS 2
#define PROCESSES (WORKERS+2)
//nodes in hiddenlayers
#define NODESHL1 7
//inputLayer
#define ROWS 4//1151
#define COLUMNS 4//20
//#define ROWS 4 //testing NN
//#define COLUMNS 4
//HiddenLayer1
#define HL1ROWS (COLUMNS-1)
#define HL1COLUMNS NODESHL1
//Output Layer
#define OLROWS NODESHL1 //connect this to previous layers col
#define OLCOLUMNS 1
//data macros
#define FILENAME "testdata.txt" //"testdata.txt" "data.txt"
#define LAST_CHAR_OFFSET 5 ////3 in real data, 5 in test
#define EPOCHS 5000
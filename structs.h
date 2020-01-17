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
    double learningRate = 0.9f;
    float *inputLayer;
    float *testData;
} typedef NeuralNetwork;
int nOfRows= 8388608;
int nOfColumns = 4;
#define AVXLOAD 8
#define EMITTER 0
#define GATHERER 1
#define WORKERS 2
#define PROCESSES (WORKERS+2)
//nodes in hiddenlayers
#define NODESHL1 7
//inputLayer
#define ROWS nOfRows//10000000//800//8944//10000000//1151
#define COLUMNS nOfColumns//8//100000//8944//8//20
//#define ROWS 4 //testing NN
//#define COLUMNS 4
//HiddenLayer1
#define HL1ROWS (COLUMNS-1)
#define HL1COLUMNS NODESHL1
//Output Layer
#define OLROWS NODESHL1 //connect this to previous layers col
#define OLCOLUMNS 1
//data macros
#define FILENAME "data.txt" //"testdata.txt" "data.txt"
#define LAST_CHAR_OFFSET 3 ////3 in real data, 5 in test
#define EPOCHS 5
/*
    Data structures for Neural Network 
*/

struct Layer{
    int *id;
    float *w;
    float *output;
    float *bias;
    int inputColumns;
    int outputColumns;
} typedef Layer;

struct NeuralNetwork{
    Layer *hiddenLayers;
    Layer *outputLayer;
    double learningRate = 0.1f;
    float *inputLayer;
    float *testData;
    int nLayers;
} typedef NeuralNetwork;

#define AVXLOAD 8
#define EMITTER 0
#define GATHERER 1
#define WORKERS 2
#define PROCESSES 4
//nodes in hiddenlayers
#define NODESHL1 4
//#define NODESHL2 2
//inputLayer
//#define ROWS 1151
//#define COLUMNS 20
#define ROWS 4 //testing NN
#define COLUMNS 4
//HiddenLayer1
#define HL1ROWS (COLUMNS-1)
#define HL1COLUMNS NODESHL1
//hiddenLayer2
//#define HL2ROWS NODESHL1
//#define HL2COLUMNS NODESHL2
//Output Layer
#define OLROWS NODESHL1 //connect this to previous layers col
#define OLCOLUMNS 1
#define FILENAME "testdata.txt" //"testdata.txt" "data.txt"
#define LAST_CHAR_OFFSET 5 ////3 in real data, 5 in test
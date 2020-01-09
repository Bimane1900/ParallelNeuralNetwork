
void setupFeedforward();
void handleSetupFeedforward();
void recieveSetupFeedforward(NeuralNetwork* nn);
void initweights(float* weights);
float sigmoid(float x, int derivate);
void feedforward(NeuralNetwork* nn);
void handleFeedforward();
void calcOutput(float* inputSlice, float* weights, float* bias, float* output, int col, int row);
void reciveFeedforward(NeuralNetwork* nn);
void calcOL(float* inputSlice, float* weights, float* bias, float* output, int iter);
void calcOutputAVX(float* inputSlice, float* weights, float* bias, float* output, int col, int row);


float sigmoid(float x, int derivate){
    if(derivate)
        return x*(1-x);
    return 1/(1+expf(-x));
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
        mask = getAVXVectorMask(size-i); 
        rand_vector = _mm256_maskload_ps(weights+i, mask);
        rand_vector = _mm256_div_ps(rand_vector,RANDMAX_vec);
        rand_vector = _mm256_mul_ps(rand_vector,twos_vec);
        rand_vector = _mm256_sub_ps(rand_vector,one_vec);
        _mm256_maskstore_ps(weights+i, mask, rand_vector);
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
    free(status);
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
    free(status);
}

void feedforward(NeuralNetwork* nn){
    float* empty = NULL;
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(nn->hiddenLayers[0].w, (HL1ROWS*HL1COLUMNS), MPI_FLOAT, i, i, MPI_COMM_WORLD);
        MPI_Send(nn->hiddenLayers[0].bias, NODESHL1, MPI_FLOAT, i, i, MPI_COMM_WORLD);
        MPI_Send(nn->outputLayer[0].w, (OLROWS*OLCOLUMNS), MPI_FLOAT, i, i, MPI_COMM_WORLD);
        MPI_Send(nn->outputLayer[0].bias, 1, MPI_FLOAT, i, i, MPI_COMM_WORLD);
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
    float* OLweights = (float*)aligned_alloc(32, (OLROWS*OLCOLUMNS)*sizeof(float));
    float* OLbias = (float*)aligned_alloc(32, sizeof(float));
    float* HLoutput = (float*)aligned_alloc(32, NODESHL1*sizeof(float));
    float* OLoutput = (float*)aligned_alloc(32, sizeof(float));
    float* empty = NULL;
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));

    MPI_Recv(weights, (HL1ROWS*HL1COLUMNS), MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(bias, NODESHL1, MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(OLweights, (OLROWS*OLCOLUMNS), MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(OLbias, 1, MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);

 
    while(true){    
        MPI_Recv(inputSlice, HL1ROWS, MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        for (int i = 0; i < NODESHL1; i++)
        {
            HLoutput[i] = 0.0;
        }
        OLoutput[0] = 0.0;
        if (status->MPI_TAG == ROWS)
            break;
        
        //calcOutput(inputSlice,weights,bias,HLoutput,NODESHL1,HL1ROWS);
        calcOutputAVX(inputSlice, weights, bias, HLoutput, NODESHL1, HL1ROWS);
        //calcOutput(HLoutput, OLweights, OLbias, OLoutput, OLCOLUMNS, OLROWS);
        calcOutputAVX(HLoutput, OLweights, OLbias, OLoutput, OLCOLUMNS, OLROWS);
        MPI_Send(HLoutput, NODESHL1, MPI_FLOAT, EMITTER, status->MPI_TAG, MPI_COMM_WORLD);
        MPI_Send(OLoutput, 1, MPI_FLOAT, EMITTER, status->MPI_TAG, MPI_COMM_WORLD);
    }
    MPI_Send(&empty, 1, MPI_FLOAT, EMITTER, ROWS, MPI_COMM_WORLD);
    free(inputSlice);
    free(weights);
    free(bias);
    free(OLweights);
    free(OLbias);
    free(HLoutput);
    free(OLoutput);
    free(status);
}

void calcOL(float* inputSlice, float* weights, float* bias, float* output, int iter){ 
    for (int i = 0; i < iter; i++)
    {
        output[0] += weights[i]*inputSlice[i];
    }
    output[0] += bias[0];
    output[0] = sigmoid(output[0], 0);
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
            //printf("%d\n", (int)status->_ucount);
            if(status->_ucount > 4){
                for (int i = 0; i < NODESHL1; i++)
                {
                    nn->hiddenLayers[0].output[(status->MPI_TAG*NODESHL1)+i] = output[i];
                }
            }
            else{
                nn->outputLayer[0].output[status->MPI_TAG] = output[0];
                printf("%f\n", output[0]);
            }
        }
    }
   free(output);
   free(status);
}


void calcOutputAVX(float* inputSlice, float* weights, float* bias, float* output, int col, int row){
    
    __m256i mask;
    for (int i = 0; i < row; i++)
    {
        __m256 VinputSlice = _mm256_set1_ps(inputSlice[i]);
        for (int j = 0; j < col; j+=AVXLOAD)
        { 
            mask = getAVXVectorMask(col-j);
            __m256 Voutput = _mm256_maskload_ps(output+j, mask);
            __m256 Vweights = _mm256_maskload_ps(weights+(col*i)+j,mask);
            Voutput = _mm256_add_ps(Voutput, _mm256_mul_ps(Vweights,VinputSlice));
            _mm256_maskstore_ps(output+j, mask, Voutput);
        }
    }
    for(int i = 0; i < col; i++){
        mask = getAVXVectorMask(col-i);
        __m256 Voutput = _mm256_maskload_ps(output+i, mask);
        __m256 Vbias = _mm256_maskload_ps(bias+i, mask);
        Voutput = _mm256_add_ps(Voutput, Vbias);
        _mm256_maskstore_ps(output+i, mask, Voutput);
    }
    for (int i = 0; i < col; i++)
    {
        output[i] = sigmoid(output[i], 0);
    }
    
}


void calcOutput(float* inputSlice, float* weights, float* bias, float* output, int col, int row){
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            output[j] += weights[(i*col)+j]*inputSlice[i];
        }
    }
    
    
    for (int j = 0; j < col; j++)
    {
        // for (int i = 0; i < row; i++)
        // {
        //     output[j] += weights[(i*col)+j]*inputSlice[i];
        // }
        output[j] += bias[j];
        output[j] = sigmoid(output[j], 0);
    }
}
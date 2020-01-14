
void startInitiateWeights();
void initiateWeights();
void recieveInitialWeights(NeuralNetwork* nn);
void computeInitialWeights(float* weights);
void startFeedforward(NeuralNetwork* nn);
void handleFeedforward();
void recieveFeedforwardOutputs(NeuralNetwork* nn);
void calcOutputAVX(float* inputSlice, float* weights, float* bias, float* output, int col, int row);


//sets the inital weights to be used in feedforward
void computeInitialWeights(float* weights,int size)
{
    __m256 one_vec = _mm256_set1_ps(1.0f);
    __m256 twos_vec = _mm256_set1_ps(2.0f);
    __m256 RANDMAX_vec = _mm256_set1_ps(RAND_MAX);
    __m256 rand_vector;
    __m256i mask;
    for (int i = 0; i < size; i++)
    {
        //sequential randomize
        weights[i] = (float)rand();
    }
    for (int i = 0; i < size; i+=AVXLOAD)
	{
        //use masking incase size is not multiple of AVXLOAD(8)
        mask = getAVXVectorMask(size-i); 
        rand_vector = _mm256_maskload_ps(weights+i, mask);
        rand_vector = _mm256_div_ps(rand_vector,RANDMAX_vec);
        rand_vector = _mm256_mul_ps(rand_vector,twos_vec);
        rand_vector = _mm256_sub_ps(rand_vector,one_vec);
        _mm256_maskstore_ps(weights+i, mask, rand_vector);
	}
}

//signals the workers to start calculations
void startInitiateWeights()
{
    int WeightSize = ((HL1ROWS * HL1COLUMNS) + HL1COLUMNS)/WORKERS;
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(&WeightSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }     
}

//sets up initial values that are need before feedforward can begin
void initiateWeights()
{
    int nWeights;
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));
    MPI_Recv(&nWeights, 1, MPI_INT, EMITTER, 0, MPI_COMM_WORLD, status);
    float* weights = (float*)aligned_alloc(32, (nWeights)*sizeof(float));
    computeInitialWeights(weights,nWeights);
    //forwards initiated weights to gatherer
    MPI_Send(weights, nWeights, MPI_FLOAT, GATHERER, 1 , MPI_COMM_WORLD);
    free(weights);
    free(status);
}

//gathers the initial weights and initiates biases for feedforward
void recieveInitialWeights(NeuralNetwork* nn)
{
    int index = 0;
    //weights per chunk recieved
    int WeightSize = ((HL1ROWS * HL1COLUMNS) + HL1COLUMNS)/WORKERS;
    //num of weights in hiddenlayer
    int WeightHL1 = (HL1ROWS * HL1COLUMNS);
    float* weights = (float*)aligned_alloc(32, (WeightSize)*sizeof(float));
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Recv(weights, WeightSize, MPI_FLOAT, i, 1, MPI_COMM_WORLD, status);
        
        for (int j = 0; j < WeightSize; j++)
        {
            //weight belong to hiddenlayer if less than num of weights
            //in hiddenlayers, else its a weight for outputlayer
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
    
    //initiate biases
    for (int i = 0; i < NODESHL1; i++)
    {
        nn->hiddenLayers[0].bias[i] = 0;
    }
    for (int i = 0; i < OLCOLUMNS; i++){
        nn->outputLayer[0].bias[i] = 0;
    }
    
    free(weights);
    free(status);
}

//initiate feedforward process, send all data, weights and bias
void startFeedforward(NeuralNetwork* nn){
    float* empty = NULL;
    double sendTime = MPI_Wtime();
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(nn->hiddenLayers[0].w, (HL1ROWS*HL1COLUMNS), MPI_FLOAT, i, i, MPI_COMM_WORLD);
        MPI_Send(nn->hiddenLayers[0].bias, NODESHL1, MPI_FLOAT, i, i, MPI_COMM_WORLD);
        MPI_Send(nn->outputLayer[0].w, (OLROWS*OLCOLUMNS), MPI_FLOAT, i, i, MPI_COMM_WORLD);
        MPI_Send(nn->outputLayer[0].bias, 1, MPI_FLOAT, i, i, MPI_COMM_WORLD);
    }       
    //work is divided to workers by row
    for (int i = 0; i < ROWS; i++)
    {
        MPI_Send(nn->inputLayer+(i * HL1ROWS), HL1ROWS, MPI_FLOAT, (i%WORKERS)+2, i, MPI_COMM_WORLD);
    }   
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(&empty, 1, MPI_FLOAT, i, ROWS, MPI_COMM_WORLD);
    }
    printTime("startFeedforward took %f\n", MPI_Wtime()-sendTime);
}

//calculations in feedforward part
void handleFeedforward(){
    float* inputSlice = (float*)aligned_alloc(32, HL1ROWS*sizeof(float));
    float* weights = (float*)aligned_alloc(32, (HL1ROWS*HL1COLUMNS)*sizeof(float));
    float* bias = (float*)aligned_alloc(32, NODESHL1*sizeof(float));
    float* OLweights = (float*)aligned_alloc(32, (OLROWS*OLCOLUMNS)*sizeof(float));
    float* OLbias = (float*)aligned_alloc(32, sizeof(float));
    float* HLoutput = (float*)aligned_alloc(32, NODESHL1*sizeof(float));
    float* OLoutput = (float*)aligned_alloc(32, OLCOLUMNS*sizeof(float));
    float* empty = NULL;
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));

    //recv weights and biases
    MPI_Recv(weights, (HL1ROWS*HL1COLUMNS), MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(bias, NODESHL1, MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(OLweights, (OLROWS*OLCOLUMNS), MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(OLbias, 1, MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);

    double feedforwardTime = MPI_Wtime();
    while(true){    
        //input data is recv'd row by row
        MPI_Recv(inputSlice, HL1ROWS, MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        for (int i = 0; i < HL1COLUMNS; i++)
        {
            HLoutput[i] = 0.0;
        }
        for (int i = 0; i < OLCOLUMNS; i++)
        {
            OLoutput[i] = 0.0;
        }
        
        if (status->MPI_TAG == ROWS)
            break;
        
        calcOutputAVX(inputSlice, weights, bias, HLoutput, HL1COLUMNS, HL1ROWS);
        calcOutputAVX(HLoutput, OLweights, OLbias, OLoutput, OLCOLUMNS, OLROWS);
        
        
        //send output separately, outputlayer has a offset in tag
        MPI_Send(HLoutput, NODESHL1, MPI_FLOAT, EMITTER, status->MPI_TAG, MPI_COMM_WORLD);
        MPI_Send(OLoutput, 1, MPI_FLOAT, EMITTER, ROWS*COLUMNS+status->MPI_TAG, MPI_COMM_WORLD);
    }
    printTime("handleFeedforward: calculations complete %f\n", MPI_Wtime()-feedforwardTime);
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

//recieves and puts the calculated data into neural network
void recieveFeedforwardOutputs(NeuralNetwork* nn){
    float* output = (float*)aligned_alloc(32, NODESHL1*sizeof(float));
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));
    int term = 0; 
    double recieveTime = MPI_Wtime();
    while(term != WORKERS){  
        MPI_Recv(output, NODESHL1, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        if(status->MPI_TAG == ROWS)
            term++;
        else{
            //tag-offset to separate outputlayer and hiddenlayer outputs
            if(status->MPI_TAG < ROWS*COLUMNS){
                for (int i = 0; i < NODESHL1; i++)
                {
                    nn->hiddenLayers[0].output[(status->MPI_TAG*NODESHL1)+i] = output[i];
                }
            }
            else{
                nn->outputLayer[0].output[status->MPI_TAG-(ROWS*COLUMNS)] = output[0];
            }
        }
    }
    printTime("recieveFeedforwardOutputs: %f\n", MPI_Wtime()-recieveTime);
   free(output);
   free(status);
}

//calculates output of layers using AVX
//masking used to prevent overshooting sizes not multiple of AVXLOAD(8)
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

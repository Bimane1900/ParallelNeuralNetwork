/*
    Backpropagation functions
*/
void handleBackProp();
void setupBackProp(NeuralNetwork* nn);
void recieveBackProp(NeuralNetwork* nn);

void setupBackProp(NeuralNetwork* nn){
    float* empty = NULL;
    double sentTime = MPI_Wtime();
    //work is divided to workers by row
    for (int i = 0; i < ROWS; i++)
    {
        MPI_Send(nn->outputLayer[0].output+(i*OLCOLUMNS), OLCOLUMNS, MPI_FLOAT, (i%WORKERS)+2, i+(ROWS*COLUMNS), MPI_COMM_WORLD);
        MPI_Send(nn->hiddenLayers[0].output+(i*NODESHL1), NODESHL1, MPI_FLOAT, (i%WORKERS)+2, i, MPI_COMM_WORLD);
        MPI_Send(nn->testData+i, 1, MPI_FLOAT, (i%WORKERS)+2, i, MPI_COMM_WORLD);
    }
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(&empty, 1, MPI_FLOAT, i, ROWS, MPI_COMM_WORLD);
    }
      
}

void handleBackProp(){
    float* OLweights = (float*)aligned_alloc(32, (OLROWS*OLCOLUMNS)*sizeof(float));
    float* HLoutput = (float*)aligned_alloc(32, NODESHL1*sizeof(float));
    float* OLoutput = (float*)aligned_alloc(32, OLCOLUMNS*sizeof(float));
    float* deltaErrors = (float*)aligned_alloc(32, ROWS*sizeof(float));
    float* deltaErrorsHL1 = (float*)aligned_alloc(32, ROWS*NODESHL1*sizeof(float));
    float* derr2 = (float*)aligned_alloc(32, NODESHL1*NODESHL1*sizeof(float));
    float* DeltaErrorRes = (float*)aligned_alloc(32, NODESHL1*(COLUMNS-1)*sizeof(float));
    float* inputLayerOut = (float*)aligned_alloc(32, (COLUMNS-1)*sizeof(float));
    float* HLoutputStore = (float*)aligned_alloc(32, ROWS*NODESHL1*sizeof(float));
    float* empty = NULL;
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));

    for (int i = 0; i < HL1ROWS*HL1COLUMNS; i++)
    {
        DeltaErrorRes[i] = 0.0f;
    }

    __m256 VdeltaErr;
    __m256 Vdiff;
    __m256 VHLoutput;
    __m256i mask = getAVXVectorMask(NODESHL1);
    float testData;
    float diff;
    while(true){
        
        MPI_Recv(OLoutput, OLCOLUMNS, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        if(status->MPI_TAG == ROWS)
            break;
        MPI_Recv(HLoutput, NODESHL1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        MPI_Recv(&testData, 1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        //Compute the different from the output and the expected output
        diff = (OLoutput[0] - testData)/ROWS;
        diff = diff*sigmoid(OLoutput[0], 1);
        MPI_Send(&diff, 1, MPI_FLOAT, GATHERER, status->MPI_TAG, MPI_COMM_WORLD);
       //store HLoutput in a variable to contain them all for later, use tag for correct row
        _mm256_maskstore_ps(HLoutputStore+(status->MPI_TAG*NODESHL1), mask, _mm256_maskload_ps(HLoutput, mask));
    }
    MPI_Send(&empty, 1, MPI_FLOAT, GATHERER, ROWS, MPI_COMM_WORLD);
    MPI_Recv(deltaErrors, ROWS, MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(OLweights, (OLROWS*OLCOLUMNS), MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);

    //Use the calculated outputlayer errors and the outputlayer weights 
    //to calculate how much error hiddenlayer had
    #pragma region deltaErrorsHL1 = OLWeights * deltaErrors
    __m256 Vres, VdelErr, wei;
    for (int i = 0; i < NODESHL1; i++)
    {   
        mask = getAVXVectorMask(NODESHL1-i);
        wei = _mm256_maskload_ps(OLweights+i, mask);
        for (int j = 0; j < ROWS; j++)
        {
            Vres = _mm256_maskload_ps(deltaErrorsHL1+(j*NODESHL1)+i, mask);
            VdelErr = _mm256_set1_ps(deltaErrors[j]);
            Vres = _mm256_mul_ps(VdelErr, wei);
            _mm256_maskstore_ps(deltaErrorsHL1+(j*NODESHL1)+i, mask, Vres);
        }
    }
    #pragma endregion
    
    __m256 VsigmoidHL, VdelErrHL, HLDerr;
    while (true)
    {
         MPI_Recv(inputLayerOut, (COLUMNS-1), MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
         if(status->MPI_TAG == ROWS)
            break;
        
        //apply derivated sigmoid function on hiddenlayer output
        #pragma region HLoutput = sigmoid(HLoutput)
        for (int i = 0; i < NODESHL1; i++)
        {
            HLoutputStore[(status->MPI_TAG*NODESHL1)+i] = sigmoid(HLoutputStore[(status->MPI_TAG*NODESHL1)+i],1);
        }
        #pragma endregion

        //keep calculating the errors of hiddenlayer using the output it had
        #pragma region deltaErrorsHL1 = deltaErrorsHL1 * HLoutput 
        for (int i = 0; i < NODESHL1; i+=AVXLOAD)
        {
            mask = getAVXVectorMask(NODESHL1-i);
            VsigmoidHL = _mm256_maskload_ps(HLoutputStore+(status->MPI_TAG*NODESHL1)+i, mask);
            VdelErrHL = _mm256_maskload_ps(deltaErrorsHL1+(status->MPI_TAG*NODESHL1)+i, mask);
            HLDerr = _mm256_mul_ps(VdelErrHL, VsigmoidHL);
            _mm256_maskstore_ps(deltaErrorsHL1+(status->MPI_TAG*NODESHL1)+i, mask, HLDerr);
        }
        #pragma endregion

        //finally finish computation by adding inputdata into the calculation
        #pragma region DeltaErrorRes += deltaErrorsHL1*inputLayerOut
        mask = getAVXVectorMask(NODESHL1);
        HLDerr = _mm256_maskload_ps(deltaErrorsHL1+(status->MPI_TAG*NODESHL1), mask);
        for (int i = 0; i < COLUMNS-1; i++)
        {
            VHLoutput = _mm256_set1_ps(inputLayerOut[i]);
            VdeltaErr =_mm256_maskload_ps(DeltaErrorRes+NODESHL1*i, mask);
            VdeltaErr =  _mm256_add_ps( _mm256_mul_ps(VHLoutput,HLDerr), VdeltaErr);
            //VdeltaErr =  _mm256_mul_ps(VHLoutput,HLDerr);
            _mm256_maskstore_ps(DeltaErrorRes+NODESHL1*i, mask, VdeltaErr);
        }
        #pragma endregion  
     
    }
    MPI_Send(DeltaErrorRes, NODESHL1*(COLUMNS-1),MPI_FLOAT, GATHERER, status->MPI_TAG, MPI_COMM_WORLD);   
    
    free(HLoutputStore);
    free(status);
    free(derr2);
    free(DeltaErrorRes);
    free(OLweights);
    free(HLoutput);
    free(OLoutput);
    free(deltaErrors);
    free(deltaErrorsHL1);
    free(inputLayerOut);
}


void recieveBackProp(NeuralNetwork* nn){
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));
    float* outputDeltaErr = (float*)aligned_alloc(32, ROWS*sizeof(float));
    float output;
    float* empty = NULL;
    //aligned_alloc did not initate value to 0
    for (int i = 0; i < ROWS; i++)
    {
        outputDeltaErr[i] = 0.0f;
    }
    int term = 0; 
    while(term != WORKERS){  
        //recieve part of differences from each worker, tag contains index
        MPI_Recv(&output, 1, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, status);
            outputDeltaErr[status->MPI_TAG] = output;
        if(status->MPI_TAG == ROWS)
            term++;
    }
    
    //send summerized differences to each worker, they need it for calculations
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(outputDeltaErr, ROWS, MPI_FLOAT, i,i, MPI_COMM_WORLD);
        MPI_Send(nn->outputLayer[0].w, (OLROWS*OLCOLUMNS), MPI_FLOAT,i,i, MPI_COMM_WORLD);
    }

    //send inputs aswell, but do it row by row
     for (int i = 0; i < ROWS; i++)
    {
        MPI_Send(nn->inputLayer+(i*(COLUMNS-1)), (COLUMNS-1), MPI_FLOAT, (i%WORKERS)+2, i, MPI_COMM_WORLD);
    }

    //termination
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(&empty, 1, MPI_FLOAT, i, ROWS, MPI_COMM_WORLD);
    }

    int recv = 0;
    float hiddenLayerDeltaErr[NODESHL1*(COLUMNS-1)] __attribute__ ((aligned (32))) = {0.0f};
    float subsums[NODESHL1*(COLUMNS-1)] __attribute__ ((aligned (32))) = {0.0f} ;
    while (true)
    {
        //recieve parts of the total errors had in hiddenlayer calculations, summerize them!
        MPI_Recv(subsums, NODESHL1*(COLUMNS-1), MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        for (int i = 0; i < NODESHL1*(COLUMNS-1); i++)
        {
            hiddenLayerDeltaErr[i] += subsums[i];
        }
        
        recv++;
        if(recv == WORKERS){
            break;
        }
        
    }

    //update weights in outputLayer
    for (int i = 0; i < NODESHL1; i++)
    {
        nn->outputLayer[0].w[i] += -1.0*nn->learningRate*outputDeltaErr[i];
    }
    //update weights in hiddenLayer
    for (int i = 0; i < HL1ROWS*HL1COLUMNS; i++)
    {
        nn->hiddenLayers[0].w[i] += -1.0*nn->learningRate*hiddenLayerDeltaErr[i];
    }
   free(status);
   free(outputDeltaErr);
}
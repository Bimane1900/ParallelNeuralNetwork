
void handleBackProp();
void setupBackProp(NeuralNetwork* nn);
void recieveBackProp(NeuralNetwork* nn);

void setupBackProp(NeuralNetwork* nn){
    float* empty = NULL;
    double sentTime = MPI_Wtime();
    //for (int i = 2; i < PROCESSES; i++)
    //{
        //MPI_Send(nn->hiddenLayers[0].w, (HL1ROWS*HL1COLUMNS), MPI_FLOAT, i, i, MPI_COMM_WORLD);
        //MPI_Send(nn->hiddenLayers[0].bias, NODESHL1, MPI_FLOAT, i, i, MPI_COMM_WORLD);
        //MPI_Send(nn->outputLayer[0].w, (OLROWS*OLCOLUMNS), MPI_FLOAT, i, i, MPI_COMM_WORLD);
        //MPI_Send(nn->outputLayer[0].bias, 1, MPI_FLOAT, i, i, MPI_COMM_WORLD);
    //}
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
    printTime("setupBackProp: %f\n", MPI_Wtime()-sentTime);
      
}

void handleBackProp(){
    //float* HLweights = (float*)aligned_alloc(32, (HL1ROWS*HL1COLUMNS)*sizeof(float));
    //float* HLbias = (float*)aligned_alloc(32, NODESHL1*sizeof(float));
    float* OLweights = (float*)aligned_alloc(32, (OLROWS*OLCOLUMNS)*sizeof(float));
    //float* OLbias = (float*)aligned_alloc(32, OLCOLUMNS*sizeof(float));
    float* HLoutput = (float*)aligned_alloc(32, NODESHL1*sizeof(float));
    float* OLoutput = (float*)aligned_alloc(32, OLCOLUMNS*sizeof(float));
    float* deltaErrors = (float*)aligned_alloc(32, ROWS*sizeof(float));
    float* deltaErrorsHL1 = (float*)aligned_alloc(32, ROWS*NODESHL1*sizeof(float));
    float* derr2 = (float*)aligned_alloc(32, NODESHL1*NODESHL1*sizeof(float));
    float* DeltaErrorRes = (float*)aligned_alloc(32, NODESHL1*(COLUMNS-1)*sizeof(float));
    //printData(DeltaErrorRes, HL1ROWS, HL1COLUMNS);
    float* inputLayerOut = (float*)aligned_alloc(32, (COLUMNS-1)*sizeof(float));
    float* HLoutputStore = (float*)aligned_alloc(32, ROWS*NODESHL1*sizeof(float));
    //float* AccDiff = (float*)aligned_alloc(32, (ROWS)*NODESHL1*sizeof(float));
    float* empty = NULL;
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));

    for (int i = 0; i < HL1ROWS*HL1COLUMNS; i++)
    {
        DeltaErrorRes[i] = 0.0f;
    }
    
    //MPI_Recv(HLweights, (HL1ROWS*HL1COLUMNS), MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    //MPI_Recv(HLbias, NODESHL1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    //MPI_Recv(OLbias, 1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    __m256 VdeltaErr;
    __m256 Vdiff;
    __m256 VHLoutput;
    __m256i mask = getAVXVectorMask(NODESHL1);
    float testData;
    float diff;
    double recvTime = MPI_Wtime();
    while(true){
        
        MPI_Recv(OLoutput, OLCOLUMNS, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        if(status->MPI_TAG == ROWS)
            break;
        MPI_Recv(HLoutput, NODESHL1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        MPI_Recv(&testData, 1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        //10
        diff = (OLoutput[0] - testData)/ROWS;
        diff = diff*sigmoid(OLoutput[0], 1);
        MPI_Send(&diff, 1, MPI_FLOAT, GATHERER, status->MPI_TAG, MPI_COMM_WORLD);
        /*
        #pragma region What is this? 
        Vdiff = _mm256_set1_ps(diff);
        for (int i = 0; i < NODESHL1; i+=AVXLOAD)
        {
            mask = getAVXVectorMask(NODESHL1-i);
            VdeltaErr =_mm256_maskload_ps(deltaErrors+i, mask);
            VHLoutput = _mm256_maskload_ps(HLoutput+i, mask);
            VdeltaErr = _mm256_add_ps(VdeltaErr, _mm256_mul_ps(VHLoutput, Vdiff));
            _mm256_maskstore_ps(deltaErrors+i, mask, VdeltaErr);
        }
        #pragma endregion
        */
       //store HLoutput in a variable to contain them all for later, use tag for correct row
        _mm256_maskstore_ps(HLoutputStore+(status->MPI_TAG*NODESHL1), mask, _mm256_maskload_ps(HLoutput, mask));
    }
    printTime("###Took %f to recv OLoutput,HLoutput,testData & send diffs\n", MPI_Wtime()-recvTime);
    recvTime = MPI_Wtime();
    MPI_Send(&empty, 1, MPI_FLOAT, GATHERER, ROWS, MPI_COMM_WORLD);
    MPI_Recv(deltaErrors, ROWS, MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(OLweights, (OLROWS*OLCOLUMNS), MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    printTime("###Took %f to send empty, recv deltaErrors and OLwiehgts\n", MPI_Wtime()-recvTime);
    // for (int i = 0; i < ROWS; i++)
    // {
    //     for (int j = 0; j < NODESHL1; j++)
    //     {
    //         deltaErrorsHL1[NODESHL1*i+j] = deltaErrors[i]*OLweights[j];
    //     }
    // }
    recvTime = MPI_Wtime();
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
    printTime("###Took %f to compute deltaErrorsHL1\n", MPI_Wtime()-recvTime);
    
    __m256 VsigmoidHL, VdelErrHL, HLDerr, VHLres;
    recvTime = MPI_Wtime();
    while (true)
    {
         MPI_Recv(inputLayerOut, (COLUMNS-1), MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
         if(status->MPI_TAG == ROWS)
            break;
        
        #pragma region HLoutput = sigmoid(HLoutput)
        for (int i = 0; i < NODESHL1; i++)
        {
            HLoutputStore[(status->MPI_TAG*NODESHL1)+i] = sigmoid(HLoutputStore[(status->MPI_TAG*NODESHL1)+i],1);
        }
        #pragma endregion

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

        #pragma region DeltaErrorRes += deltaErrorsHL1*inputLayerOut
        mask = getAVXVectorMask(NODESHL1);
        HLDerr = _mm256_maskload_ps(deltaErrorsHL1+(status->MPI_TAG*NODESHL1), mask);
        for (int i = 0; i < COLUMNS-1; i++)
        {
            VHLoutput = _mm256_set1_ps(inputLayerOut[i]);
            VdeltaErr =  _mm256_add_ps( _mm256_mul_ps(VHLoutput,HLDerr), VdeltaErr);
            VdeltaErr =  _mm256_mul_ps(VHLoutput,HLDerr);
            _mm256_maskstore_ps(DeltaErrorRes+NODESHL1*i, mask, VdeltaErr);
        }
        #pragma endregion  
     
    }
    MPI_Send(DeltaErrorRes, NODESHL1*(COLUMNS-1),MPI_FLOAT, GATHERER, status->MPI_TAG, MPI_COMM_WORLD);   
    
    printTime("###Took %f to HLoutput,inputLayerout & send DeltaErrorRes\n", MPI_Wtime()-recvTime);
    recvTime = MPI_Wtime();
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
    printTime("###Took %f to free data structures in handle Backprop\n", MPI_Wtime()-recvTime);
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
    float dErr = 0.0;
    int term = 0; 
    double localTime = MPI_Wtime();
    while(term != WORKERS){  
        MPI_Recv(&output, 1, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, status);
            outputDeltaErr[status->MPI_TAG] = output;
        if(status->MPI_TAG == ROWS)
            term++;
    }
    printTime("output deltaErr gathered, %f\n", MPI_Wtime()-localTime);

    localTime= MPI_Wtime();
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(outputDeltaErr, ROWS, MPI_FLOAT, i,i, MPI_COMM_WORLD);
        MPI_Send(nn->outputLayer[0].w, (OLROWS*OLCOLUMNS), MPI_FLOAT,i,i, MPI_COMM_WORLD);
    }
    printTime("Gatherer sent OLLayer.w & outputDeltaErr: %f\n",MPI_Wtime()-localTime);

     for (int i = 0; i < ROWS; i++)
    {
        MPI_Send(nn->inputLayer+(i*(COLUMNS-1)), (COLUMNS-1), MPI_FLOAT, (i%WORKERS)+2, i, MPI_COMM_WORLD);
    }

    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(&empty, 1, MPI_FLOAT, i, ROWS, MPI_COMM_WORLD);
    }

    int recv = 0;
    float hiddenLayerDeltaErr[NODESHL1*(COLUMNS-1)] __attribute__ ((aligned (32))) = {0.0f};
    float subsums[NODESHL1*(COLUMNS-1)] __attribute__ ((aligned (32))) = {0.0f} ;
    //printData(hiddenLayerDeltaErr, HL1ROWS, HL1COLUMNS);
    //printData(subsums, HL1ROWS, HL1COLUMNS);
    localTime = MPI_Wtime();
    while (true)
    {
        
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
    //printData(hiddenLayerDeltaErr, HL1ROWS, HL1COLUMNS);
    printTime("recieved hiddenLayerDeltaErr %f\n", MPI_Wtime()-localTime);
    localTime = MPI_Wtime();
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
    printTime("weights updated %f\n", MPI_Wtime()-localTime);
   free(status);
   free(outputDeltaErr);
}
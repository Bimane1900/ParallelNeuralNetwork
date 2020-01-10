
void handleBackProp();
void setupBackProp(NeuralNetwork* nn);
void recieveBackProp(NeuralNetwork* nn);

void setupBackProp(NeuralNetwork* nn){
    float* empty = NULL;
    MPI_Send(nn->hiddenLayers[0].output, ROWS*NODESHL1, MPI_FLOAT, GATHERER,0,MPI_COMM_WORLD);

    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(nn->testData, ROWS, MPI_FLOAT, i, i, MPI_COMM_WORLD);
        MPI_Send(nn->hiddenLayers[0].w, (HL1ROWS*HL1COLUMNS), MPI_FLOAT, i, i, MPI_COMM_WORLD);
        MPI_Send(nn->hiddenLayers[0].bias, NODESHL1, MPI_FLOAT, i, i, MPI_COMM_WORLD);
        //MPI_Send(nn->outputLayer[0].w, (OLROWS*OLCOLUMNS), MPI_FLOAT, i, i, MPI_COMM_WORLD);
        MPI_Send(nn->outputLayer[0].bias, 1, MPI_FLOAT, i, i, MPI_COMM_WORLD);
    }
    //work is divided to workers by row
    for (int i = 0; i < ROWS; i++)
    {
        MPI_Send(nn->outputLayer[0].output+(i*OLCOLUMNS), OLCOLUMNS, MPI_FLOAT, (i%WORKERS)+2, i+(ROWS*COLUMNS), MPI_COMM_WORLD);
        MPI_Send(nn->hiddenLayers[0].output+(i*NODESHL1), NODESHL1, MPI_FLOAT, (i%WORKERS)+2, i, MPI_COMM_WORLD);
    }
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(&empty, 1, MPI_FLOAT, i, ROWS, MPI_COMM_WORLD);
    }
      
}

void handleBackProp(){
    float* testData = (float*)aligned_alloc(32, ROWS*sizeof(float));
    float* HLweights = (float*)aligned_alloc(32, (HL1ROWS*HL1COLUMNS)*sizeof(float));
    float* HLbias = (float*)aligned_alloc(32, NODESHL1*sizeof(float));
    float* OLweights = (float*)aligned_alloc(32, (OLROWS*OLCOLUMNS)*sizeof(float));
    float* OLbias = (float*)aligned_alloc(32, OLCOLUMNS*sizeof(float));
    float* HLoutput = (float*)aligned_alloc(32, NODESHL1*sizeof(float));
    float* OLoutput = (float*)aligned_alloc(32, OLCOLUMNS*sizeof(float));
    float* deltaErrors = (float*)aligned_alloc(32, NODESHL1*sizeof(float));
    float* deltaErrorsHL1 = (float*)aligned_alloc(32, ROWS*NODESHL1*sizeof(float));
    float* derr2 = (float*)aligned_alloc(32, NODESHL1*NODESHL1*sizeof(float));
    float* DeltaErrorRes = (float*)aligned_alloc(32, (COLUMNS-1)*sizeof(float));
    float* inputLayerOut = (float*)aligned_alloc(32, (COLUMNS-1)*sizeof(float));
    float* AccDiff = (float*)aligned_alloc(32, (ROWS)*2*sizeof(float));
    
    float* empty = NULL;
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));

    MPI_Recv(testData, ROWS, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(HLweights, (HL1ROWS*HL1COLUMNS), MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(HLbias, NODESHL1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(OLbias, 1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    __m256 VdeltaErr;
    __m256 Vdiff;
    __m256 VHLoutput;
    __m256i mask;

    float diff;
    
    while(true){
        
        MPI_Recv(OLoutput, OLCOLUMNS, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        if(status->MPI_TAG == ROWS)
            break;
        MPI_Recv(HLoutput, NODESHL1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        //10
        diff = (OLoutput[0] - testData[status->MPI_TAG])/ROWS;
        diff = diff*sigmoid(OLoutput[0], 1);
        MPI_Send(&diff, 1, MPI_FLOAT, GATHERER, status->MPI_TAG, MPI_COMM_WORLD);
        Vdiff = _mm256_set1_ps(diff);
        for (int i = 0; i < NODESHL1; i+=AVXLOAD)
        {
            mask = getAVXVectorMask(NODESHL1-i);
            VdeltaErr =_mm256_maskload_ps(deltaErrors+i, mask);
            VHLoutput = _mm256_maskload_ps(HLoutput+i, mask);
            VdeltaErr = _mm256_add_ps(VdeltaErr, _mm256_mul_ps(VHLoutput, Vdiff));
            _mm256_maskstore_ps(deltaErrors+i, mask, VdeltaErr);
        }
    }

    MPI_Send(&empty, 1, MPI_FLOAT, GATHERER, ROWS, MPI_COMM_WORLD);
    MPI_Recv(deltaErrors, ROWS, MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(OLweights, (OLROWS*OLCOLUMNS), MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    
    // for (int i = 0; i < ROWS; i++)
    // {
    //     for (int j = 0; j < NODESHL1; j++)
    //     {
    //         deltaErrorsHL1[NODESHL1*i+j] = deltaErrors[i]*OLweights[j];
    //     }
    // }
    //printData(deltaErrors,ROWS,1);
    //printData(OLweights,NODESHL1,1);

    __m256 Vres, VdelErr, wei;
    for (int i = 0; i < NODESHL1; i++)
    {
        wei = _mm256_set1_ps(OLweights[i]);
        for (int j = 0; j < ROWS; j++)
        {
            mask = getAVXVectorMask(NODESHL1-i);
            //Vres = _mm256_set1_ps(deltaErrorsHL[j]);
            Vres = _mm256_maskload_ps(deltaErrorsHL1+(i*NODESHL1)+j, mask);
            //VdelErr = _mm256_maskload_ps(deltaErrors+j, mask);
            VdelErr = _mm256_set1_ps(deltaErrors[j]);
            Vres = _mm256_mul_ps(VdelErr, wei);
            _mm256_maskstore_ps(deltaErrorsHL1+(i*NODESHL1)+j, mask, Vres);
            //res[(i*NODESHL1)+j] =  w[i] * err[j];
            //res[(i*NODESHL1)+j] =  OLweights[i] * deltaErrors[j];
        }
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    if(rank == 2)
        printData(deltaErrorsHL1,ROWS,NODESHL1);
    __m256 VsigmoidHL, VdelErrHL, HLDerr, VHLres;
    while (true)
    {
         MPI_Recv(HLoutput, NODESHL1, MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
         if(status->MPI_TAG == ROWS)
            break;
         MPI_Recv(inputLayerOut, COLUMNS-1, MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        
        for (int i = 0; i < NODESHL1; i++)
        {
            HLoutput[i] = sigmoid(HLoutput[i],1);
        }

        for (int i = 0; i < NODESHL1; i+=AVXLOAD)
        {
            mask = getAVXVectorMask(NODESHL1-i);
            VsigmoidHL = _mm256_maskload_ps(HLoutput+i, mask);
            VdelErrHL = _mm256_maskload_ps(deltaErrorsHL1+(status->MPI_TAG*NODESHL1)+i, mask);
            HLDerr = _mm256_mul_ps(VdelErrHL, VsigmoidHL);
            _mm256_maskstore_ps(deltaErrorsHL1+(status->MPI_TAG*NODESHL1)+i, mask, HLDerr);
        }
        
        for (int i = 0; i < NODESHL1; i+=AVXLOAD)
        {
            mask = getAVXVectorMask(NODESHL1-i);
            VdeltaErr =_mm256_maskload_ps(DeltaErrorRes+i, mask);
            VHLoutput = _mm256_maskload_ps(inputLayerOut+(COLUMNS-1)+i, mask);
            HLDerr = _mm256_maskload_ps(deltaErrorsHL1+(status->MPI_TAG*NODESHL1)+i, mask);
            VdeltaErr = _mm256_add_ps(VdeltaErr, _mm256_mul_ps(VHLoutput,HLDerr));
            _mm256_maskstore_ps(DeltaErrorRes+i, mask, VdeltaErr);
        }
    }

    
    //printf("worker:\n");
    //printData(deltaErrorsHL1, NODESHL1, NODESHL1);
    //printData(DeltaErrorRes, HL1ROWS, 1);
    //printf("1:\n");


}


void recieveBackProp(NeuralNetwork* nn){
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));
    MPI_Recv(nn->hiddenLayers[0].output, ROWS*NODESHL1, MPI_FLOAT, EMITTER, MPI_ANY_TAG ,MPI_COMM_WORLD,status);
    float* deltaErrors = (float*)aligned_alloc(32, ROWS*sizeof(float));
    float output;
    float* empty = NULL;
    //aligned_alloc did not initate value to 0
    for (int i = 0; i < ROWS; i++)
    {
        deltaErrors[i] = 0.0f;
    }
    float dErr = 0.0;
    int term = 0; 
    while(term != WORKERS){  
        MPI_Recv(&output, 1, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, status);
            //printf("%i",status->MPI_TAG);
            deltaErrors[status->MPI_TAG] = output;
        if(status->MPI_TAG == ROWS)
            term++;
        //printData(deltaErrors, NODESHL1, 1);
    }
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(deltaErrors, ROWS, MPI_FLOAT, i,i, MPI_COMM_WORLD);
        MPI_Send(nn->outputLayer[0].w, (OLROWS*OLCOLUMNS), MPI_FLOAT,i,i, MPI_COMM_WORLD);
    }

     for (int i = 0; i < ROWS; i++)
    {
        MPI_Send(nn->hiddenLayers[0].output+(i*NODESHL1), NODESHL1, MPI_FLOAT, (i%WORKERS)+2, i, MPI_COMM_WORLD);
        MPI_Send(nn->inputLayer+(i*(COLUMNS-1)), (COLUMNS-1), MPI_FLOAT, (i%WORKERS)+2, i, MPI_COMM_WORLD);

    }

    for (int i = 0; i < NODESHL1; i++)
    {
        nn->outputLayer[0].w[i] += -1.0*deltaErrors[i];//nn->learningRate*deltaErrors[i];
    }
    
   for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(&empty, 1, MPI_FLOAT, i, ROWS, MPI_COMM_WORLD);
    }

   free(status);
}
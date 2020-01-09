
void handleBackProp();
void setupBackProp(NeuralNetwork* nn);
void recieveBackProp(NeuralNetwork* nn);

void setupBackProp(NeuralNetwork* nn){
    float* empty = NULL;
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(nn->testData, ROWS, MPI_FLOAT, i, i, MPI_COMM_WORLD);
        MPI_Send(nn->hiddenLayers[0].w, (HL1ROWS*HL1COLUMNS), MPI_FLOAT, i, i, MPI_COMM_WORLD);
        MPI_Send(nn->hiddenLayers[0].bias, NODESHL1, MPI_FLOAT, i, i, MPI_COMM_WORLD);
        MPI_Send(nn->outputLayer[0].w, (OLROWS*OLCOLUMNS), MPI_FLOAT, i, i, MPI_COMM_WORLD);
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
    float* empty = NULL;
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));

    MPI_Recv(testData, ROWS, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(HLweights, (HL1ROWS*HL1COLUMNS), MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(HLbias, NODESHL1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(OLweights, (OLROWS*OLCOLUMNS), MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
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
        
        diff = (OLoutput[0] - testData[status->MPI_TAG]);
        diff = diff*sigmoid(OLoutput[0], 1);
        Vdiff = _mm256_set1_ps(diff);
        for (int i = 0; i < NODESHL1; i+=AVXLOAD)
        {
            mask = getAVXVectorMask(NODESHL1-i);
            VdeltaErr =_mm256_maskload_ps(deltaErrors+i, mask);
            VHLoutput = _mm256_maskload_ps(HLoutput+i, mask);
            VdeltaErr = _mm256_add_ps(VdeltaErr, _mm256_mul_ps(VHLoutput, Vdiff));
            _mm256_maskstore_ps(deltaErrors+i, mask, VdeltaErr);
        }
        // for (int i = 0; i < NODESHL1; i++)
        // {
        //     deltaErrors[i] += diff *HLoutput[i];
        // }
        
        

    }
    MPI_Send(deltaErrors, NODESHL1, MPI_FLOAT, GATHERER, ROWS, MPI_COMM_WORLD);
    MPI_Recv(deltaErrors, NODESHL1, MPI_FLOAT, GATHERER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    printf("worker:\n");
    printData(deltaErrors, NODESHL1, 1);

}


void recieveBackProp(NeuralNetwork* nn){
    float* deltaErrors = (float*)aligned_alloc(32, NODESHL1*sizeof(float));
    float* output = (float*)aligned_alloc(32, NODESHL1*sizeof(float));
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));
    int term = 0; 
    while(term != WORKERS){  
        MPI_Recv(output, NODESHL1, MPI_FLOAT, MPI_ANY_SOURCE, ROWS, MPI_COMM_WORLD, status);
        for (int i = 0; i < NODESHL1; i++)
        {
            deltaErrors[i] += output[i];
        }
        term++;
    }
    printf("gather:\n");
    printData(deltaErrors, NODESHL1, 1);
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(deltaErrors, NODESHL1, MPI_FLOAT, i,i, MPI_COMM_WORLD);
    }
    
   free(output);
   free(status);
}
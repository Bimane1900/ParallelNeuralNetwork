
void setupFeatureScaling(float *data, int n);
void featureScale(float *inputData, float max, float min, int n);
void handleFeatureScaling(int n);
void recieveFeatureScaledData(float* data, int n);

//Reads data, finds min/max and forwards data to workers
void setupFeatureScaling(float *data, int n){
    float max = 0.0f, min = 20000.0f;
    find_min_max(data, ROWS, COLUMNS-1, &max, &min);
    float *empty = NULL;
    //send min/max to workers
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(&min, 1, MPI_FLOAT, i, n, MPI_COMM_WORLD);
        MPI_Send(&max, 1, MPI_FLOAT, i, n, MPI_COMM_WORLD);
    }

    //send data in chunks of 8 floats
    int sendTo = 0;
    for (int i = 0; i < n; i+=AVXLOAD)
    {
        MPI_Send(data+i, AVXLOAD, MPI_FLOAT, (sendTo%WORKERS)+2, i, MPI_COMM_WORLD);
        sendTo++;
        
    }

    //send termination
    for (int i = 2; i < PROCESSES; i++)
    {
        MPI_Send(&empty, 1, MPI_FLOAT, i, n, MPI_COMM_WORLD);
    }
}

//Recieves data, featureScales it and forwards to a reciever
void handleFeatureScaling(int n){
    float *data = (float*)aligned_alloc(32,sizeof(float)*AVXLOAD);
    float min,max;
    float* empty = NULL;
    float sendSize = AVXLOAD;
        
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));
    //recv min and max
    MPI_Recv(&min, 1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    MPI_Recv(&max, 1, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
    while(true){
        MPI_Recv(data, AVXLOAD, MPI_FLOAT, EMITTER, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        if(status->MPI_TAG == n) // if tag == n then its termination time
            break;
        //if we are about to overshoot AVXLOAD-size, we adjust sendSize
        if(status->MPI_TAG+AVXLOAD > n){
            sendSize = n-status->MPI_TAG;
            featureScale(data, max, min, sendSize);
        }
        else{
            featureScale(data, max, min, AVXLOAD);
            sendSize = AVXLOAD;
        }
        //forward scaled data to gatherer
        MPI_Send(data, sendSize, MPI_FLOAT, GATHERER, status->MPI_TAG, MPI_COMM_WORLD);
    }
    //send termination to gatherer
    MPI_Send(&empty, 1, MPI_FLOAT, GATHERER, n, MPI_COMM_WORLD);
    free(data);
    free(status);
}

//Normalizes data
void featureScale(float *inputData, float max, float min, int n){
    //mask incase n != AVX vector size
    __m256i mask = getAVXVectorMask(n);
    __m256 vmax = _mm256_set1_ps(max);
    __m256 vmin = _mm256_set1_ps(min);
    __m256 diff = _mm256_sub_ps(vmax, vmin);
    __m256 x = _mm256_maskload_ps(inputData, mask);
    x = _mm256_sub_ps(x, vmin);
    x = _mm256_div_ps(x, diff);
    _mm256_maskstore_ps(inputData, mask, x);

}

//Wait for featureScaled data and read it
void recieveFeatureScaledData(float* data, int n){
    float *incoming = (float*)aligned_alloc(32,sizeof(float)*AVXLOAD);
    int term = 0; //keep track of workers terminating
    int count = 0;
    MPI_Status *status = (MPI_Status*)malloc(sizeof(MPI_Status));
    while (term != WORKERS)
    {
        //recv from any worker
        MPI_Recv(incoming, AVXLOAD, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, status);
        //get the count of recv data incase total data size is not multiple of AVXLOAD(8)
        MPI_Get_count(status, MPI_FLOAT, &count);
        if(status->MPI_TAG == n)
            term++;
        else{
            //store incoming data
            for (int i = 0; i < count; i++)
            {
                //tag is used to indicate index thoughout the message passing
                data[i+status->MPI_TAG] = incoming[i];
            }
        }

    }
    free(incoming);
    free(status);
}
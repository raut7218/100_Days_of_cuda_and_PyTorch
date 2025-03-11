#include<iostream>
__global__ void Matrixadd_A(const float* A, const float*B, float* C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i<N){
        for (int j = 0; j < N; j++){
            int pos = i*N + j;
            C[pos] = A[pos] + B[pos];
        }
    }
}

__global__ void Matrixadd_B(const float* A,const float *B,float *C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if ((i>=N) && (j>=N)){return ;}

    int pos = i*N + j;
    C[pos] = A[pos] + B[pos];
}

__global__ void Matrixadd_C(const float* A,const float *B,float *C, int N){
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (j<N){
        for (int i = 0; i < N; i++){
            int pos = i*N + j;
            C[pos] = A[pos] + B[pos];
        }
    }
}

void printmatrix(const char* name, const float* matrix, int N){
    printf("%s:\n", name);
    for (int i =0;i<N; i++){
        for (int j = 0;j<N;j++){
            printf("%.2f", matrix[i*N + j]);
        }
        printf("\n");
    }
}

int main() {
    const int N = 5;
    float *A, *B, *C;
    A = (float*)malloc(N*N* sizeof(float));
    B = (float*)malloc(N*N* sizeof(float));
    C = (float*)malloc(N*N* sizeof(float));
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            int pos = i*N + j;
            A[pos] = 1.0f;
            B[pos] = 2.0f;
            C[pos] = 0.0f;
        }
    }

    float *A_d, *B_d,*C_d;
    cudaMalloc((void **)&A_d,N*N*sizeof(float));
    cudaMalloc((void **)&B_d,N*N*sizeof(float));
    cudaMalloc((void **)&C_d,N*N*sizeof(float));

    cudaMemcpy(A_d,A,N*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B,N*N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 dimBlock(32,16);
    dim3 dimGrid(ceil(N/32.0f),ceil(N/16.0f));
    Matrixadd_A<<<dimGrid,dimBlock>>>(A_d,B_d,C_d,N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, C_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    printmatrix("C",C,N);
    printmatrix("A",A,N);
    printmatrix("B",B,N);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A);
    free(B);
    free(C);
}
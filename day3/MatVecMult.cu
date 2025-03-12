#include <iostream>

__global__ void vecMatMul(float *A,float *B, float *C,int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<N){
        float sum = 0.0f;
        for (int j = 0; j<N; j++){
            sum += A[i*N + j] * B[j];
        }
        C[i] = sum;
    }
}

int main(){
    int N = 10;
    float *A,*B,*C;
    A = (float *)malloc(N*N*sizeof(float));
    B = (float *)malloc(N*N*sizeof(float));
    C = (float *)malloc(N*N*sizeof(float));

    for (int i = 0; i< N; i++){
        for (int j = 0; j<N;j++){
            A[i*N + j] = 1.0f;
        }
        B[i] = 2.0f;
        C[i] = 0.0f;
        }
    
    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d,N*N*sizeof(float));
    cudaMalloc(&B_d,N*sizeof(float));
    cudaMalloc(&C_d,N*sizeof(float));
    cudaMemcpy(A_d,A,N*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B,N*sizeof(float),cudaMemcpyHostToDevice);
    int BlockSize = 1024;
    int gridsize = (N+BlockSize-1)/ BlockSize;
    vecMatMul<<<gridsize,BlockSize>>>(A_d,B_d,C_d,N);
    cudaDeviceSynchronize();
    cudaMemcpy(C,C_d,N*sizeof(float),cudaMemcpyDeviceToHost);
    printf("A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            printf("%.2f ", A[i * N + j]); 
        }
        printf("\n"); 
    }
    printf("\n");
     printf("B:\n");
    for (int i = 0; i < N; i++) {


            printf("%.2f ", B[i ]); 

    }
    printf("\n");
    printf("C:\n");
    for (int i = 0; i < N; i++) {


            printf("%.2f ",C[i]); 

    }
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
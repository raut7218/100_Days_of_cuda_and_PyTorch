import torch 
import math

def main():
  N = 10
  A = torch.ones((N,N),dtype=torch.float32)
  B = torch.full((N,N),2.0,dtype=torch.float32)
  C = torch.zeros((N,N),dtype=torch.float32)

  device = "cuda" 
  A = A.to(device)
  B = B.to(device)
  C = C.to(device)

  C = A + B

  C = C.cpu()

  print_matrix("C", C, N)
  print_matrix("A", A, N)
  print_matrix("B", B, N)

def print_matrix(name, matrix, N):
  print(f"{name}:")
  for i in range(N):
    for j in range(N):
      print(f"{matrix[i, j]:.2f}",end = " ")
    print()

if __name__ == "__main__":
  main()


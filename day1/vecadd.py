import torch
import numpy as np

def main():
  N = 10
  A = torch.tensor([
    float(i)  for i in range(N)
  ],dtype = torch.float32)

  B = torch.tensor([
     float(i*2.0) for i in range(N)
  ],dtype = torch.float32)
#device = 'cpu' if gpu is not present
  device = 'cuda'

  A = A.to(device)
  B = B.to(device)

  C = A + B

  A_cpu = A.cpu()
  B_cpu = B.cpu()
  C_cpu = C.cpu()

  print("vecotr addition:")
  for i in range(N):
    print(f"{A_cpu[i]} + {B_cpu[i]} = {C_cpu[i]}")

if __name__ == "__main__":
  main()

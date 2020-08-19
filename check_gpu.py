import torch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#print(torch.cuda.device_count())
#for i in range(0):
print(torch.cuda.get_device_name(0))

print(torch.cuda.current_device())
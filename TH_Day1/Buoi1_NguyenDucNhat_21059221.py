

##-------------------------------- Load from list
print("##--------------------------------Load from list--------------------------------")
import torch
lst = [[1,2,3], [4,5,6]]
tensor = torch.tensor(lst)
print(tensor)

#-------------------------------- Load from NumPy array
import numpy as np
import torch
print("##--------------------------------Load from NumPy array--------------------------------")
# Tạo một mảng NumPy
np_array = np.array([[1, 2], [3, 4]])

# Chuyển đổi NumPy array thành PyTorch tensor
np_tensor = torch.from_numpy(np_array)

print("Mảng NumPy:")
print(np_array)

print("\nTensor PyTorch:")
print(np_tensor)

# Thay đổi giá trị trong tensor sẽ ảnh hưởng đến NumPy array
np_tensor[0, 0] = 99
print("\nSau khi thay đổi giá trị trong tensor:")
print(np_array)  # Mảng NumPy cũng thay đổi

#-------------------------------- Tensor attributes
print("##--------------------------------Tensor attributes--------------------------------")
# Tensor shape
tensor = torch.tensor(lst)
print("tensor.shape: ",tensor.shape)
# Tensor device
print("tensor.device: ",tensor.device)
# Tensor data type
print("tensor.dtype: ", tensor.dtype)

#-------------------------------- Getting started with tensor operatons
print("##--------------------------------Getting started with tensor operatons--------------------------------")
# Compatible shapes
# addition / subtraction
a = torch.tensor([[1,1], [2,2]])
b = torch.tensor([[2,2], [3,3]])
print("a + b: ", a+b)
# Incompatible shapes
# addition / subtraction
# --> Error: "RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1 "
# a = torch.tensor([[1,1], [2,2]])
# c = torch.tensor([[2,2,4], [3,3,5]])
# print("a + c: ", a + c)
# Element-wise multiplication
print("a * b: ", a * b)





#-------------------------------- Our first neural network
print("##--------------------------------Our first neural network--------------------------------")

# input - output
import torch.nn as nn

# create input_tensor with three features
input_tensor = torch.tensor(
    [[0.3471, 0.4547, -0.2356]]
)
# Define our first linear layer
linear_layer = nn.Linear(in_features=3, out_features=2)
# Pass input through linear layer
output = linear_layer(input_tensor)
print("Print output: ", output)

#-------------------------------- Getting to know the linear layer operation
print("##--------------------------------Getting to know the linear layer operation--------------------------------")
# .weight
print("--linear_layer.weight: ",linear_layer.weight) 
# .bias
print("--linear_layer.bias: ",linear_layer.bias) 


#-------------------------------- Stacking layers with nn.Senquential()
# Output = linear_layer(input_tensor)
# for input X, weight W0, bias b0
# y0 = W0.X + b0
# Create etwork with three linear layers
# output = linear_layer(input_tensor)
# model = nn.Sequential(
#     nn.Linear(10, 18),
#     nn.Linear(18, 20),
#     nn.Linear(20,5)
# )
# print(model)
# print(input_tensor)

# output_tensor = model(input_tensor)
# print(output_tensor)


#-------------------------------- Binary classification: forward pass
print("##--------------------------------Binary classification: forward pass--------------------------------")

import torch
import torch.nn as nn

# Tạo dữ liệu đầu vào (5 mẫu, 6 đặc trưng)
input_data = torch.tensor([
    [-0.4421, 1.5207, 2.0607, -0.3647, 0.4691, 0.0946],
    [-0.9155, -0.0475, -1.3645, 0.6336, -1.9520, -0.3398],
    [0.7406, 1.6763, -0.8511, 0.2432, 0.1123, -0.0633],
    [-1.6630, -0.0718, -0.1285, 0.5396, -0.0288, -0.8622],
    [-0.7413, 1.7920, -0.0883, -0.6685, 0.4745, -0.4245]
])

# Tạo mô hình phân loại nhị phân
model = nn.Sequential(
    nn.Linear(6, 4),  # Lớp tuyến tính đầu tiên (6 đầu vào, 4 đầu ra)
    nn.Linear(4, 1),  # Lớp tuyến tính thứ hai (4 đầu vào, 1 đầu ra)
    nn.Sigmoid()     # Hàm kích hoạt Sigmoid (đưa đầu ra về khoảng 0-1)
)

# Đưa dữ liệu qua mô hình (forward pass)
output = model(input_data)

# In kết quả
print("Dữ liệu đầu vào:")
print(input_data)
print("\nĐầu ra (xác suất):")
print(output)

#Kiểm tra shape của output
print("\nShape của output:")
print(output.shape)

print("##--------------------------------Multi-class classification: forward pass--------------------------------")

import torch
import torch.nn as nn

# Số lượng lớp
n_classes = 3

# Tạo dữ liệu đầu vào (ví dụ: 5 mẫu, 6 đặc trưng)
input_data = torch.randn(5, 6) # Sử dụng dữ liệu ngẫu nhiên cho ví dụ

# Tạo mô hình
model = nn.Sequential(
    nn.Linear(6, 4),
    nn.Linear(4, n_classes),
    nn.Softmax(dim=-1)
)

# Forward pass
output = model(input_data)

# In kết quả
print("Shape của input_data:", input_data.shape)
print("Output:")
print(output)
print("Shape của output:", output.shape)

print("##--------------------------------Regression: forward pass--------------------------------")

# Tạo dữ liệu đầu vào (ví dụ: 5 mẫu, 6 đặc trưng)
input_data = torch.randn(5, 6) # Sử dụng dữ liệu ngẫu nhiên cho ví dụ

# Tạo mô hình hồi quy
model = nn.Sequential(
    nn.Linear(6, 4),
    nn.Linear(4, 1)
)

# Forward pass
output = model(input_data)

# In kết quả
print("Shape của input_data:", input_data.shape)
print("Output:")
print(output)
print("Shape của output:", output.shape)


##--------------------------------Cross entropy loss in PyTorch--------------------------------
print("##--------------------------------Cross entropy loss in PyTorch--------------------------------")
import torch
from torch.nn import CrossEntropyLoss

scores = torch.tensor([[-0.1211, 0.1059]])
target = torch.tensor([0]) # Chỉ số lớp

criterion = CrossEntropyLoss()
loss = criterion(scores, target)
print(loss)
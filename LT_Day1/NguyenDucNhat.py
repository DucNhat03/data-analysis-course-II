import torch

# Đầu vào (điểm trung bình, điểm hạnh kiểm, điểm rèn luyện)
inputs = torch.tensor([9, 3, 8], dtype=torch.float32)  # [x1, x2, x3]

# Trọng số (w1, w2, w3)
weights = torch.tensor([2, 1, 1], dtype=torch.float32)  # [w1, w2, w3]

# Bias
bias = torch.tensor(-24, dtype=torch.float32)

# Tính toán điểm danh hiệu
v = torch.dot(weights, inputs) + bias

# Hàm kích hoạt (Step Function)
output = 1 if v >= 0 else 0

# In kết quả
print(f"Điểm danh hiệu: {v.item()}")
print(f"Sinh viên đạt danh hiệu giỏi toàn diện: {bool(output)}")

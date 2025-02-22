# Import các thư viện cần thiết
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Đặt seed cho tính tái lập
torch.manual_seed(2)

# ===============================
# 1. Chuẩn bị dữ liệu
# ===============================

# Thực hiện các phép biến đổi ảnh
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Tải dataset MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transform, 
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transform, 
                                          download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=64, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=64, 
                                          shuffle=False)

# ===============================
# 2. Xây dựng Model Neural Network
# ===============================

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)     # Hidden layer
        self.fc3 = nn.Linear(64, 10)      # Output layer
    
    def forward(self, x):
        x = x.view(-1, 28*28)             # Flatten ảnh 28x28 thành vector 784
        x = F.relu(self.fc1(x))           # Activation ReLU
        x = torch.sigmoid(self.fc2(x))    # Activation Sigmoid
        x = torch.tanh(self.fc3(x))       # Activation Tanh
        return x

# Khởi tạo model
model = NeuralNet()

# ===============================
# 3. Loss và Optimizer
# ===============================

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ===============================
# 4. Huấn luyện Model
# ===============================

num_epochs = 5
train_loss_list = []

for epoch in range(num_epochs):
    epoch_loss = 0
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    average_loss = epoch_loss / len(train_loader)
    train_loss_list.append(average_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

# ===============================
# 5. Đánh giá Model
# ===============================

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'\nAccuracy of the model on the test images: {accuracy:.2f}%')

# ===============================
# 6. Hiển thị Loss
# ===============================

plt.plot(train_loss_list, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# 7. Hiển thị Một Số Hình Ảnh Mẫu
# ===============================

examples = iter(test_loader)
example_data, example_targets = examples.next()

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i][0], cmap='gray')
    plt.title(f'Label: {example_targets[i]}')
    plt.axis('off')
plt.show()

# ===============================
# 8. Lưu Model
# ===============================

torch.save(model.state_dict(), 'mnist_model.pth')
print("\nModel đã được lưu thành công dưới tên 'mnist_model.pth'.")

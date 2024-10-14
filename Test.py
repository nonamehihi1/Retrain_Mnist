import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# định nghĩa lại model
class SimpleModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(7 * 7 * 64, num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)

        return out


# Định nghĩa các biến chuyển đổi tương tự như khi train
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Tải mô hình đã train
model = torch.load('model/best_model.pth', map_location=torch.device('cpu'))
model.eval()  # Chuyển mô hình sang chế độ đánh giá

# Sử dụng CPU
device = torch.device("cpu")
model.to(device)

def predict_image(image_path):
    # Mở hình ảnh
    image = Image.open(image_path)

    # Áp dụng các biến chuyển đổi
    image = transform(image).unsqueeze(0)  # Thêm một chiều batch
    image = image.to(device)  # Chuyển ảnh sang GPU nếu có

    # Dự đoán
    with torch.no_grad():  # Tắt gradient để tiết kiệm bộ nhớ
        output = model(image)
        _, predicted = output.max(1)  # Lấy chỉ số của lớp có xác suất cao nhất

    return predicted.item()  # Trả về dự đoán


# Sử dụng hàm để dự đoán
image_path = 'Data/augmented/1/0_aug_2.png'  # Thay bằng đường dẫn đến ảnh cần dự đoán

predicted_number = predict_image(image_path)
print(f"Dự đoán: {predicted_number}")
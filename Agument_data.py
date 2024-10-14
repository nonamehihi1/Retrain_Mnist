import os
import random
from PIL import Image
import torchvision.transforms as transforms

def augment_data(input_folder, output_folder, num_augmentations=5):
    # Tạo bộ biến đổi để tăng cường dữ liệu
    augmentation_transforms = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    ])

    # Duyệt qua tất cả các thư mục con
    for digit_folder in os.listdir(input_folder):
        digit_path = os.path.join(input_folder, digit_folder)
        if os.path.isdir(digit_path):
            # Tạo thư mục đầu ra tương ứng
            output_digit_folder = os.path.join(output_folder, digit_folder)
            os.makedirs(output_digit_folder, exist_ok=True)

            # Duyệt qua tất cả các ảnh trong thư mục digit
            for image_file in os.listdir(digit_path):
                image_path = os.path.join(digit_path, image_file)
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Mở ảnh
                    image = Image.open(image_path).convert('RGB')

                    # Lưu ảnh gốc
                    image.save(os.path.join(output_digit_folder, image_file))

                    # Tạo và lưu các phiên bản tăng cường
                    for i in range(num_augmentations):
                        augmented_image = augmentation_transforms(image)
                        augmented_filename = f"{os.path.splitext(image_file)[0]}_aug_{i}{os.path.splitext(image_file)[1]}"
                        augmented_image.save(os.path.join(output_digit_folder, augmented_filename))

    print("Quá trình tăng cường dữ liệu đã hoàn tất.")

# Sử dụng hàm
input_folder = 'Data/output_data_ver2'  # Thay đổi đường dẫn này theo cấu trúc thư mục của bạn
output_folder = 'Data/augmented'  # Thư mục đầu ra cho dữ liệu đã tăng cường
augment_data(input_folder, output_folder, num_augmentations=5)
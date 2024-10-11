import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 이미지 데이터셋 불러오기 (여기서는 예시로 ImageFolder 사용)
data_transform = transforms.Compose([
    transforms.ToTensor()  # 텐서로 변환
])

dataset = datasets.ImageFolder('D:/Project/CNN_project/image', transform=data_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# 평균과 표준 편차 계산
def calculate_mean_std(dataloader):
    mean = 0.
    std = 0.
    total_images_count = 0

    for images, _ in dataloader:
        batch_images_count = images.size(0)  # 배치 크기
        total_images_count += batch_images_count

        # (배치, 채널, 높이, 너비) 형식의 이미지에서 채널별 평균과 표준편차 계산
        mean += images.mean([0, 2, 3]) * batch_images_count
        std += images.std([0, 2, 3]) * batch_images_count

    mean /= total_images_count
    std /= total_images_count

    return mean, std

# 평균과 표준 편차 계산
mean, std = calculate_mean_std(dataloader)
print('Mean:', mean)
print('Std:', std)

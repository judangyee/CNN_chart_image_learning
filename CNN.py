import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import random
import time
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. 주가 차트 이미지를 로드할 디렉토리 설정
rise_directory = 'D:\Project\CNN_project\image\상승'  # 상승 차트 이미지 폴더
decrise_directory = 'D:\Project\CNN_project\image\하락'  # 하락 차트 이미지 폴더

# 2. 차트 이미지 파일 경로 리스트
rise_images_filepaths = sorted([os.path.join(rise_directory, f) for f in os.listdir(rise_directory)])
decrise_images_filepaths = sorted([os.path.join(decrise_directory, f) for f in os.listdir(decrise_directory)])
images_filepaths = [*rise_images_filepaths, *decrise_images_filepaths]

random.seed(42)
random.shuffle(images_filepaths)
train_images_filepaths = images_filepaths[:400]
val_images_filepaths = images_filepaths[400:-10]
test_images_filepaths = images_filepaths[-10:]

print(len(train_images_filepaths), len(val_images_filepaths), len(test_images_filepaths))


# 3. Dataset 클래스 수정
class StockChartDataset(Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)  # 이미지 로드

        # 이미지가 RGBA면 RGB로 변환
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # 상승, 하락 클래스 라벨링 (파일명에 'Rise', 'Decrise'로 라벨 구분)
        label = 1 if 'Rise' in img_path else 0

        img_transformed = self.transform(img, self.phase)
        return img_transformed, label


# 4. 이미지 변환 클래스
class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.ToTensor(),  # 이미지만 텐서로 변환
                transforms.Normalize(mean, std)  # 정규화
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),  # 이미지만 텐서로 변환
                transforms.Normalize(mean, std)  # 정규화
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),  # 이미지만 텐서로 변환
                transforms.Normalize(mean, std)  # 정규화
            ])
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)


# 5. DataLoader 설정
mean = (0.9719, 0.9709, 0.9681)  # 평균
std = (0.1434, 0.1423, 0.1551)  # 표준 편차
batch_size = 32  # 배치 크기

train_dataset = StockChartDataset(train_images_filepaths, transform=ImageTransform(mean, std), phase='train')
val_dataset = StockChartDataset(val_images_filepaths, transform=ImageTransform(mean, std), phase='val')
test_dataset = StockChartDataset(test_images_filepaths, transform=ImageTransform(mean, std), phase='test')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}


# 6. AlexNet 모델 및 훈련 루틴 동일
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model = AlexNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()


# 7. 훈련 루틴
def train_model(model, dataloader_dict, criterion, optimizer, num_epoch):
    since = time.time()
    best_acc = 0.0

    # 손실과 정확도를 기록할 리스트
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            # 기록 저장
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # 손실과 정확도 그래프 그리기
    epochs = range(1, num_epoch + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_history, label='Training loss')
    plt.plot(epochs, val_loss_history, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_history, label='Training accuracy')
    plt.plot(epochs, val_acc_history, label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model


# 8. 모델 훈련
num_epoch = 10
model = train_model(model, dataloader_dict, criterion, optimizer, num_epoch)

# 전체 모델 저장
torch.save(model, 'stock_chart_model_full.pth')
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

# 테스트 데이터셋 평가
evaluate_model(model, test_dataloader)

def visualize_predictions(model, dataloader, num_images=5):
    model.eval()
    images, labels = next(iter(dataloader))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    # 결과 시각화
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i].cpu().permute(1, 2, 0))  # 채널 차원 변경
        plt.title(f'Actual: {labels[i].item()}, Predicted: {preds[i].item()}')
        plt.axis('off')
    plt.show()

# 예측 결과 시각화
visualize_predictions(model, test_dataloader)


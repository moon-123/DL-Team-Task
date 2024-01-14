# DL-Team-Task
# Weather Detection

# 1. 데이터
### 1-1. 데이터 불러오기

> 케글 API를 사용하여 데이터 불러오기
```
import os

os.environ['KAGGLE_USERNAME'] = '본인 케글 이름'
os.environ['KAGGLE_KEY'] = '본인 케글 키'
```

kaggle 데이터셋 불러오는 명령어

```
!kaggle datasets download -d jehanbhathena/weather-dataset
```

불러온 데이터 압출 풀기 

```
!unzip -q weather-dataset.zip
```

### 1-2. train, validation 나누기

```
import os
import random
import shutil

def split_dataset_by_class(dataset_path, train_path, validation_path, validation_ratio=0.2):
    # 클래스 폴더 목록 가져오기
    classes = os.listdir(dataset_path)

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        train_class_path = os.path.join(train_path, class_name)
        validation_class_path = os.path.join(validation_path, class_name)

        # 폴더 생성
        if not os.path.exists(train_class_path):
            os.makedirs(train_class_path)
        if not os.path.exists(validation_class_path):
            os.makedirs(validation_class_path)

        # 클래스 폴더 내의 파일 목록 가져오기
        file_list = os.listdir(class_path)

        # 클래스 별 데이터셋 섞기
        random.shuffle(file_list)

        # 클래스 별 데이터셋을 train과 validation으로 나누기
        num_validation = int(len(file_list) * validation_ratio)
        validation_files = file_list[:num_validation]
        train_files = file_list[num_validation:]

        # validation 폴더로 파일 이동
        for file in validation_files:
            src_path = os.path.join(class_path, file)
            dest_path = os.path.join(validation_class_path, file)
            shutil.move(src_path, dest_path)

        # train 폴더로 파일 이동
        for file in train_files:
            src_path = os.path.join(class_path, file)
            dest_path = os.path.join(train_class_path, file)
            shutil.move(src_path, dest_path)

# 사용 예시
dataset_path = 'dataset'
train_path = 'train'
validation_path = 'validation'

split_dataset_by_class(dataset_path, train_path, validation_path, validation_ratio=0.2)
```

```
train_list_dew = os.listdir('train/dew')
valid_list_dew = os.listdir('validation/dew')

print(len(train_list_dew), len(valid_list_dew))
```
### 1-3. 모듈 설정, dataset 객체 만들기

```
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    'validation': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
}

def target_transforms(target):
    return torch.FloatTensor([target])

image_datasets = {
    'train': datasets.ImageFolder('train', data_transforms['train']),
    'validation': datasets.ImageFolder('validation', data_transforms['validation'])
}

# label을 float형으로 변경함
# label의 차원이 2D임, 각 요소는 1D
# target_transform=target_transforms를 하지 않으면 각 요소는 long타입에 0D임
```

클래스 개수 확인
```
image_datasets['train'].classes, len(image_datasets['train'].classes) # 11개
```

### 1-4. 데이터 로더
```
dataloaders = {
    'train': DataLoader(
        image_datasets['train'],
        batch_size=32,
        shuffle=True
    ),
    'validation': DataLoader(
        image_datasets['validation'],
        batch_size=32,
        shuffle=False
    )
}
```

이미지 시각화
```
imgs, labels = next(iter(dataloaders['train']))
fig, axes = plt.subplots(4, 8, figsize=(20, 10))

for img, label, ax in zip(imgs, labels, axes.flatten()):
    ax.set_title(label.item())
    ax.imshow(img.permute(1, 2, 0))
    ax.axis('off')
```
![image](https://github.com/moon-123/DL-Team-Task/assets/59769304/03cb9e7b-04f0-46cc-bf38-05e5943a2569)

# 2. 학습 

### 2-1. 전이학습에 사용할 모델 선정
* 기존에 사용한 efficientnet_b4 보다 적합한 모델을 찾아야함
* RESNET101을 사용해보겠음
```
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)

WeightsEnum.get_state_dict = get_state_dict

model = resnet101(weights=ResNet101_Weights.DEFAULT)
```

시도한 모델을 포함
```
# FC Layer 수정
for param in model.parameters():
    param.requires_grad = False # 가져온 파라미터 (W, b)를 업데이트하지 않음

# model.fc = nn.Sequential(
#     nn.Linear(2048, 512),
#     nn.ReLU(),
#     nn.Linear(512, 11)
# ).to(device)

# model.fc = nn.Sequential(
#     nn.Linear(2048, 1024),
#     nn.ReLU(),
#     nn.Linear(1024, 512),
#     nn.ReLU(),
#     nn.Linear(512, 11)
# ).to(device)

# model.fc = nn.Sequential(
#     nn.Linear(2048, 1024),
#     nn.ReLU(),
#     nn.Linear(1024, 512),
#     nn.ReLU(),
#     nn.Linear(512, 256),
#     nn.ReLU(),
#     nn.Linear(256, 32),
#     nn.ReLU(),
#     nn.Linear(32, 11)
# ).to(device)

model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 32),
    nn.ReLU(),
    nn.Linear(32, 11)
).to(device)

model = model.to(device)

```

전이학습용 모델 fc 확인
```
imgs, label = next(iter(dataloaders['train']))
# y_pred = model(dataloaders['train'].)
model.fc
```

모델 학습
```
# 학습
# optimizer: Adam
# epochs: 10
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
# optimizer = optim.Adam(model.parameters(), lr=0.001) # fc 파라미터와 다름

epochs = 10

for epoch in range(epochs+1):
    for phase in ['train', 'validation']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        sum_loss = 0
        sum_acc = 0
        # cnt = 0

        length = len(dataloaders[phase])
        for x_batch, y_batch in dataloaders[phase]:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_pred = model(x_batch)
            loss = nn.CrossEntropyLoss()(y_pred, y_batch)

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            sum_loss = sum_loss + loss.item()

            y_prob = nn.Softmax(1)(y_pred)

            y_pred_index = torch.argmax(y_prob, axis=1)

            accuracy = (y_batch == y_pred_index).float().sum() / len(y_batch) * 100
            sum_acc = sum_acc + accuracy

            # print(f'{phase:10s}: running {cnt}/{length} Loss: {loss:.6f} Accuracy: {accuracy:.2f}%')
            # cnt += 1

        avg_loss = sum_loss / len(dataloaders[phase])
        avg_acc = sum_acc / len(dataloaders[phase])

        print(f'{phase:10s}: Epoch {epoch:4d}/{epochs} Loss: {avg_loss:.6f} Accuracy: {avg_acc:.2f}%')
```

가장 성능이 좋게 평가된 모델 저장
```
torch.save(model.state_dict(), 'weather_ResNet101_model.pth')
```

# Resnet101


---


## 1차시도
batch size: 32, shuffle: True, lr:0.001

### Model(FC)
```
Sequential(
  (0): Linear(in_features=2048, out_features=512, bias=True)
  (1): ReLU()
  (2): Linear(in_features=512, out_features=11, bias=True)
)
```
### Result
```
train     : Epoch   10/10 Loss: 0.131252 Accuracy: 95.56%
validation: Epoch   10/10 Loss: 0.466486 Accuracy: 88.61%
```

filename: model1.pth

---


## 2차시도
batch size: 32, shuffle: True, lr:0.001

### Model(FC)
```
Sequential(
  (0): Linear(in_features=2048, out_features=1024, bias=True)
  (1): ReLU()
  (2): Linear(in_features=1024, out_features=512, bias=True)
  (3): ReLU()
  (4): Linear(in_features=512, out_features=11, bias=True)
)
```
### Result
```
train     : Epoch   10/10 Loss: 0.151008 Accuracy: 94.49%
validation: Epoch   10/10 Loss: 0.487187 Accuracy: 87.31%
```
filename: x


---

## 3차시도
batch size: 32, shuffle: True, lr:0.001

### Model(FC)
```
Sequential(
  (0): Linear(in_features=2048, out_features=1024, bias=True)
  (1): ReLU()
  (2): Linear(in_features=1024, out_features=512, bias=True)
  (3): ReLU()
  (4): Linear(in_features=512, out_features=256, bias=True)
  (5): ReLU()
  (6): Linear(in_features=256, out_features=32, bias=True)
  (7): ReLU()
  (8): Linear(in_features=32, out_features=11, bias=True)
)
```
### Result
```
train     : Epoch   10/10 Loss: 0.183164 Accuracy: 94.09%
validation: Epoch   10/10 Loss: 0.469859 Accuracy: 87.23%

```
* Train 정확도와 Validation 정확도의 차이가 많이 나는걸 보아 과적합이 일어났다고 판단
* early stop 혹은 drop out 사용이 필요


filename: model3.pth


---

# Resnet101 - 드롭아웃 추가

## 1차시도 ✓
batch size: 32, shuffle: True, lr:0.001

### Model(FC)
```
Sequential(
  (0): Linear(in_features=2048, out_features=1024, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.5, inplace=False)
  (3): Linear(in_features=1024, out_features=512, bias=True)
  (4): ReLU()
  (5): Dropout(p=0.5, inplace=False)
  (6): Linear(in_features=512, out_features=256, bias=True)
  (7): ReLU()
  (8): Dropout(p=0.5, inplace=False)
  (9): Linear(in_features=256, out_features=32, bias=True)
  (10): ReLU()
  (11): Linear(in_features=32, out_features=11, bias=True)
)
```
### Result
```
train     : Epoch   10/10 Loss: 0.321553 Accuracy: 90.27%
validation: Epoch   10/10 Loss: 0.386997 Accuracy: 89.26%
```
* 드롭아웃을 많이 추가하여 과적합이 잡힌 것 같다.
* 혹시나 과소적합이진 않을까?
* 학습률을 높여서 확인해보자.

filename: model4.pth


---

## 2차시도
batch size: 32, shuffle: True, lr:0.002

### Model(FC)
```
Sequential(
  (0): Linear(in_features=2048, out_features=1024, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.5, inplace=False)
  (3): Linear(in_features=1024, out_features=512, bias=True)
  (4): ReLU()
  (5): Dropout(p=0.5, inplace=False)
  (6): Linear(in_features=512, out_features=256, bias=True)
  (7): ReLU()
  (8): Dropout(p=0.5, inplace=False)
  (9): Linear(in_features=256, out_features=32, bias=True)
  (10): ReLU()
  (11): Linear(in_features=32, out_features=11, bias=True)
)
```
### Result
```
train     : Epoch   10/10 Loss: 0.389684 Accuracy: 88.73%
validation: Epoch   10/10 Loss: 0.476691 Accuracy: 87.11%
```
* 1차시도와의 정확도 차이가 1%로 작지만 train, validation간의 차이는 1차시도보다 줄었음.
* 1차시도가 더 좋은 모델이라고 판단됨!

filename: model5.pth

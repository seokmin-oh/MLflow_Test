# Cifar10 데이터셋 사용
# pytorch기반 간단한 CNN모델 사용
# MLflow 활용, 모델 평가 및 버전 관리 테스트

# python mlflow_test.py 실행 후
# 에폭 다 돌아가고 실행 다 되면
# cmd 창에 mlflow ui 치면 화면 나옵니당

# 모델 버전관리 뿐 아니라 배포도 할 수 있습니다

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# CNN 모델 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 정확도 계산 함수
def calculate_accuracy(net, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    # 실험 설정
    experiment_name = "seokmin"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    # 버전 관리를 위한 run name 설정
    run_name = "version_12"  # 첫 번째 실행일 경우. 두 번째 실행시 "version_2"로 변경

    # 데이터 준비
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

    # MLflow 실험 시작
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"Current Run ID: {run_id}")
        print(f"Current Run Name: {run_name}")

        # 하이퍼파라미터 로깅
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("batch_size", 4)
        mlflow.log_param("epochs", 1)

        # 모델 및 최적화 도구 초기화
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # 학습 루프
        for epoch in range(1):  # ~ 에포크 실행
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    avg_loss = running_loss / 2000
                    mlflow.log_metric("train_loss", avg_loss, step=epoch * len(trainloader) + i)
                    running_loss = 0.0

            # 각 에포크 끝에 테스트 셋에 대한 정확도 계산 및 로깅
            accuracy = calculate_accuracy(net, testloader)
            mlflow.log_metric("test_accuracy", accuracy, step=epoch)
            print(f"Epoch {epoch+1}, Test Accuracy: {accuracy:.2f}%")

        # 최종 정확도 계산 및 로깅
        final_accuracy = calculate_accuracy(net, testloader)
        mlflow.log_metric("final_test_accuracy", final_accuracy)
        
        # 모델 저장
        mlflow.pytorch.log_model(net, "model")

    print("Finished Training")
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")
    print(f"Experiment ID: {experiment_id}")
    print(f"Run ID: {run_id}")
    print(f"Run Name: {run_name}")

if __name__ == '__main__':
    main()
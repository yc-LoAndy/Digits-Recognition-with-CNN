import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

EPOCHS = 20
BATCH_SIZE = 32
LR = 0.0015
DEVICE = torch.device("cuda")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Part identical to that of Encoder in conv_denoiseAE.py
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, device=DEVICE)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, device=DEVICE)
        self.maxPool = nn.MaxPool2d(kernel_size=2, return_indices=False)

        # New structure in Net
        self.fc = nn.Linear(32*4*4, 10, device=DEVICE)
    
    def forward(self, inputs):
        # inputs dimension = 1*28*28
        out = self.conv1(inputs)    # 16*24*24
        out = self.maxPool(out)     # 16*12*12
        out = F.tanh(out)
        out = self.conv2(out)       # 32*8*8
        out = self.maxPool(out)     # 32*4*4
        out = F.tanh(out)

        # Flatten to 1-D as the input of fc layer
        out = torch.flatten(out, start_dim=1)
        out = F.softmax(self.fc(out), dim=1)  # shape = (10, )
        return out

def train(net: nn.Module, train_loader: DataLoader, optimizer, loss_func, scheduler) -> list:
    log_loss = []
    for epoch in range(EPOCHS):
        total_loss = 0
        for data, label in train_loader:
            inputs = data.to(DEVICE)
            label = label.to(DEVICE)
            net.zero_grad()

            # Forward
            predict = net(inputs)
            loss = loss_func(predict, label)
            loss.backward()
            optimizer.step()
            total_loss += loss
            log_loss.append(loss)
        total_loss /= len(train_loader.dataset)
        scheduler.step()

        if epoch % 5 == 0:
            print('[{}/{}] Loss:'.format(epoch+1, EPOCHS), total_loss.item())
    print('[{}/{}] Loss:'.format(epoch+1, EPOCHS), total_loss.item())
    return log_loss

def test(net: nn.Module, test_loader: DataLoader, loss_func) -> None:
    net.eval()
    with torch.no_grad():
        total_loss = 0
        correct = 0
        for data, label in test_loader:
            # Forward
            predict = net(data.to(DEVICE)).to(DEVICE)
            label = label.to(DEVICE)
            total_loss += loss_func(predict, label).item()
            predict = torch.argmax(predict, dim=1)
            correct += len(predict[predict==label])
        print(f"Avg. Test Cross Entropy = {total_loss/len(test_loader.dataset)}")
        print(f"Avg. Test Cross Accuracy = {correct/len(test_loader.dataset)}")

def main() -> None:
    net = Net()
    # pretrained_params = torch.load("/home/yclo/pyproj/practice/scripts/model/encoder_state_dict.pth")
    # net.load_state_dict(pretrained_params, strict=False)

    # Training Phase
    train_loader = DataLoader(
        datasets.MNIST(
            "/home/yclo/pyproj/practice/scripts/MNIST",
            train=True,
            download=False,
            transform=transforms.ToTensor()
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    optimizer = optim.Adam(net.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss().to(DEVICE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=range(0, 30, 5), gamma=0.7)

    log_loss = train(net, train_loader, optimizer, loss_func, scheduler)
    log_loss = torch.tensor(log_loss, device='cpu')
    plt.plot(log_loss)
    plt.savefig("/home/yclo/pyproj/practice/scripts/pic/digit_recog_log_loss.jpg")
    plt.clf()

    # Testing Phase
    test_loader = DataLoader(
        datasets.MNIST(
            "/home/yclo/pyproj/practice/scripts/MNIST",
            train=False, download=False, transform=transforms.ToTensor()
        ),
        batch_size=20
    )
    test(net, test_loader, loss_func)

    # Save the classification model
    torch.save(net.state_dict(), "/home/yclo/pyproj/practice/scripts/model/net_state_dict.pth")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
import matplotlib.pyplot as plt

EPOCHS = 30
BATCH_SIZE = 32
LR = 0.0025
NOISE_FACTOR = 0.3
DEVICE = torch.device("cuda")

# Model Structure
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, device=DEVICE)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, device=DEVICE)
        self.maxPool = nn.MaxPool2d(kernel_size=2, return_indices=False)

    def forward(self, inputs):
        # inputs dimension = 1*28*28
        out = self.conv1(inputs)    # 16*24*24
        out = self.maxPool(out)     # 16*12*12
        out = F.tanh(out)
        out = self.conv2(out)       # 32*8*8
        out = self.maxPool(out)     # 32*4*4
        out = F.tanh(out)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.transConv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=6, stride=2, device=DEVICE)
        self.transConv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=6, stride=2, device=DEVICE)

    def forward(self, inputs):
        # inputs dimension = 32*4*4
        out = F.tanh(self.transConv1(inputs))  # 16*12*12
        out = self.transConv2(out)  # 1*28*28
        out = F.sigmoid(out)
        return out

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)
        return codes, decoded

def train(model: AutoEncoder, train_loader: DataLoader) -> list:
    log_loss = []
    for epoch in range(EPOCHS):
        total_loss = 0
        for data, _ in train_loader:
            inputs = data.to("cpu")
            noisy_inputs = inputs + (NOISE_FACTOR*torch.normal(0, 1, inputs.shape))
            noisy_inputs = noisy_inputs.clip(0, 1).to(DEVICE)
            inputs = inputs.to(DEVICE)
            model.zero_grad()

            # Forward
            codes, decoded = model(noisy_inputs)
            loss = loss_func(decoded, inputs)
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

def main() -> None:
    train_loader = DataLoader(
        datasets.MNIST(
            "/home/yclo/pyproj/practice/scripts/MNIST",
            train=True,
            download=True,
            transform=transforms.ToTensor()
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    global loss_func, optimizer, scheduler, loss_func
    model_ae = AutoEncoder().to(DEVICE)
    optimizer = optim.Adam(model_ae.parameters(), lr=LR)
    loss_func = nn.MSELoss().to(DEVICE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=range(5, 50, 5), gamma=0.8)

    log_loss = train(model_ae, train_loader)
    log_loss = torch.tensor(log_loss, device='cpu')
    plt.plot(log_loss)
    plt.savefig("/home/yclo/pyproj/practice/scripts/pic/log_loss.jpg")
    plt.clf()

    # Save the whole model as denoise_ae.pth
    torch.save(model_ae, "/home/yclo/pyproj/practice/scripts/model/denoise_ae.pth")
    # Save the encoder part of the model as encoder.pth
    torch.save(model_ae.encoder.state_dict(), "/home/yclo/pyproj/practice/scripts/model/encoder_state_dict.pth")

def reconstruct():
    # Settings
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # Show images
    def show_images(images, title):
        plt.title(title)
        # sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
        for index, image in enumerate(images):
            # plt.subplot(sqrtn, sqrtn, index+1)
            plt.subplot(4, 5, index+1)
            plt.imshow(image.reshape(28, 28))
            plt.axis('off')
        plt.savefig(f"/home/yclo/pyproj/practice/scripts/pic/{title}.jpg")
        plt.clf()
        plt.cla()

    # Load model
    model_ae = torch.load("/home/yclo/pyproj/practice/scripts/model/denoise_ae.pth")
    model_ae.eval()

    # DataLoader
    test_loader = DataLoader(
        datasets.MNIST(
            "/home/yclo/pyproj/practice/scripts/MNIST",
            train=False, download=False, transform=transforms.ToTensor()
        ),
        batch_size=20
    )
    # Test
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            inputs = data
            show_images(inputs, f'Original_images_{i+1}')
            # Forward
            codes, outputs = model_ae(inputs.to(DEVICE))
            outputs = outputs.detach().cpu()
            show_images(outputs, f'Restructured_Image_by_AE_{i+1}')
            print(f"Test MSE = {loss_func(outputs, inputs).item()}")
            if i>=3:
                break

if __name__ == "__main__":
    main()
    reconstruct()

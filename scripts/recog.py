import os

import torch
import torchvision.io as tio
from train_clf import Net
from img_proc import ImgProcessor

INPUTSZ = 28
PROJPATH = "/home/yclo/pyproj/practice"
DEVICE = torch.device("cuda")

imgproc = ImgProcessor(
        imgpath = f"{PROJPATH}/scripts/pic/digits/original_digits.jpg",
        crop = None
    )
imgproc.convert(outdir=f"{PROJPATH}/scripts/pic/digits/", outsize=INPUTSZ)

i = 0
digits = torch.tensor([])
while True:
    impath = f"{PROJPATH}/scripts/pic/digits/digit_{i}.jpg"
    if not os.path.isfile(impath):
        break
    d = tio.read_image(f"{PROJPATH}/scripts/pic/digits/digit_{i}.jpg")
    digits = torch.cat((digits, d))
    i += 1
ntest = i

labels = []
with open(f"{PROJPATH}/scripts/pic/digits/label.txt", "r") as f:
    for line in f:
        labels.append(int(line.strip('\n')))
labels = torch.tensor(labels)

net = Net()
net.load_state_dict(torch.load(f"{PROJPATH}/scripts/model/net_state_dict.pth"))
net.eval()

with torch.no_grad():
    correct = 0
    for i, digit in enumerate(digits):
        digit = digit.to(DEVICE).reshape((1, 1, INPUTSZ, INPUTSZ))
        label = labels[i].to(DEVICE)
        output = net(digit).to(DEVICE)
        predict = torch.argmax(output, dim=1)
        print(f"predicit: {predict.item()}\tlabel: {label}")
        if predict.item() == label:
            correct += 1
    print(f"Average accuracy: {correct / ntest}")

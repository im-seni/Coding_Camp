import torch
import torch.nn as nn
import torchvision

class vanillaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.cv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.cv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.cv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.cv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout()
        self.head = nn.Linear(in_features=9216, out_features=20)
    
    def forward(self, x):
        out = self.cv1(x)
        out = self.relu(out)
        out = self.pool1(out)

        out = self.cv2(out)
        out = self.relu(out)
        out = self.pool2(out)

        out = self.cv3(out)
        out = self.relu(out)

        out = self.cv4(out)
        out = self.relu(out)

        out = self.cv5(out)
        out = self.relu(out)
        out = self.pool3(out)

        out = self.dropout(out)
        out = out.view(out.size(0), -1) #flatten
        out = self.head(out)
        return out

class vanillaCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.cv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.cv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.cv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.cv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout()
        
        self.head = nn.Sequential(
            nn.Linear(9216, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 20)
        )

    def forward(self, x):
        out = self.cv1(x)
        out = self.relu(out)
        out = self.pool1(out)

        out = self.cv2(out)
        out = self.relu(out)
        out = self.pool2(out)

        out = self.cv3(out)
        out = self.relu(out)

        out = self.cv4(out)
        out = self.relu(out)

        out = self.cv5(out)
        out = self.relu(out)
        out = self.pool3(out)

        out = self.dropout(out)
        out = out.view(out.size(0), -1) #flatten
        out = self.head(out)

        return out
    
class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        
        print("loading Imagenet pretrained VGG19")
        self.vgg = torchvision.models.vgg19(weights='IMAGENET1K_V1', progress=True)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=20)
        )
        # replace classifier of pretrained VGG-19 with self defined classifier
        setattr(self.vgg, 'classifier', self.classifier)
    
    def forward(self, x):
        return self.vgg(x)


import torch
import torch.nn as nn

class SimpleVGG(nn.Module):
    """
    A simple VGG-style CNN for CIFAR-10 classification.
    
    Architecture:
    - Block 1: Conv(32) -> Conv(64) -> MaxPool
    - Block 2: Conv(128) -> Conv(128) -> MaxPool
    - Block 3: Conv(256) -> Conv(256) -> MaxPool
    - Global Average Pooling
    - Classifier: Linear(256 -> 128) -> Dropout -> Linear(128 -> 10)
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(SimpleVGG, self).__init__()
        
        # Block 1: 3 -> 32 -> 64
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 64 x 16 x 16
        )
        
        # Block 2: 64 -> 128 -> 128
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 128 x 8 x 8
        )
        
        # Block 3: 128 -> 256 -> 256
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 256 x 4 x 4
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # Output: 256 x 1 x 1
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # Test with a dummy input to verify shapes
    model = SimpleVGG()
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

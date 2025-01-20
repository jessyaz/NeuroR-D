import torch
import torch.nn as nn
import torch.nn.functional as F

class EGGS_1(nn.Module):
    def __init__(self, num_classes):
        super(EGGS_1, self).__init__()
        
        # Première couche de convolution
        self.conv1 = nn.Sequential(
            nn.Conv1d(8, 64, kernel_size=7, stride=2, padding=3),  # 8 électrodes, 64 filtres
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Couches suivantes
        self.conv2 = self._make_layer(64, 128, 2)
        self.conv3 = self._make_layer(128, 256, 2, stride=2)
        self.conv4 = self._make_layer(256, 512, 2, stride=2)

        # Couche fully connected pour la classification finale
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        layers.append(nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        ))

        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # On permute pour que la dimension batch soit à l'avant si nécessaire
        x = x.permute(0, 2, 1)    # Transformation si nécessaire

        # Passage dans les couches convolutives
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Pooling adaptatif pour réduire la dimension (finalisation du passage)
        x = F.adaptive_avg_pool1d(x, 1)

        # Flatten la sortie pour la passer dans la couche fully connected
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

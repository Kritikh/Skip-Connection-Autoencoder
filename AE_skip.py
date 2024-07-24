import torch 
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels, skip_features):
        super(Encoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, skip_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(skip_features),
            nn.ReLU(inplace = True),
            nn.Conv2d(skip_features, skip_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(skip_features),
            nn.ReLU(inplace = True)
            )
        
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, skip_features):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, skip_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(skip_features),
            nn.ReLU(inplace = True),
            nn.Conv2d(skip_features,skip_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(skip_features),
            nn.ReLU(inplace = True)
            )
        
    def forward(self, x):
        return self.decoder(x)

class Skip_connect(nn.Module):
    def __init__(self, in_channels = 3, skip_features = 32):
        super(Skip_connect, self).__init__()
        self.encoder1 = Encoder(in_channels, skip_features)
        self.encoder2 = Encoder(skip_features, skip_features*2)
        self.encoder3 = Encoder(skip_features*2, skip_features*4)
        self.decoder1 = Decoder(skip_features*4+skip_features*2, skip_features*2)
        self.decoder2 = Decoder(skip_features*2+skip_features, skip_features)
        self.decoder3 = Decoder(skip_features, in_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
    
    def forward(self,x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))

        dec1 = self.upsample(enc3)
        #print(dec1.shape)
        #print(enc2.shape)
        dec1 = torch.cat((dec1, enc2), dim=1)
        #print(dec1.shape)
        dec1 = self.decoder1(dec1)

        dec2 = self.upsample(dec1)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec3 = self.decoder3(dec2)
        
        return dec3
                
                
                
                
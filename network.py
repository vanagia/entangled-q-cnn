import torch
import torch.nn as nn

from layers import Spinrep, Conv_layer, Prodpool, Classif


class QCNN(nn.Module):
    """Main cnn class."""

    def __init__(self, max_frequency=1, legendre=False, biasf=False, device='cpu'):
        super(QCNN, self).__init__()

        self.device = device
        self.biasf = biasf
        if legendre:
            self.modes = max_frequency
            self.r = 2 * self.modes
        else:
            self.modes = 2 * max_frequency
            self.r = self.modes

        self.batchmean = [None] * 10
        self.batchvar = [None] * 10
            
        # Representation layer
        self.rep = Spinrep(self.modes, self.device, legendre=legendre, flatt_LR=False)

        # 1st hidden layer
        self.norm1 = nn.BatchNorm1d(self.modes)
        self.hid1 = Conv_layer(self.modes, 2 * self.r, self.device, biasf=self.biasf)

        self.pool = Prodpool()

        # 2nd hidden layer
        self.norm2 = nn.BatchNorm1d(2 * self.r)
        self.hid2 = Conv_layer(2 * self.r, 3 * self.r, self.device, biasf=self.biasf)
        
        # 3rd hidden layer
        self.norm3 = nn.BatchNorm1d(3 * self.r)
        self.hid3 = Conv_layer(3 * self.r, 4 * self.r, self.device, biasf=self.biasf)
        
        # 4th hidden layer
        self.norm4 = nn.BatchNorm1d(4 * self.r)
        self.hid4 = Conv_layer(4 * self.r, 5 * self.r, self.device, biasf=self.biasf)
        
        # 5th hidden layer
        self.norm5 = nn.BatchNorm1d(5 * self.r)
        self.hid5 = Conv_layer(5 * self.r, 6 * self.r, self.device, biasf=self.biasf)
        
        # 6th hidden layer
        self.norm6 = nn.BatchNorm1d(6 * self.r)
        self.hid6 = Conv_layer(6 * self.r, 7 * self.r, self.device, biasf=self.biasf)
        
        # 7th hidden layer
        self.norm7 = nn.BatchNorm1d(7 * self.r)
        self.hid7 = Conv_layer(7 * self.r, 8 * self.r,  self.device, biasf=self.biasf)
        
        # 8th hidden layer
        self.norm8 = nn.BatchNorm1d(8 * self.r)
        self.hid8 = Conv_layer(8 * self.r, 9 * self.r, self.device, biasf=self.biasf)

        # Classification layer
        self.norm9 = nn.BatchNorm1d(9 * self.r)
        self.classifier = Classif(9 * self.r, 10, self.device, biasf=self.biasf)

    def forward(self, input: torch.Tensor):
        """Forward pass."""

        features, x = self.rep(input)

        #self.batchmean[1] = torch.mean(features, (0,2)).detach()## NOT NEEDED FOR EVALMODE EE
        #self.batchvar[1] = torch.var(features, (0,2)).detach()## NOT NEEDED FOR EVALMODE EE
        features = self.norm1(features)
        features = self.hid1(features)
        features = self.pool(features)

        #self.batchmean[1] = torch.mean(features, (0,2)).detach()## NOT NEEDED FOR EVALMODE EE
        #self.batchvar[1] = torch.var(features, (0,2)).detach()## NOT NEEDED FOR EVALMODE EE
        features = self.norm2(features)
        features = self.hid2(features)
        features = self.pool(features)

        #self.batchmean[2] = torch.mean(features, (0,2)).detach()## NOT NEEDED FOR EVALMODE EE
        #self.batchvar[2] = torch.var(features, (0,2)).detach()## NOT NEEDED FOR EVALMODE EE
        features = self.norm3(features)
        features = self.hid3(features)
        features = self.pool(features)

        #self.batchmean[3] = torch.mean(features, (0,2)).detach()## NOT NEEDED FOR EVALMODE EE
        #self.batchvar[3] = torch.var(features, (0,2)).detach()## NOT NEEDED FOR EVALMODE EE
        features = self.norm4(features)
        features = self.hid4(features)
        features = self.pool(features)
      
        #self.batchmean[4] = torch.mean(features, (0,2)).detach()## NOT NEEDED FOR EVALMODE EE
        #self.batchvar[4] = torch.var(features, (0,2)).detach()## NOT NEEDED FOR EVALMODE EE
        features = self.norm5(features)
        features = self.hid5(features)
        features = self.pool(features)

        #self.batchmean[5] = torch.mean(features, (0,2)).detach()## NOT NEEDED FOR EVALMODE EE
        #self.batchvar[5] = torch.var(features, (0,2)).detach()## NOT NEEDED FOR EVALMODE EE
        features = self.norm6(features)
        features = self.hid6(features)
        features = self.pool(features)

        #self.batchmean[6] = torch.mean(features, (0,2)).detach()## NOT NEEDED FOR EVALMODE EE
        #self.batchvar[6] = torch.var(features, (0,2)).detach()## NOT NEEDED FOR EVALMODE EE
        features = self.norm7(features)
        features = self.hid7(features)
        features = self.pool(features)

        #self.batchmean[7] = torch.mean(features, (0,2)).detach()## NOT NEEDED FOR EVALMODE EE
        #self.batchvar[7] = torch.var(features, (0,2)).detach()## NOT NEEDED FOR EVALMODE EE
        features = self.norm8(features)
        features = self.hid8(features)
        features = self.pool(features)

        #self.batchmean[8] = torch.mean(features, (0,2)).detach()## NOT NEEDED FOR EVALMODE EE
        #self.batchvar[8] = torch.var(features, 0).view(-1).detach()## NOT NEEDED FOR EVALMODE EE
        features = self.norm9(features)
        features = features.view(-1, 9 * self.r)
        features = self.classifier(features)
      
        return features
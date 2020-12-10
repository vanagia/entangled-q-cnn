import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Spinrep(nn.Module):
    """Representation layer."""

    def __init__(self, dim, device, legendre=False, flatt_LR=False):
        super(Spinrep, self).__init__()

        self.dim = dim
        self.device = device
        self.legendre = legendre
        self.flatt_LR = flatt_LR

    def forward(self, input:torch.Tensor):
        """Takes in shape (b,c,p1,p2), outputs (b,c,p=p1*p2)."""

        # flattening for left/right entanglement entropy
        if self.flatt_LR: 
            x = torch.rot90(torch.flip(input, (3,)), 1, (2,3)).reshape(input.shape[0], input.shape[1], -1)
            x = input.view(input.shape[0], input.shape[1], -1)
        # flattening for up/down entanglement entropy
        else:
            x = input.reshape(input.shape[0], input.shape[1], -1)

        output = torch.empty(x.shape[0], self.dim, x.shape[2], device=self.device)

        if self.legendre:
            assert self.dim <= 19
            freq = int(self.dim)
            for n in range(freq):
                if n == 0:
                    output[:, n, :] = (3./2.)**(1./2.) * x[:, 0, :]
                if n == 1:
                    output[:, n, :] = (5./2.)**(1./2.) * (1./2.) * (x[:, 0, :]**2 - 1.)
                if n == 2:
                    output[:, n, :] = (7./2.)**(1./2.) * (1./2.) * (5. * x[:, 0, :]**3 - 3. * x[:, 0, :])
                if n == 3:
                    output[:, n, :] = (9./2.)**(1./2.) * (1./8.) * (35. * x[:, 0, :]**4 - 30. * x[:, 0, :]**2 + 3.)
                if n == 4:
                    output[:, n, :] = (11./2.)**(1./2.) * (1./8.) * (63. * x[:, 0, :]**5 - 70. * x[:, 0, :]**3 + 15. * x[:, 0, :])
                if n == 5:
                    output[:, n, :] = (13./2.)**(1./2.) * (1./16.) * (231. * x[:, 0, :]**6 - 315. * x[:, 0, :]**4 + 105. * x[:, 0, :]**2 - 5.)
                if n == 6:
                    output[:, n, :] = (15./2.)**(1./2.) * (1./16.) * (429. * x[:, 0, :]**7 - 693. * x[:, 0, :]**5 + 315. * x[:, 0, :]**3 - 35. * x[:, 0, :])
                if n == 7:
                    output[:, n, :] = (17./2.)**(1./2.) * (1./128.) * (6435. * x[:, 0, :]**8 - 12012. * x[:, 0, :]**6 + 6930. * x[:, 0, :]**4 - 1260. * x[:, 0, :]**2 + 35.)
                if n == 8:
                    output[:, n, :] = (19./2.)**(1./2.) * (1./128.) * (12155. * x[:, 0, :]**9 - 25740. * x[:, 0, :]**7 + 18018. * x[:, 0, :]**5 - 4620. * x[:, 0, :]**3 + 315. * x[:, 0, :])
                if n == 9:
                    output[:, n, :] = (21./2.)**(1./2.) * (1./256.) * (46189. * x[:, 0, :]**10 - 109395. * x[:, 0, :]**8 + 90090. * x[:, 0, :]**6 - 30030. * x[:, 0, :]**4 + 3465. * x[:, 0, :]**2 - 63.)
                if n == 10:
                    output[:, n, :] = (23./2.)**(1./2.) * (1./256.) * (88179. * x[:, 0, :]**10 - 230945. * x[:, 0, :]**8 + 218790. * x[:, 0, :]**6 - 90090. * x[:, 0, :]**4 + 15015. * x[:, 0, :]**2 - 693.)
                if n == 11:
                    output[:, n, :] = (25./2.)**(1./2.) * (1./1024.) * (676039. * x[:, 0, :]**12 - 1939938. * x[:, 0, :]**10 + 2078505. * x[:, 0, :]**8 - 1021020. * x[:, 0, :]**6 + 225225. * x[:, 0, :]**4 - 18018. * x[:, 0, :]**2 + 231.)
                if n == 12:
                    output[:, n, :] = (27./2.)**(1./2.) * (1./1024.) * (1300075. * x[:, 0, :]**12 - 4056234. * x[:, 0, :]**10 + 4849845. * x[:, 0, :]**8 - 2771340. * x[:, 0, :]**6 + 765765. * x[:, 0, :]**4 - 90090. * x[:, 0, :]**2 + 3003.)
                if n == 13:
                    output[:, n, :] = (29./2.)**(1./2.) * (1./2048.) * (5014575. * x[:, 0, :]**14 - 16900975. * x[:, 0, :]**12 + 22309287. * x[:, 0, :]**10 - 14549535. * x[:, 0, :]**8 + 4849845. * x[:, 0, :]**6 - 765765. * x[:, 0, :]**4 + 45045. * x[:, 0, :]**2 - 429.)
                if n == 14:
                    output[:, n, :] = (31./2.)**(1./2.) * (1./2048.) * (9694845. * x[:, 0, :]**14 - 35102025. * x[:, 0, :]**12 + 50702925. * x[:, 0, :]**10 - 37182145. * x[:, 0, :]**8 + 14549535. * x[:, 0, :]**6 - 2909907. * x[:, 0, :]**4 + 255255. * x[:, 0, :]**2 - 6435.)
                if n == 15:
                    output[:, n, :] = (33./2.)**(1./2.) * (1./32768.) * (300540195. * x[:, 0, :]**16 - 1163381400. * x[:, 0, :]**14 + 1825305300. * x[:, 0, :]**12 - 1487285800. * x[:, 0, :]**10 + 669278610. * x[:, 0, :]**8 - 162954792. * x[:, 0, :]**6 + 19399980. * x[:, 0, :]**4 - 875160. * x[:, 0, :]**2 + 6435.)
                if n == 16:
                    output[:, n, :] = (35./2.)**(1./2.) * (1./32768.) * (583401555. * x[:, 0, :]**16 - 2404321560. * x[:, 0, :]**14 + 4071832900. * x[:, 0, :]**12 - 3650610600. * x[:, 0, :]**10 + 1859107250. * x[:, 0, :]**8 - 535422888. * x[:, 0, :]**6 + 81477396. * x[:, 0, :]**4 - 5542680. * x[:, 0, :]**2 + 109395.)
                if n == 17:
                    output[:, n, :] = (37./2.)**(1./2.) * (1./65536.) * (2268783825. * x[:, 0, :]**18 - 9917826435. * x[:, 0, :]**16 + 18032411700. * x[:, 0, :]**14 - 17644617900. * x[:, 0, :]**12 + 10039179150. * x[:, 0, :]**10 - 3346393050. * x[:, 0, :]**8 + 624660036. * x[:, 0, :]**6 - 58198140. * x[:, 0, :]**4 + 2078505. * x[:, 0, :]**2 - 12155.)
                if n == 18:
                    output[:, n, :] = (39./2.)**(1./2.) * (1./65536.) * (4418157975. * x[:, 0, :]**18 - 20419054425. * x[:, 0, :]**16 + 39671305740. * x[:, 0, :]**14 - 42075627300. * x[:, 0, :]**12 + 26466926850. * x[:, 0, :]**10 - 10039179150. * x[:, 0, :]**8 + 2230928700. * x[:, 0, :]**6 - 267711444. * x[:, 0, :]**4 + 14549535. * x[:, 0, :]**2 - 230945.)
        else:
            assert self.dim % 2 == 0
            freq = int(self.dim / 2)
            for n in range(freq):
                output[:, 2*n, :] = torch.cos(2. * math.pi * (n+1) * x[:, 0, :])
                output[:, 2*n+1, :] = torch.sin(2. * math.pi * (n+1) * x[:, 0, :])

        return output, x


class Conv_layer(nn.Module):
    """Main convolutional layer."""

    def __init__(self, in_fields, out_fields, device, biasf=False):
        super(Conv_layer, self).__init__()

        self.in_fields = in_fields
        self.out_fields = out_fields
        self.device = device
        self.biasf = biasf

        weights = torch.empty(out_fields, in_fields, 1).to(self.device)
        self.register_parameter("weights", nn.Parameter(weights, requires_grad = True))
        if self.biasf:
            bias = torch.empty(out_fields).to(self.device)
            self.register_parameter("bias", nn.Parameter(bias, requires_grad = True))

        self._init_weights()

    def _init_weights(self):
        """Weight initialization."""
        torch.nn.init.xavier_uniform_(self.weights, gain=1.0).to(self.device)
        if self.biasf:
            self.bias.data = torch.zeros_like(self.bias).to(self.device)

    def forward(self, input: torch.Tensor):
        """Forward pass."""
       
        if self.biasf:
            filter, bias = self._build_filter()
            output = F.conv1d(input, filter, bias=bias)
        else:
            filter = self._build_filter()
            output = F.conv1d(input, filter)

        return output


    def _build_filter(self):
        """Not essential, but allows for possible customization."""

        filter = torch.empty(self.out_fields, self.in_fields, 1).to(self.device)
        filter = self.weights

        if self.biasf:
            bias = torch.empty(self.out_fields).to(self.device)
            bias = self.bias

            return filter, bias
        else:
            return filter


class Prodpool(nn.Module):
    """Product pooling layer."""

    def __init__(self):
        super(Prodpool, self).__init__()

    def forward(self, input: torch.Tensor, view = False):
        """Forward pass."""

        assert input.shape[2] % 2 == 0

        even = input[:, :, ::2]
        odd = input[:, :, 1::2]

        output = torch.einsum('kli,kli->kli', even, odd)

        return output


class Classif(nn.Module):
    """Classification layer."""

    def __init__(self, in_fields, out_fields, device, biasf=False):

        super(Classif, self).__init__()

        self.in_fields = in_fields
        self.out_fields = out_fields
        self.device = device
        self.biasf = biasf

        weights = torch.empty(out_fields, in_fields).to(self.device)
        self.register_parameter("weights", nn.Parameter(weights, requires_grad = True))

        if self.biasf:
            bias = torch.empty(out_fields).to(self.device)#.double()
            self.register_parameter("bias", nn.Parameter(bias, requires_grad = True))

        self._init_weights()


    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.weights, gain = 1.0).to(self.device)
        if self.biasf:
            self.bias.data = torch.zeros_like(self.bias).to(self.device)

    def forward(self, input: torch.Tensor, view = False):
        if self.biasf:
            filter, bias = self._build_filter()
            output = F.linear(input, filter, bias = bias)
        else:
            filter = self._build_filter()
            output = F.linear(input, filter)

        return output

    def _build_filter(self):
        filter = torch.empty(self.out_fields, self.in_fields).to(self.device)#.double()
        filter = self.weights

        if self.biasf:
            bias = torch.empty(self.out_fields).to(self.device)
            bias = self.bias

            return filter, bias
        else:
            return filter

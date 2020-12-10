import torch
import torchvision
import torchvision.transforms as transforms


class torchMNIST():
    """MNIST dataset loader."""
    
    classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')

    def __init__(self, batchsize=10, legendre=False, download=False):
        if legendre:
            self.transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # range [-1.0, 1.0]
        else:
            self.transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()]) # range [0.0, 1.0]

        self.trainset = torchvision.datasets.MNIST(root='data', train=True, download=download, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batchsize, shuffle=True, num_workers=0)

        self.testset = torchvision.datasets.MNIST(root='data', train=False, download=download, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batchsize, shuffle=False, num_workers=0)


class FMNIST():
    """FMNIST dataset loader."""

    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

    def __init__(self, batchsize=10, legendre=False, download=False):
        if legendre:
            self.transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # range [-1.0, 1.0]
        else:
            self.transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()]) # range [0.0, 1.0]

        self.trainset = torchvision.datasets.FashionMNIST(root='data', train=True, download=download, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batchsize, shuffle=True, num_workers=0)

        self.testset = torchvision.datasets.FashionMNIST(root='data', train=False, download=download, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batchsize, shuffle=False, num_workers=0)

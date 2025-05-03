from torchvision.datasets import FashionMNIST
import torchvision
import torch

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()
                                            , torchvision.transforms.Lambda(lambda x: torch.flatten(x))

])

trainset_fashion = FashionMNIST(root="data", train = True, download= True, transform= transform)
train_subset, val_subset = torch.utils.data.random_split(trainset_fashion, [50000, 10000], generator=torch.Generator().manual_seed(1))
testset_fashion = FashionMNIST(root = "data", train = False, download= True, transform=transform) 

batch_size = 32
trainloader_fashion = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
validationloader_fashion = torch.utils.data.DataLoader(val_subset, batch_size = batch_size, shuffle=True)
testloader_fashion = torch.utils.data.DataLoader(testset_fashion, batch_size=batch_size, shuffle=True)
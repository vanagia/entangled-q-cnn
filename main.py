import torch
import time
import numpy as np
import random

from network import QCNN
import loader
from training import Training


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1234)
    random.seed(1234)
    torch.set_printoptions(precision = 10)

    start = time.time()

    net = QCNN(max_frequency=9, legendre=False, biasf=False, device=device).to(device)
    net.double()

    trainer = Training(net, loader.torchMNIST(batchsize=50), n_epochs=90, stats=True)
    #trainer = Training(net, loader.FMNIST(batchsize=50), n_epochs=90, stats=True)

    EE, cost, train_acc, test_acc, test_cost = trainer.train_loop(valid=True)

    elapsed = time.time() - start
    print('(%.1fs)\n' % elapsed)

    trainer.test(valid=True)

    torch.save(net.state_dict(), "net.pt")

    EE = np.asarray(EE)
    cost = np.asarray(cost)
    train_acc = np.asarray(train_acc)
    test_acc = np.asarray(test_acc)
    test_cost = np.asarray(test_cost)
    np.save("plotdata_EE.npy", EE)
    np.save("plotdata_cost.npy", cost)
    np.save("plotdata_train_acc.npy", train_acc)
    np.save("plotdata_test_acc.npy", test_acc)
    np.save("plotdata_test_cost.npy", test_cost)
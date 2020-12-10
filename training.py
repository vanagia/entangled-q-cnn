import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

import loader
import torchTN


def specialized_loss(output, target):
    """Computes square distance loss."""
    
    label = torch.zeros(output.shape[0], output.shape[1], device=output.device)
    for i in range(output.shape[0]):
        label[i, int(target[i].data)] = 1.

    loss = torch.sum((output - label)**2, 1)
    loss = torch.mean(loss, 0)

    return loss


class Training():
    """Class for training the network."""

    def __init__(self, net, loader, stats=False, n_epochs=1):
        self.n_epochs = n_epochs
        self.net = net
        self.loader = loader
        self.stats = stats

        self.loss_fn = specialized_loss
        self.optimizer = optim.AdamW(self.net.parameters(), lr=1e-2, weight_decay=0.01)
        self.device = next(self.net.parameters()).device

    def train_step(self, input=torch.empty(1), label=torch.empty(1)):
        """Performs a single training step and returns the corresponding loss."""

        self.optimizer.zero_grad()
        self.net.train()
        
        output = self.net(input)
        loss = self.loss_fn(output, label)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_loop(self, valid=False):
        """Performs the entire training."""

        # Keeps track of loss during training
        running_loss = 0.0

        # Lists to return
        EE = []
        cost = []
        train_acc = []
        test_acc = []
        test_cost = []

        print("\nTraining on %s images has started... (%d epochs, batch size %d)\n" % (len(self.loader.trainset), self.n_epochs, self.loader.trainloader.batch_size))

        # Main epoch loop with mini-batches and dataloader from dataset
        for epoch in range(self.n_epochs):
            # Decrease learning rate every 10 epochs
            if (epoch + 1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.5 * param_group['lr']
    
            print("\nTraining in epoch %d/%d has started!" % (epoch + 1, self.n_epochs))
            for i, data in tqdm(enumerate(self.loader.trainloader, 1), ncols=50, total=int(len(self.loader.trainset)/self.loader.trainloader.batch_size), miniters=1):
                input_batch = data[0].to(self.device).double()
                label_batch = data[1].to(self.device)

                loss = self.train_step(input_batch, label_batch)
                
                # Print loss stats at 10 intervals
                print_interval = int(len(self.loader.trainset)/(self.loader.trainloader.batch_size*10))
                running_loss += loss
                if i % print_interval == 0:
                    tqdm.write("Average loss: %.4f" % (running_loss / print_interval))
                    # Keep track of various stats
                    if self.stats:
                        EE.append(torchTN.EEhalf(self.net))
                        cost.append(running_loss / print_interval)
                    running_loss = 0.0

            if valid:
                train_acc_dat = self.test(valid=False)
                test_acc_dat, test_cost_dat = self.test(valid=True)
                
                if self.stats:
                    train_acc.append(train_acc_dat)
                    test_acc.append(test_acc_dat)
                    test_cost.append(test_cost_dat)

        print("Training complete!\n")

        return EE, cost, train_acc, test_acc, test_cost


    def test(self, valid=True):
        """Test network on either the test or train dataset."""

        acc = []
        cost = []

        if valid:
            current_set = self.loader.testset
            load = self.loader.testloader
            print("\nTesting on %s images has started..." % len(current_set))
        else:
            current_set = self.loader.trainset
            load = self.loader.trainloader
            print("\nTesting on %s images has started..." % len(current_set))

        self.net.eval()

        total = 0
        class_total = list(0. for i in range(len(self.loader.classes)))
        correct = 0
        class_correct = list(0. for i in range(len(self.loader.classes)))
        tempcost = 0

        with torch.no_grad():
            for data in tqdm(load, ncols=50, total=int(len(current_set)/load.batch_size), miniters=1):
                images = data[0].to(self.device).double()
                labels = data[1].to(self.device)

                outputs = self.net(images)
                tempcost += specialized_loss(outputs, labels)

                ff, predicted = torch.max(torch.abs(outputs), 1)
                c = (predicted == labels)

                # Total score
                correct += c.sum().item()
                total += labels.size(0)

                # Individual class scores
                for i in range(labels.shape[0]):
                    label = labels[i]
                    class_correct[label] += c[i].item() # either 0 or 1
                    class_total[label] += 1

        print("Testing complete!\n")
        print("class_correct", class_correct)
        print("class total", class_total)

        for i in range(len(self.loader.classes)):
            print('Accuracy of %s: %d%%' % (self.loader.classes[i], 100 * class_correct[i] / class_total[i]))
        print('\nOverall accuracy of the network: %.2f%%\n' % (100 * correct / total))

        if self.stats:
            acc.append(correct / total)
            if valid:
                toappend = tempcost / (len(current_set) / load.batch_size)
                cost.append(toappend.cpu().numpy())

        # return numpy arrays
        if valid:
            return np.asarray(acc), np.asarray(cost)
        else:
            return np.asarray(acc)
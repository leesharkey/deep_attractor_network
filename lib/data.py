import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import lib.utils

# Define some handy helper functions. They're unpythonic, I know.
listdetach = lambda y : [x.detach() for x in y]
listtodevice = lambda y, device: [x.to(device) for x in y]
listsplit = lambda y, size, dim: [torch.split(x, size, dim) for x in y]
liststack = lambda y, dim : [torch.stack(x, dim) for x in y]
listcat = lambda y, dim : [torch.cat(x, dim) for x in y]

class Dataset():
    def __init__(self, args, train_set=True, shuffle=True):
        if args.dataset == 'CIFAR10':
            self.dataset = datasets.CIFAR10('./data',
                                            download=train_set,
                                            transform=transforms.ToTensor())
        elif args.dataset == 'MNIST':
            self.dataset = datasets.MNIST('./data',
                                            download=train_set,
                                            transform=transforms.ToTensor())

        self.dataset = self.dataset
        self.loader = DataLoader(self.dataset,
                                 batch_size=args.batch_size,
                                 shuffle=shuffle,
                                 drop_last=True
                                 )

class SampleBuffer:
    def __init__(self, args, device, max_samples=10000):
        self.args = args
        self.max_samples = max_samples
        self.num_neg_samples = int(self.args.sample_buffer_prob * \
                               self.args.batch_size)
        self.num_rand_samples = self.args.batch_size - \
                                self.num_neg_samples
        self.device = device
        self.neg_buffer = []
        self.max_p_neg_samples = 10000

    def push(self, states, pos=False):

        states = listtodevice(listdetach(states), 'cpu')
        neg_and_rand_states = [state for state in states]
        zippee_neg_and_rand = neg_and_rand_states
        zippee_neg_and_rand = listsplit(zippee_neg_and_rand, size=1, dim=0)
        for sample_and_class_id in zip(*zippee_neg_and_rand):
            sample = sample_and_class_id
            self.neg_buffer.append((listdetach(sample)))
            if len(self.neg_buffer) > self.max_samples:
                self.neg_buffer.pop(0)

    def get(self):
        neg_items  = random.choices(self.neg_buffer,
                                     k=self.num_neg_samples)
        neg_samples  = zip(*neg_items)  # Unzips
        # Combines each of N lists of 1 state layers (of len=k) into k lists
        # of len=N
        neg_samples = listcat(neg_samples, dim=0)
        neg_samples = listtodevice(neg_samples, self.device)
        samples = neg_samples

        return samples

    def sample_buffer(self, initter_network=None):

        if len(self.neg_buffer) < 1:
            # Generate all rand states and class_ids when buffers are empty
            rand_states = lib.utils.generate_random_states(self.args,
                                                           self.args.state_sizes,
                                                           self.device,
                                                           initter_network)
            return rand_states

        replay_sample = self.get()
        new_rand_state_sizes = [s.copy() for s in self.args.state_sizes]
        for size in new_rand_state_sizes:
            size[0] = self.num_rand_samples
        random_sample = lib.utils.generate_random_states(self.args,
                                                          new_rand_state_sizes,
                                                          self.device,
                                                          initter_network)
        return listcat(zip(replay_sample, random_sample), 0)

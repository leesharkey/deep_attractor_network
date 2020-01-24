import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
import lib.utils



class Dataset():
    def __init__(self, args):
        if args.dataset == 'CIFAR10':
            self.dataset = datasets.CIFAR10('./data',
                                            download=True,
                                            transform=transforms.ToTensor())
        elif args.dataset == 'MNIST':
            self.dataset = datasets.MNIST('./data',
                                            download=True,
                                            transform=transforms.ToTensor())

        self.dataset = self.dataset
        self.loader = DataLoader(self.dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 drop_last=True
                                 )
                            #num_workers=4) #this makes debugging very very slow
        self.loader = tqdm(enumerate(sample_data(self.loader)))

class SampleBuffer:
    def __init__(self, args, batch_size, p, device, max_samples=10000):
        self.args = args
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.p = p
        self.device = device
        self.buffer = []
        if self.args.cd_mixture:
            self.pos_buffer = []
            self.max_pos_samples = 1000

    def __len__(self):
        return len(self.buffer)

    def push(self, states, class_ids=None, pos=False):

        if pos:
            buffer = self.pos_buffer
            max_samples = self.max_pos_samples
        else:
            buffer = self.buffer
            max_samples = self.max_samples

        states = listtodevice(listdetach(states), 'cpu')
        class_ids = class_ids.detach().to('cpu')
        zippee = states + [class_ids]
        zippee = listsplit(zippee, size=1, dim=0)

        for sample_and_class_id in zip(*zippee):
            sample, class_id = sample_and_class_id[:-1],sample_and_class_id[-1]
            buffer.append((listdetach(sample), class_id))
            if len(buffer) > max_samples:
                buffer.pop(0)

    def get(self, n_samples, device='cuda'):
        if self.args.cd_mixture and len(self.buffer)>1000:
            n_samples_pos = (torch.rand(n_samples) < self.args.pos_buffer_frac
                             ).sum()
            n_samples_neg = int(n_samples - n_samples_pos)
            neg_items = random.sample(self.buffer, k=n_samples_neg) #TODO change to pop without replacement
            neg_samples, neg_class_ids = zip(*neg_items)  # Unzips
            neg_samples = zip(*neg_samples)
            neg_samples = listcat(neg_samples, dim=0)
            neg_class_ids = torch.tensor(neg_class_ids)
            if n_samples_pos > 0:
                pos_items = random.sample(self.pos_buffer, k=int(n_samples_pos)) #TODO change to pop without replacement
                pos_samples, pos_class_ids = zip(*pos_items)  # Unzips
                pos_samples = zip(*pos_samples)
                pos_samples = listcat(pos_samples, dim=0)
                pos_class_ids = torch.tensor(pos_class_ids)
                samples = listcat(zip(pos_samples, neg_samples), dim=0)
                class_ids = torch.cat([pos_class_ids, neg_class_ids])
                samples = listtodevice(samples, device)
                class_ids = class_ids.to(device)
            else:
                samples = listtodevice(neg_samples, device)
                class_ids = neg_class_ids.to(device)
        else:
            items = random.choices(self.buffer, k=n_samples)
            samples, class_ids = zip(*items)  # Unzips

            # Combines each of N lists of 1 state layers (of len=k) into k lists
            # of len=N
            samples = zip(*samples)
            samples = listcat(samples, dim=0)
            class_ids = torch.tensor(class_ids)
            samples = listtodevice(samples, device)
            class_ids = class_ids.to(device)
        return samples, class_ids

    def sample_buffer(self):

        state_sizes = self.args.state_sizes


        if len(self.buffer) < 1:
            rand_states = lib.utils.generate_random_states(self.args.state_sizes, self.device)
            return (rand_states, #TODO check these work okay with convolutions
                torch.randint(0, 10, (self.batch_size,), device=self.device),
            )

        n_replay = (np.random.rand(self.batch_size) < self.p).sum()

        replay_sample, replay_id = self.get(n_replay)
        new_rand_state_sizes = self.args.state_sizes
        new_rand_state_sizes = [[size[0] - n_replay, size[1], size[2], size[3]]
                                for size in new_rand_state_sizes]
        random_sample = lib.utils.generate_random_states(new_rand_state_sizes, self.device)
        random_id = torch.randint(0, 10, (self.batch_size - n_replay,), device=self.device)
        #TODO a func that does rand inits for arbitrary state_size args.
        return (
            listcat(zip(replay_sample, random_sample), 0),
            torch.cat([replay_id, random_id], 0),
        )


def sample_data(loader):
    loader_iter = iter(loader)

    while True:
        try:
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)

            yield next(loader_iter)

listdetach = lambda y : [x.detach() for x in y]
listtodevice = lambda y, device: [x.to(device) for x in y]
listsplit = lambda y, size, dim: [torch.split(x, size, dim) for x in y]
liststack = lambda y, dim : [torch.stack(x, dim) for x in y]
listcat = lambda y, dim : [torch.cat(x, dim) for x in y]





shapes = lambda x : [y.shape for y in x]
nancheck = lambda x : (x != x).any()
listout = lambda y : [x for x in y]
gradcheck = lambda  y : [x.requires_grad for x in y]
leafcheck = lambda  y : [x.is_leaf for x in y]
existgradcheck = lambda  y : [(x.grad is not None) for x in y]
existgraddatacheck = lambda  y : [(x.grad.data is not None) for x in y]
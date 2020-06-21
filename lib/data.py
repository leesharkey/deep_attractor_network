import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
import lib.utils



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
                            #num_workers=4) #this makes debugging very very slow
        #self.loader = tqdm(enumerate(sample_data(self.loader)))

class SampleBuffer: #TODO init that uses biases
    def __init__(self, args, device, max_samples=10000):
        self.args = args
        self.max_samples = max_samples
        self.num_p_neg_samples = int(self.args.batch_size * \
                                self.args.pos_buffer_frac)
        self.num_neg_samples = int(self.args.sample_buffer_prob * \
                               (self.args.batch_size -
                                self.num_p_neg_samples))
        self.num_rand_samples = self.args.batch_size - \
                                self.num_neg_samples - \
                                self.num_p_neg_samples
        self.device = device
        self.neg_buffer = []
        self.p_neg_buffer = []  # for 'positive negative buffer'
        self.max_p_neg_samples = 10000

    def push(self, states, class_ids=None, pos=False):

        if pos:
            # During positive phases, we only extend p_neg_buffer
            states = listtodevice(listdetach(states), 'cpu')
            class_ids = class_ids.detach().to('cpu')
            zippee = states + [class_ids]
            zippee = listsplit(zippee, size=1, dim=0)

            for sample_and_class_id in zip(*zippee):
                sample, class_id = sample_and_class_id[:-1], \
                                   sample_and_class_id[-1]
                self.p_neg_buffer.append((listdetach(sample), class_id))
                if len(self.p_neg_buffer) > self.max_p_neg_samples:
                    self.p_neg_buffer.pop(0)
        else:
            states = listtodevice(listdetach(states), 'cpu')
            class_ids = class_ids.detach().to('cpu')

            neg_and_rand_class_ids = class_ids[self.num_p_neg_samples:]
            neg_and_rand_states = [state[self.num_p_neg_samples:]
                                   for state in states]
            zippee_neg_and_rand = neg_and_rand_states + [neg_and_rand_class_ids]
            zippee_neg_and_rand = listsplit(zippee_neg_and_rand, size=1, dim=0)
            for sample_and_class_id in zip(*zippee_neg_and_rand):
                sample, class_id = sample_and_class_id[:-1],\
                                   sample_and_class_id[-1]
                self.neg_buffer.append((listdetach(sample), class_id))
                if len(self.neg_buffer) > self.max_samples:
                    self.neg_buffer.pop(0)

            if self.args.cd_mixture and self.args.pos_buffer_frac > 0.0:
                p_neg_states =        [state[:self.num_p_neg_samples]
                                       for state in states]
                p_neg_class_ids =        class_ids[:self.num_p_neg_samples]
                zippee_p_neg = p_neg_states + [p_neg_class_ids]
                zippee_p_neg = listsplit(zippee_p_neg, size=1, dim=0)
                for sample_and_class_id in zip(*zippee_p_neg):
                    sample, class_id = sample_and_class_id[:-1],\
                                       sample_and_class_id[-1]
                    self.p_neg_buffer.append((listdetach(sample), class_id))
                    if len(self.p_neg_buffer) > self.max_p_neg_samples:
                        self.p_neg_buffer.pop(0)

    def get(self):
        neg_items   = random.choices(self.neg_buffer,
                                     k=self.num_neg_samples)
        neg_samples,   neg_class_ids   = zip(*neg_items)  # Unzips
        # Combines each of N lists of 1 state layers (of len=k) into k lists
        # of len=N
        neg_samples = zip(*neg_samples)
        neg_samples = listcat(neg_samples, dim=0)
        neg_class_ids = torch.tensor(neg_class_ids)
        neg_samples = listtodevice(neg_samples, self.device)
        neg_class_ids = neg_class_ids.to(self.device)

        if self.args.cd_mixture and self.args.pos_buffer_frac > 0.0:
            p_neg_items = random.choices(self.p_neg_buffer,
                                         k=self.num_p_neg_samples)
            p_neg_samples, p_neg_class_ids = zip(*p_neg_items)  # Unzips
            # Combines each of N lists of 1 state layers (of len=k) into k lists
            # of len=N
            p_neg_samples = zip(*p_neg_samples)
            p_neg_samples = listcat(p_neg_samples, dim=0)
            if self.args.shuffle_pos_frac > 0.0:
                shuffle_num = round(self.num_p_neg_samples * \
                                     self.args.shuffle_pos_frac)
                # Splits the imgs from the pos neg buffer
                shuff_imgs     = p_neg_samples[0][:shuffle_num]
                non_shuff_imgs = p_neg_samples[0][shuffle_num:]

                # Shuffles a fraction of the images
                rand_inds = torch.randperm(shuffle_num)
                shuff_imgs = shuff_imgs[rand_inds,:,:,:]
                new_imgs = torch.cat([shuff_imgs, non_shuff_imgs])
                p_neg_samples[0] = new_imgs


            p_neg_class_ids = torch.tensor(p_neg_class_ids)
            p_neg_samples = listtodevice(p_neg_samples, self.device)
            p_neg_class_ids = p_neg_class_ids.to(self.device)

            samples = listcat(zip(p_neg_samples, neg_samples), dim=0)
            class_ids = torch.cat((p_neg_class_ids, neg_class_ids), dim=0)
        else:
            samples, class_ids = neg_samples, neg_class_ids

        return samples, class_ids

    def sample_buffer(self, initter_network=None):

        state_sizes = self.args.state_sizes

        if len(self.neg_buffer) < 1:
            # Generate all rand states and class_ids when buffers are empty
            rand_states = lib.utils.generate_random_states(self.args,
                                                           self.args.state_sizes,
                                                           self.device,
                                                           self.args.state_scales,
                                                           initter_network)
            return (rand_states,
                torch.randint(0, 10, (self.args.batch_size,),
                              device=self.device),
            )

        replay_sample, replay_id = self.get()
        new_rand_state_sizes = [s.copy() for s in self.args.state_sizes]
        for size in new_rand_state_sizes:
            size[0] = self.num_rand_samples
        random_sample = lib.utils.generate_random_states(self.args,
                                                          new_rand_state_sizes,
                                                          self.device,
                                                          self.args.state_scales,
                                                          initter_network)
        random_id = torch.randint(0, 10, (self.num_rand_samples,), device=self.device)
        return (
            listcat(zip(replay_sample, random_sample), 0),
            torch.cat([replay_id, random_id], 0),
        )


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
import torch
from torchvision import datasets, transforms
import random
from torchvision.datasets import ImageFolder
import numpy as np
from pdb import set_trace as st
from .rotation import RotateImageFolder
from .MyTinyImageNet import TinyImageNet as T
import os

def get_imagenet(train, split, batch_size, image_size):
    if train:

        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    save_path = '/home/data/preprocessed_imagenet'

    if not os.path.exists(save_path):
        # print('create cache>>>')
        dataset = ImageFolder(root='/home/data/imagenettrain', transform=transform)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        torch.manual_seed(20200221)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        os.makedirs(save_path, exist_ok=True)

        # train_dataset = torch.load(os.path.join(save_path, 'train_dataset.pth'))
        # test_dataset = torch.load(os.path.join(save_path, 'test_dataset.pth'))

        if split == '0':
            rand_allclass = np.random.RandomState(seed=20230621).permutation(len(dataset.classes)).tolist()
        elif split == '1':
            rand_allclass = np.random.RandomState(seed=20230631).permutation(len(dataset.classes)).tolist()
        elif split == '2':
            rand_allclass = np.random.RandomState(seed=20230641).permutation(len(dataset.classes)).tolist()

        knownclass = rand_allclass[:100]
        unknownclass = rand_allclass[100:]

        if train:

            kmask = [i for i, e in enumerate(train_dataset) if e[1] in knownclass]
            unkmask = [i for i, e in enumerate(train_dataset) if e[1] in unknownclass]

            random.shuffle(kmask)
            validationportion = int(0.1 * len(kmask))

            kmask_rand_val = kmask[:validationportion]
            kmask_rand_train = kmask[validationportion:]

            known_set_train = torch.utils.data.Subset(train_dataset, kmask_rand_train)
            known_set_val = torch.utils.data.Subset(train_dataset, kmask_rand_val)

            known_set_train = RotateImageFolder(known_set_train)

            torch.save(known_set_train, os.path.join(save_path, 'known_set_train.pth'))
            torch.save(known_set_val, os.path.join(save_path, 'known_set_val.pth'))
            torch.save(knownclass, os.path.join(save_path, 'knownclass.pth'))

            known_data_loader_train = torch.utils.data.DataLoader(
                dataset=known_set_train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2)

            known_data_loader_val = torch.utils.data.DataLoader(
                dataset=known_set_val,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2)
            
            return known_data_loader_train, known_data_loader_val, knownclass

        else:

            kmask = [i for i, e in enumerate(test_dataset) if e[1] in knownclass]
            unkmask = [i for i, e in enumerate(test_dataset) if e[1] in unknownclass]

            known_set = torch.utils.data.Subset(test_dataset, kmask)
            unknown_set = torch.utils.data.Subset(test_dataset, unkmask)

            torch.save(known_set, os.path.join(save_path, 'known_set.pth'))
            torch.save(unknown_set, os.path.join(save_path, 'unknown_set.pth'))   

            known_data_loader = torch.utils.data.DataLoader(
                dataset=known_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2)
            unknown_data_loader = torch.utils.data.DataLoader(
                dataset=unknown_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2)
            
            return known_data_loader, unknown_data_loader, knownclass
    else:
        # print('load cache>>>')
        knownclass = torch.load(os.path.join(save_path, 'knownclass.pth'))


        if train:

            known_set_train = torch.load(os.path.join(save_path, 'known_set_train.pth'))
            known_set_val = torch.load(os.path.join(save_path, 'known_set_val.pth'))

            known_data_loader_train = torch.utils.data.DataLoader(
                dataset=known_set_train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2)

            known_data_loader_val = torch.utils.data.DataLoader(
                dataset=known_set_val,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2)

            return known_data_loader_train, known_data_loader_val, knownclass
        
        else:

            # known_set = torch.load(os.path.join(save_path, 'known_set.pth'))
            # unknown_set = torch.load(os.path.join(save_path, 'unknown_set.pth'))
            test_dataset = ImageFolder(root='/home/data/imagenetval/imagenetval/', transform=transform)
            if split == '0':
                rand_allclass = np.random.RandomState(seed=20230621).permutation(len(test_dataset.classes)).tolist()
            elif split == '1':
                rand_allclass = np.random.RandomState(seed=20230631).permutation(len(test_dataset.classes)).tolist()
            elif split == '2':
                rand_allclass = np.random.RandomState(seed=20230641).permutation(len(test_dataset.classes)).tolist()
            knownclass = rand_allclass[:100]
            unknownclass = rand_allclass[100:]
            kmask = [i for i, e in enumerate(test_dataset) if e[1] in knownclass]
            unkmask = [i for i, e in enumerate(test_dataset) if e[1] in unknownclass]
            known_set = torch.utils.data.Subset(test_dataset, kmask) 
            unknown_set = torch.utils.data.Subset(test_dataset, unkmask)

            known_data_loader = torch.utils.data.DataLoader(
                dataset=known_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0)
            unknown_data_loader = torch.utils.data.DataLoader(
                dataset=unknown_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0)

            return known_data_loader, unknown_data_loader, knownclass
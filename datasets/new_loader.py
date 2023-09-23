import os.path

import torch

from .dataloader import get_data_loader


def My_loader(args, adv_path):
    split = args.split
    kdataloader_trn, _, knownclass = get_data_loader(name=args.datasetname, train=True, split=args.split,
                                                     batch_size=args.batchsize, image_size=args.imgsize)

    if split == '0':
        knownclass = [0, 1, 2, 4, 5, 9]
        unknownclass = [3, 6, 7, 8]
    elif split == '1':
        knownclass = [0, 3, 5, 7, 8, 9]
        unknownclass = [1, 2, 4, 6]
    elif split == '2':
        knownclass = [0, 1, 5, 6, 7, 8]
        unknownclass = [2, 3, 4, 9]
    elif split == '3':
        knownclass = [3, 4, 5, 7, 8, 9]
        unknownclass = [0, 1, 2, 6]
    elif split == '4':
        knownclass = [0, 1, 2, 3, 7, 8]
        unknownclass = [4, 5, 6, 9]
    elif split == '10':
        knownclass = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        unknownclass = []

    loaded_data = torch.load(os.path.join(adv_path, 'adversarial_samples.pt'))
    loaded_samples = loaded_data['samples']
    loaded_labels = loaded_data['labels']
    adversarial_dataset = torch.utils.data.TensorDataset(loaded_samples, loaded_labels)
    kmask = [i for i, e in enumerate(adversarial_dataset) if e[1] in knownclass]
    unkmask = [i for i, e in enumerate(adversarial_dataset) if e[1] in unknownclass]

    known_set = torch.utils.data.Subset(adversarial_dataset, kmask)
    unknown_set = torch.utils.data.Subset(adversarial_dataset, unkmask)

    known_data_loader = torch.utils.data.DataLoader(
        dataset=known_set,
        batch_size=args.batchsize,
        shuffle=False)

    unknown_data_loader = torch.utils.data.DataLoader(
        dataset=unknown_set,
        batch_size=args.batchsize,
        shuffle=False)

    return kdataloader_trn, known_data_loader, unknown_data_loader, knownclass

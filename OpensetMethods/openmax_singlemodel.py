import datetime
import sys

sys.path.append('../')

import os
import os.path as osp
from collections import OrderedDict
import torchvision.utils as vutils
import torch
import torch.optim as optim
from torch import nn
from misc.utils import mkdir, init_model, lab_conv
from torch.nn import DataParallel
import numpy as np
import h5py
import torch.nn.functional as F
import libmr
from models import *
from sklearn.metrics import f1_score
from pdb import set_trace as st

WEIBULL_TAIL_SIZE = 20


def openmax(args, kdataloader_trn, kdataloader_tst, ukdataloader_tst, knownclass, model, testPy=False, test_black=False):
    model.eval()
    device = torch.device(f"cuda:{args.idx}" if torch.cuda.is_available() else "cpu")
    activation_vectors, mean_activation_vectors, weibulls = precalc_weibull(args, kdataloader_trn, knownclass, model,
                                                                            device)

    known_acc, known_scores = openset_weibull(args, kdataloader_tst, knownclass, model,
                                              activation_vectors, mean_activation_vectors, weibulls,  test_black, device, mode='closeset')

    unknown_scores = openset_weibull(args, ukdataloader_tst, knownclass, model,
                                     activation_vectors, mean_activation_vectors, weibulls, test_black, device)

    f1 = 0

    auc = plot_roc(known_scores, unknown_scores)

    if testPy == True:  # if run test.py write
        SaveEvaluation(args, known_acc, auc)
    else:
        return np.array(known_acc.cpu()), auc ,f1


def precalc_weibull(args, dataloader_train, knownclass, model, device):
    # First generate pre-softmax 'activation vectors' for all training examples
    print("Weibull: computing features for all correctly-classified training data")
    activation_vectors = {}

    if args.adv == 'PGD' or args.adv=='adv_clean':
        from advertorch.attacks import PGDAttack
        adversary = PGDAttack(predict1=model, predict2=None, nb_iter=args.adv_iter)
    elif args.adv == 'FGSM':
        from advertorch.attacks import FGSM
        adversary = FGSM(predict1=model, predict2=None)

    # print("**********Conduct Attack**********")
    for _, (images, labels, _, _) in enumerate(dataloader_train):
    # for _, (images, labels) in enumerate(dataloader):

        labels = lab_conv(knownclass, labels)

        images, labels = images.to(device), labels.long().to(device)
        if args.adv == 'clean':
            advimg = images
        else:
            advimg = adversary.perturb(images, labels)
        with torch.no_grad():
            logits = model(advimg)

        correctly_labeled = (logits.data.max(1)[1] == labels)
        labels_np = labels.cpu().numpy()
        logits_np = logits.data.cpu().numpy()
        for i, label in enumerate(labels_np):
            if not correctly_labeled[i]:
                continue
            # If correctly labeled, add this to the list of activation_vectors for this class
            if label not in activation_vectors:
                activation_vectors[label] = []
            activation_vectors[label].append(logits_np[i])
    print("Computed activation_vectors for {} known classes".format(len(activation_vectors)))

    # Compute a mean activation vector for each class
    print("Weibull computing mean activation vectors...")
    mean_activation_vectors = {}
    for class_idx in activation_vectors:
        mean_activation_vectors[class_idx] = np.array(activation_vectors[class_idx]).mean(axis=0)

    # Initialize one libMR Wiebull object for each class
    print("Fitting Weibull to distance distribution of each class")
    weibulls = {}
    for class_idx in activation_vectors:
        distances = []
        mav = mean_activation_vectors[class_idx]
        for v in activation_vectors[class_idx]:
            distances.append(np.linalg.norm(v - mav))
        mr = libmr.MR()
        tail_size = min(len(distances), WEIBULL_TAIL_SIZE)
        mr.fit_high(distances, tail_size)
        weibulls[class_idx] = mr
        # print("Weibull params for class {}: {}".format(class_idx, mr.get_params()))

    return activation_vectors, mean_activation_vectors, weibulls


def openset_weibull(args, dataloader_test, knownclass, model, activation_vectors, mean_activation_vectors,
                    weibulls, test_black, device, mode='openset'):
    # Apply Weibull score to every logit
    weibull_scores = []
    logits = []
    classes = activation_vectors.keys()


    running_corrects = 0.0

    epoch_size = 0.0

    if args.adv == 'PGD' or args.adv=='adv_clean':
        from advertorch.attacks import PGDAttack
        adversary = PGDAttack(predict1=model, predict2=None, nb_iter=args.adv_iter)
    elif args.adv == 'FGSM':
        from advertorch.attacks import FGSM
        adversary = FGSM(predict1=model, predict2=None)

    # reclosslist = []
    # print("**********Conduct Attack**********")
    for steps, (images, labels) in enumerate(dataloader_test):

        labels = lab_conv(knownclass, labels)
        images, labels = images.to(device), labels.long().to(device)
        # print("Calculate weibull_scores in step {}/{}".format(steps, len(dataloader_test)))
        if args.adv == 'clean':
            advimg = images
        else:        
            if test_black == False:
                if mode == 'closeset':
                    advimg = adversary.perturb(images, labels)
                else:
                    advimg = adversary.perturb(images)
            else:
                advimg = images

        with torch.no_grad():
            batch_logits_torch = model(advimg)

        batch_logits = batch_logits_torch.data.cpu().numpy()
        batch_weibull = np.zeros(shape=batch_logits.shape)

        for activation_vector in batch_logits:
            weibull_row = np.ones(len(knownclass))
            for class_idx in classes:
                mav = mean_activation_vectors[class_idx]
                dist = np.linalg.norm(activation_vector - mav)
                weibull_row[class_idx] = 1 - weibulls[class_idx].w_score(dist)
            weibull_scores.append(weibull_row)
            logits.append(activation_vector)

        if mode == 'closeset':
            _, preds = torch.max(batch_logits_torch, 1)  # torch.max(a,1) 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
            # statistics
            running_corrects += torch.sum(preds == labels.data)
            epoch_size += images.size(0)

    if mode == 'closeset':
        running_corrects = running_corrects.double() / epoch_size
        print('Test Acc: {:.4f}'.format(running_corrects))

    weibull_scores = np.array(weibull_scores)
    logits = np.array(logits)
    openmax_scores = -np.log(np.sum(np.exp(logits * weibull_scores), axis=1))

    if mode == 'closeset':
        return running_corrects, np.array(openmax_scores)
    else:
        return np.array(openmax_scores)


def plot_roc(known_scores, unknown_scores):
    from sklearn.metrics import roc_curve, roc_auc_score
    y_true = np.array([0] * len(known_scores) + [1] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    # fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)

    print('AUC {:.03f}'.format(auc_score))

    return auc_score


def SaveEvaluation(args, known_acc, auc):
    sub_path = '{}-{}-{}-{}-{}'.format(args.datasetname, args.adv, args.split,
                                       str(args.adv_iter), args.training_type)
    # filefolder = osp.join('results', 'Test', 'accuracy', args.datasetname+'-'+args.split)
    filefolder = osp.join('results', sub_path, 'test')
    mkdir(filefolder)

    filepath = osp.join(filefolder, 'result.txt')

    output_file = open(filepath, 'a')
    output_file.write('\nClose-set Accuracy: ' + str(np.array(known_acc.cpu())))
    output_file.write('\nOpen-set AUROC: ' + str(auc))
    output_file.write('\nTestseed: ' + str(args.seed))
    output_file.write('\n' + str(datetime.datetime.now()) + '\n' + '----------------------------\n')
    output_file.close()

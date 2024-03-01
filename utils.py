import numpy as np
import random
from scipy.stats import entropy
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from cifar10models import *
import torchattacks
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
from utils_svhn import svhnDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import copy

funSoft = nn.Softmax(dim=1)
criterion = nn.CrossEntropyLoss()


def clamp(X, lower, upper):
    return torch.max(torch.min(X, upper), lower)


def get_loaders(dir_, batch_size, dataName):
    if dataName == "mnist":
        data_train = datasets.MNIST(dir_, train=True, download=True, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()]))
        train_loader = torch.utils.data.DataLoader(
            dataset=data_train,
            batch_size=batch_size,
            shuffle=True)
        data_test = datasets.MNIST(dir_, train=False, download=True, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()]))
        test_split = datasets.MNIST(dir_, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()]))
        test_split.data = data_test.data[5000:]
        test_split.targets = data_test.targets[5000:]
        test_sampler = SubsetRandomSampler(range(5000, len(data_test)))
        val_sampler = SubsetRandomSampler(range(5000))
        test_loader = torch.utils.data.DataLoader(
            dataset=data_test,
            batch_size=batch_size,
            sampler=test_sampler,
            shuffle=False)
        val_loader = torch.utils.data.DataLoader(
            dataset=data_test,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False)
    elif dataName == "svhn":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        data_train = svhnDataset(split='train', transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            dataset=data_train,
            batch_size=batch_size,
            shuffle=True)
        data_test = svhnDataset(split='test', transform=transforms.Compose([transforms.ToTensor()]))
        test_split = svhnDataset(transform=transforms.Compose([transforms.ToTensor()]))
        test_split.data = data_test.data[5000:]
        test_split.labels = data_test.labels[5000:]
        test_sampler = SubsetRandomSampler(range(5000, len(data_test)))
        val_sampler = SubsetRandomSampler(range(5000))
        test_loader = torch.utils.data.DataLoader(
            dataset=data_test,
            batch_size=batch_size,
            sampler=test_sampler,
            shuffle=False)
        val_loader = torch.utils.data.DataLoader(
            dataset=data_test,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False)
    elif dataName == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        num_workers = 2
        data_train = datasets.CIFAR10(
            dir_, train=True, transform=train_transform, download=True)
        data_test = datasets.CIFAR10(
            dir_, train=False, transform=test_transform, download=True)

        test_split = datasets.CIFAR10(dir_, transform=test_transform)
        test_split.data = data_test.data[5000:]
        test_split.targets = data_test.targets[5000:]
        test_sampler = SubsetRandomSampler(range(5000, len(data_test)))
        val_sampler = SubsetRandomSampler(range(5000))
        train_loader = torch.utils.data.DataLoader(
            dataset=data_train,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=data_test,
            batch_size=batch_size,
            sampler=test_sampler,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=data_test,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            sampler=val_sampler,
            num_workers=num_workers,
        )
    elif dataName == "fashion":
        data_train = datasets.FashionMNIST(dir_, train=True, download=True, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()]))
        train_loader = torch.utils.data.DataLoader(
            dataset=data_train,
            batch_size=batch_size,
            shuffle=True)
        data_test = datasets.FashionMNIST(dir_, train=False, download=True, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()]))
        test_split = datasets.FashionMNIST(dir_, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
        test_split.data = data_test.data[5000:]
        test_split.targets = data_test.targets[5000:]
        test_sampler = SubsetRandomSampler(range(5000, len(data_test)))
        val_sampler = SubsetRandomSampler(range(5000))
        test_loader = torch.utils.data.DataLoader(
            dataset=data_test,
            batch_size=batch_size,
            sampler=test_sampler,
            shuffle=False)
        val_loader = torch.utils.data.DataLoader(
            dataset=data_test,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False)
    else:
        print("wrong data name")
        return
    return train_loader, data_train, test_loader, test_split, val_loader


def evaluate_standard(test_loader, model):
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

    return test_acc / n


def evaluate_adv(test_loader, model, attack, dataName):
    if dataName == "mnist" or dataName == "fashion":
        epsilon = 0.3
        alpha = 0.01
        attack_iters = 50
        class_num = 10
    elif dataName == "svhn":
        epsilon = 8 / 255
        alpha = 2 / 255
        attack_iters = 50
        class_num = 10
    else:
        epsilon = 8 / 255
        alpha = 2 / 255
        attack_iters = 50
        class_num = 10

    pgd_acc = 0
    n = 0
    model.eval()
    if attack == "pgd":
        atk = torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=attack_iters, random_start=True)
    elif attack == "vni":
        atk = attack = torchattacks.VMIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2)
    elif attack=="deepfool":
        atk =torchattacks.DeepFool(model, steps=50, overshoot=0.02)
    elif attack=="jitter":
        atk=torchattacks.Jitter(model, eps=8/255, alpha=2/255, steps=10,
         scale=10, std=0.1, random_start=True)
    elif attack == "sparsefool":
        atk=torchattacks.SparseFool(model, steps=10, lam=3, overshoot=0.02)
    elif attack == "pixle":
        atk=torchattacks.Pixle(model, x_dimensions=(0.1, 0.2), restarts=10, max_iterations=50)
    elif attack == "fab":
        atk=torchattacks.FAB(model, norm='Linf', steps=10, eps=8/255, n_restarts=1, alpha_max=0.1, eta=1.05, beta=0.9,seed=0, n_classes=10)
    elif attack == "eaden":
        atk=torchattacks.EADEN(model, kappa=0, lr=0.01, max_iterations=100)
    elif attack == "nifgsm":
        atk=torchattacks.NIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0)
    elif attack == "cw":
        atk=torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01)
    elif attack == "bim":
        atk=torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=10)
    elif attack == "upgd":
        atk=torchattacks.UPGD(model, eps=8/255, alpha=2/255, steps=10, random_start=False)
    elif attack == "spsa":
        atk=torchattacks.SPSA(model, eps=0.3)
    elif attack == "auto":
        atk=torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None)
    else:
        atk = torchattacks.Square(model, eps=epsilon)
    for i, (X, y) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)
        adv_data = atk(X, y)
        with torch.no_grad():
            output_adv = model(adv_data)
            pgd_acc += (output_adv.max(1)[1] == y).sum().item()
            n += y.size(0)

    return pgd_acc / n


def selectLoader(train_dataset, model, num, metric, batch_size, sub_idx, class_num=10, modelName=None, bin_num=None,dataName=None,epsilon=None):
    num = min(num, len(sub_idx))
    full_idx = np.arange(len(train_dataset))
    if metric == "random":
        current_select_idx = random.sample(sub_idx, k=num)
    elif metric == "bald":
        current_select_idx = baldSel(train_dataset, model, num, batch_size, sub_idx, class_num, modelName=modelName)
    elif metric == "dfal":
        current_select_idx = dfalSel(train_dataset, model, num, sub_idx)
    elif metric == "entropy":
        current_select_idx = entropySel(train_dataset, model, num, batch_size, sub_idx)
    elif metric == "entropyDrop":
        current_select_idx = entropyDropSel(train_dataset, model, num, batch_size, sub_idx, class_num, modelName=modelName)
    elif metric == "gini":
        current_select_idx = giniSel(train_dataset, model, num, batch_size, sub_idx)
    elif metric == "kcenter":
        current_select_idx = kcenterSel(train_dataset, model, num, sub_idx, batch_size, class_num)
    elif metric == "lc":
        current_select_idx = leastConfSel(train_dataset, model, num, batch_size, sub_idx)
    elif metric == "margin":
        current_select_idx = marginSel(train_dataset, model, num, batch_size, sub_idx)
    elif metric == "mcp":
        current_select_idx = mcpSel(train_dataset, model, num, batch_size, sub_idx, class_num)
    elif metric == "EGL":
        current_select_idx = eglSel(train_dataset, model, num, batch_size, sub_idx, class_num)
    elif metric == "dre":
        current_select_idx = dreSel(train_dataset, model, num, batch_size, sub_idx, bin_num=bin_num)
    elif metric == "kcenterOpt":
        current_select_idx = kcenterSelOptimized(train_dataset, model, num, sub_idx, batch_size, class_num)
        # For kcenterOpt, you directly get the selected indices to label, so there's no need to calculate `current_remain_index` based on `current_select_idx` not being in `sub_idx`
        current_remain_index = [item for item in sub_idx if item not in current_select_idx]
    else:
        print("wrong metric")
        return

    if metric != "kcenterOpt":  # This check ensures that for kcenterOpt, `current_remain_index` is not overwritten
        current_remain_index = [item for item in sub_idx if item not in current_select_idx]
    
    select_idx = np.delete(full_idx, current_remain_index)
    sub_sampler = SubsetRandomSampler(select_idx)

    sub_train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=sub_sampler)

    return sub_train_loader, list(current_remain_index)
"""
def kcenterSelOptimized(train_dataset, model, num, sub_idx, batch_size, class_num):
    full_idx = np.arange(len(train_dataset))
    labeled_idx = np.setdiff1d(full_idx, sub_idx)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs = torch.zeros((len(train_dataset), class_num)).to(device)
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            current_output = model(x).detach()
            outputs[i * batch_size:i * batch_size + len(x), :] = current_output

    centers = outputs[labeled_idx]
    candidates = outputs[sub_idx]
    select_index = []

    for _ in range(num):
        # Calculate distances using broadcasting, and find the minimum distance for each candidate to any center
        distances = torch.cdist(candidates, centers).min(dim=1)[0]
        # Select the candidate with the maximum minimum distance
        selected_idx = distances.argmax().item()
        select_index.append(sub_idx[selected_idx])
        # Add the selected candidate to centers and remove it from candidates
        centers = torch.cat([centers, candidates[selected_idx:selected_idx+1]], dim=0)
        candidates = torch.cat([candidates[:selected_idx], candidates[selected_idx+1:]], dim=0)
        sub_idx = np.delete(sub_idx, selected_idx)

    return np.array(select_index)
"""
def kcenterSelOptimized(train_dataset, model, num, sub_idx, batch_size, class_num, epsilon=0.01):
    full_idx = np.arange(len(train_dataset))
    labeled_idx = np.setdiff1d(full_idx, sub_idx)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs = torch.zeros((len(train_dataset), class_num)).to(device)
    model.eval()
    
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            # Apply Gaussian noise as an example of input transformation
            x += torch.randn_like(x) * epsilon
            current_output = model(x).detach()
            outputs[i * batch_size:i * batch_size + len(x), :] = current_output

    centers = outputs[labeled_idx]
    candidates = outputs[sub_idx]
    select_index = []

    for _ in range(num):
        distances = torch.cdist(candidates, centers).min(dim=1)[0]
        selected_idx = distances.argmax().item()
        select_index.append(sub_idx[selected_idx])
        centers = torch.cat([centers, candidates[selected_idx:selected_idx+1]], dim=0)
        candidates = torch.cat([candidates[:selected_idx], candidates[selected_idx+1:]], dim=0)
        sub_idx = np.delete(sub_idx, selected_idx)

    return np.array(select_index)
def giniSel(train_dataset, model, num, batch_size, sub_idx):
    sub_train_dataset = Subset(train_dataset, sub_idx)
    sub_train_loader = torch.utils.data.DataLoader(
        dataset=sub_train_dataset,
        batch_size=batch_size,
        shuffle=False)
    scores = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in sub_train_loader:
            x_batch = x_batch.to(device)
            output = model(x_batch)
            output_copy = funSoft(torch.Tensor.cpu(output)).detach().numpy()
            scores = np.concatenate((scores, 1 - np.sum(np.power(output_copy, 2), axis=1)))
    scores_sort = np.argsort(scores)
    scores_gini = [int(item) for item in scores_sort]
    select_loc = scores_gini[-num:]
    return np.array(sub_idx)[np.array(select_loc)]


def marginSel(train_dataset, model, num, batch_size, sub_idx):
    sub_train_dataset = Subset(train_dataset, sub_idx)
    sub_train_loader = torch.utils.data.DataLoader(
        dataset=sub_train_dataset,
        batch_size=batch_size,
        shuffle=False)
    scores = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in sub_train_loader:
            x_batch = x_batch.to(device)
            output = model(x_batch)
            output_copy = funSoft(torch.Tensor.cpu(output)).detach().numpy()
            output_sort = np.sort(output_copy)
            margin_value = output_sort[:, -1] - output_sort[:, -2]
            scores = np.concatenate((scores, margin_value))
    scores_sort = np.argsort(scores)
    scores_margin = [int(item) for item in scores_sort]
    select_loc = scores_margin[:num]
    return np.array(sub_idx)[np.array(select_loc)]


def baldSel(train_dataset, model, num, batch_size, sub_idx, class_num, n_drops=10, modelName=None):
    sub_train_dataset = Subset(train_dataset, sub_idx)
    sub_train_loader = torch.utils.data.DataLoader(
        dataset=sub_train_dataset,
        batch_size=batch_size,
        shuffle=False)
    probs = torch.zeros([n_drops, len(sub_idx), class_num])
    for i in range(n_drops):
        modelDrop = make_drop(modelName, model)
        with torch.no_grad():
            for idx, (x_batch, _) in enumerate(sub_train_loader):
                x_batch = x_batch.to(device)
                output = modelDrop(x_batch)
                output_copy = funSoft(torch.Tensor.cpu(output))
                probs[i][idx * batch_size:(idx + 1) * batch_size] += output_copy
    probs_mean = probs.mean(0)
    entropy1 = (-probs_mean * torch.log(probs_mean)).sum(1)
    entropy2 = (-probs * torch.log(probs)).sum(2).mean(0)
    scores = torch.Tensor.cpu(entropy2 - entropy1).detach().numpy()

    scores_sort = np.argsort(scores)
    scores_lc = [int(item) for item in scores_sort]
    select_loc = scores_lc[:num]
    return np.array(sub_idx)[np.array(select_loc)]


# only when the models include dropout layer
def entropyDropSel(train_dataset, model, num, batch_size, sub_idx, class_num, n_drops=10, modelName=None):
    sub_train_dataset = Subset(train_dataset, sub_idx)
    sub_train_loader = torch.utils.data.DataLoader(
        dataset=sub_train_dataset,
        batch_size=batch_size,
        shuffle=False)
    probs = torch.zeros([len(sub_idx), class_num])
    for i in range(n_drops):
        modelDrop = make_drop(modelName, model)
        with torch.no_grad():
            for idx, (x_batch, _) in enumerate(sub_train_loader):
                x_batch = x_batch.to(device)
                output = modelDrop(x_batch)
                output_copy = funSoft(torch.Tensor.cpu(output))
                probs[idx * batch_size:(idx + 1) * batch_size] += output_copy
    probs = probs / n_drops
    entropyDrop = (probs * torch.log(probs)).sum(1)
    scores = torch.Tensor.cpu(entropyDrop).detach().numpy()

    scores_sort = np.argsort(scores)
    scores_lc = [int(item) for item in scores_sort]
    select_loc = scores_lc[:num]
    return np.array(sub_idx)[np.array(select_loc)]


def leastConfSel(train_dataset, model, num, batch_size, sub_idx):
    sub_train_dataset = Subset(train_dataset, sub_idx)
    sub_train_loader = torch.utils.data.DataLoader(
        dataset=sub_train_dataset,
        batch_size=batch_size,
        shuffle=False)
    scores = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in sub_train_loader:
            x_batch = x_batch.to(device)
            output = model(x_batch)
            output_copy = funSoft(torch.Tensor.cpu(output)).detach().numpy()
            lc_value = np.max(output_copy, axis=1)
            scores = np.concatenate((scores, lc_value))
    scores_sort = np.argsort(scores)
    scores_lc = [int(item) for item in scores_sort]
    select_loc = scores_lc[:num]
    return np.array(sub_idx)[np.array(select_loc)]


def entropySel(train_dataset, model, num, batch_size, sub_idx):
    sub_train_dataset = Subset(train_dataset, sub_idx)
    sub_train_loader = torch.utils.data.DataLoader(
        dataset=sub_train_dataset,
        batch_size=batch_size,
        shuffle=False)
    scores = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in sub_train_loader:
            x_batch = x_batch.to(device)
            output = model(x_batch)
            output_copy = funSoft(torch.Tensor.cpu(output)).detach().numpy()
            entropy_value = entropy(output_copy, base=2, axis=1)
            scores = np.concatenate((scores, entropy_value))
    scores_sort = np.argsort(scores)
    scores_entropy = [int(item) for item in scores_sort]
    select_loc = scores_entropy[-num:]
    return np.array(sub_idx)[np.array(select_loc)]


def eglSel(train_dataset, model, num, batch_size, sub_idx, class_num):
    sub_train_dataset = Subset(train_dataset, sub_idx)
    sub_train_loader = torch.utils.data.DataLoader(
        dataset=sub_train_dataset,
        batch_size=batch_size,
        shuffle=False)
    scores = []
    model.eval()
    for x_batch, y_batch in sub_train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch.requires_grad = True
        output = model(x_batch)
        output_copy = funSoft(torch.Tensor.cpu(output))
        grad_sum = np.zeros(len(x_batch))
        for label_index in range(class_num):
            y_likely = torch.zeros_like(y_batch) + label_index
            loss = criterion(output, y_likely)
            x_grad = torch.autograd.grad(loss, x_batch, retain_graph=True, create_graph=False)[0]
            grad_sum += np.linalg.norm(torch.Tensor.cpu(x_grad).detach().numpy().squeeze().reshape(len(x_grad), -1), axis=1) * torch.Tensor.cpu(output_copy[:, label_index]).detach().numpy()
        scores.extend(grad_sum)
    scores_sort = np.argsort(scores)
    scores_egl = [int(item) for item in scores_sort]
    select_loc = scores_egl[-num:]
    return np.array(sub_idx)[np.array(select_loc)]

from torch.utils.data import DataLoader, Subset

cri = torch.nn.CrossEntropyLoss()
dev = "cpu"

def obtain_feature(model, sub_train_loader, device):
    scores = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in sub_train_loader:
            x_batch = x_batch.to(device)
            output = model(x_batch)
            output_softmax = torch.softmax(output, dim=1)
            feature_value = torch.from_numpy(entropy(output_softmax.cpu().numpy(), base=2, axis=1))
            scores.append(feature_value)
    return torch.cat(scores).numpy()

def get_susceptibility_score(model, x_batch, target_labels, criterion, device):
    model.eval()
    x_batch.requires_grad_()
    x_batch = x_batch.to(device)
    target_labels = target_labels.to(device)
    output = model(x_batch)
    loss = criterion(output, target_labels)
    model.zero_grad()
    loss.backward()
    gradients = x_batch.grad.data
    gradient_magnitudes = gradients.norm(2, dim=tuple(range(1, gradients.dim())))
    return gradient_magnitudes.cpu().numpy()

def calculate_susceptibility_scores(model, sub_train_loader, criterion, device):
    susceptibility_scores = []
    for x_batch, y_batch in sub_train_loader:
        scores = get_susceptibility_score(model, x_batch, y_batch, criterion, device)
        susceptibility_scores.extend(scores)
    return np.array(susceptibility_scores)

def dreSel(train_dataset, model, num, batch_size, sub_idx, criterion=cri, device=dev, bin_num=50):
    sub_train_dataset = Subset(train_dataset, sub_idx)
    sub_train_loader = DataLoader(dataset=sub_train_dataset, batch_size=batch_size, shuffle=False)
    
    scores = obtain_feature(model, sub_train_loader, device)
    susceptibility_scores = calculate_susceptibility_scores(model, sub_train_loader, criterion, device)
    
    hist_entire, bin_edge_entire = np.histogram(scores, bins=bin_num, weights=np.ones(len(scores)) / len(scores), density=False)
    selection_sizes = np.ceil(hist_entire * num).astype(int)
    
    select_index = []
    for i in range(bin_num):
        indices = np.where((scores >= bin_edge_entire[i]) & (scores < bin_edge_entire[i+1]))[0]
        if indices.size > 0:
            sorted_indices = indices[np.argsort(susceptibility_scores[indices])[::-1]]
            bin_selection_size = min(selection_sizes[i], sorted_indices.size)
            selected_indices = sorted_indices[:bin_selection_size]
            select_index.extend(selected_indices)
    
    select_index = np.array(select_index)
    if select_index.size > num:
        select_index = np.random.choice(select_index, num, replace=False)
    
    selected_indices_original = [sub_idx[idx] for idx in select_index]
    
    return np.array(selected_indices_original)


def deepfoolGene(x, model):
    with torch.no_grad():
        true_label = model(x).max(1)[1]
    x_try = torch.clone(x)
    try_num = 0
    while try_num < 10:
        atk = torchattacks.PGD(model, steps=10)
        adv_data = atk(x_try, true_label)
        adv_label = model(x_try).max(1)[1]
        x_try = adv_data
        if adv_label != true_label:
            break
        try_num += 1
    if adv_label == true_label:
        return np.inf
    elif adv_label != true_label:
        perturbation = torch.Tensor.cpu((adv_data - x).flatten()).detach().numpy()
        return np.linalg.norm(perturbation)


def dfalSel(train_dataset, model, num, sub_idx):
    # select a subset of size 10*num data
    num_subset = min(30 * num, len(sub_idx))
    sub_idx_random = np.random.permutation(len(sub_idx))
    subset_idx = []
    scores = []
    model.eval()
    for i in range(num_subset):
        loc = sub_idx_random[i]
        subset_idx.append(sub_idx[loc])
        x = train_dataset[sub_idx[loc]][0].to(device)
        x.unsqueeze_(0)
        scores.append(deepfoolGene(x, model))
    scores_sort = np.argsort(scores)
    scores_lcr = [int(item) for item in scores_sort]
    select_loc = scores_lcr[:num]
    return np.array(subset_idx)[np.array(select_loc)]


def kcenterSel(train_dataset, model, num, sub_idx, batch_size, class_num):
    full_idx = np.arange(len(train_dataset))
    labeled_idx = np.setdiff1d(full_idx, sub_idx)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False)
    outputs = torch.zeros((len(train_dataset), class_num)).to(device)
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            current_output = funSoft(torch.Tensor.cpu(model(x))).to(device)
            outputs[i * batch_size:i * batch_size + len(x), :] = current_output
    centers = labeled_idx
    candidates = sub_idx
    select_index = np.arange(num)
    for i in range(num):
        outputs_candidate = outputs[candidates]
        outputs_center = outputs[centers]
        distances = torch.zeros(len(candidates)).to(device)
        for idx, output_single in enumerate(outputs_candidate):
            output_repeat = output_single.repeat(len(centers), 1)
            distances[idx] = torch.norm(outputs_center - output_repeat, dim=1).min()
        select_loc = torch.Tensor.cpu(torch.argmax(distances)).detach().numpy()
        select_index[i] = sub_idx[select_loc]
        centers = np.concatenate((centers, [select_index[i]]))
        candidates.pop(select_loc)

    return select_index


def mcpSel(train_dataset, model, num, batch_size, sub_idx, class_num):
    sub_train_dataset = Subset(train_dataset, sub_idx)
    sub_train_loader = torch.utils.data.DataLoader(
        dataset=sub_train_dataset,
        batch_size=batch_size,
        shuffle=False)
    dicratio = [[] for i in range(class_num * class_num)]
    dicindex = [[] for i in range(class_num * class_num)]
    model.eval()
    with torch.no_grad():
        for batch_no, (x_batch, _) in enumerate(sub_train_loader):
            x_batch = x_batch.to(device)
            output = model(x_batch)
            output_copy = funSoft(torch.Tensor.cpu(output)).detach().numpy()
            for i in range(len(output_copy)):
                index_in_data = batch_size * batch_no + i
                act = output_copy[i]
                max_index, sec_index, ratio = find_second(act, class_num)
                dicratio[max_index * class_num + sec_index].append(ratio)
                dicindex[max_index * class_num + sec_index].append(index_in_data)
    selected_list = select_from_firstsec_dic(num, dicratio, dicindex, class_num * class_num)
    select_loc = selected_list[:num]
    return np.array(sub_idx)[np.array(select_loc)]


def find_second(act, matrix_size):
    max_ = 0
    second_max = 0
    sec_index = 0
    max_index = 0
    for i in range(matrix_size):
        if act[i] > max_:
            max_ = act[i]
            max_index = i

    for i in range(matrix_size):
        if i == max_index:
            continue
        if act[i] > second_max:
            second_max = act[i]
            sec_index = i
    ratio = 1.0 * second_max / max_
    return max_index, sec_index, ratio


def no_empty_number(dicratio):
    no_empty = 0
    for i in range(len(dicratio)):
        if len(dicratio[i]) != 0:
            no_empty += 1
    return no_empty


def select_from_firstsec_dic(selectsize, dicratio, dicindex, ms):
    selected_lst = []
    tmpsize = selectsize

    noempty = no_empty_number(dicratio)
    while selectsize >= noempty:
        for i in range(ms):
            if len(dicratio[i]) != 0:
                tmp = max(dicratio[i])
                j = dicratio[i].index(tmp)
                selected_lst.append(dicindex[i][j])
                dicratio[i].remove(tmp)
                dicindex[i].remove(dicindex[i][j])
        selectsize = tmpsize - len(selected_lst)
        noempty = no_empty_number(dicratio)

    while len(selected_lst) != tmpsize:
        max_tmp = [0 for i in range(selectsize)]
        max_index_tmp = [0 for i in range(selectsize)]
        for i in range(ms):
            if len(dicratio[i]) != 0:
                tmp_max = max(dicratio[i])
                if tmp_max > min(max_tmp):
                    index = max_tmp.index(min(max_tmp))
                    max_tmp[index] = tmp_max
                    max_index_tmp[index] = dicindex[i][dicratio[i].index(tmp_max)]
        if len(max_index_tmp) == 0 and len(selected_lst) != tmpsize:
            print('wrong!!!!!!')
            break
        selected_lst = selected_lst + max_index_tmp
    assert len(selected_lst) == tmpsize
    return selected_lst


def save_best(state, checkpoint_fpath):
    torch.save(state, checkpoint_fpath)


def load_best(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_metric = checkpoint['metric_best']
    best_epoch = checkpoint['epoch_best']
    candidate_index = checkpoint['candidate_index']
    current_stage = checkpoint['current_stage']

    return model, optimizer, best_epoch, best_metric, candidate_index, current_stage


def make_drop(modelName, model, dropoutRate=0.1):
    modelDrop = copy.deepcopy(model)
    if modelName == "lenet5":
        feats_list = list(modelDrop.convnet)
        new_feats_list = []
        for feat in feats_list:
            new_feats_list.append(feat)
            if isinstance(feat, nn.ReLU):
                new_feats_list.append(nn.Dropout(p=dropoutRate, inplace=True))
        modelDrop.convnet = nn.Sequential(*new_feats_list)
    elif modelName == "VGG8":
        modelDrop.pool0 = nn.Sequential(nn.Dropout(p=dropoutRate, inplace=True), model.pool0)
        modelDrop.pool1 = nn.Sequential(nn.Dropout(p=dropoutRate, inplace=True), model.pool1)
        modelDrop.pool2 = nn.Sequential(nn.Dropout(p=dropoutRate, inplace=True), model.pool2)
        modelDrop.pool3 = nn.Sequential(nn.Dropout(p=dropoutRate, inplace=True), model.pool3)
        modelDrop.pool4 = nn.Sequential(nn.Dropout(p=dropoutRate, inplace=True), model.pool4)
    elif modelName == "VGG16":
        feats_list = list(modelDrop.features)
        new_feats_list = []
        for feat in feats_list:
            new_feats_list.append(feat)
            if isinstance(feat, nn.ReLU):
                new_feats_list.append(nn.Dropout(p=dropoutRate, inplace=True))
        modelDrop.convnet = nn.Sequential(*new_feats_list)
    elif modelName == "ResNet18" or modelName == "PreActResNet18":
        modelDrop.layer1 = nn.Sequential(nn.Dropout(p=dropoutRate, inplace=True), modelDrop.layer1)

    for m in modelDrop.modules():
        if isinstance(m, nn.Dropout):
            m.train()
        else:
            m.eval()

    return modelDrop



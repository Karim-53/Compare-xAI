import random
from itertools import chain, combinations

import numpy as np
from sklearn.preprocessing import StandardScaler

from explainers.archipelago_lib.baselines.mahe_madex.madex.utils.general_utils import get_sample_distances


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


def random_subset(s):
    out = [el for el in s if random.randint(0, 1) == 0]
    return tuple(out)


def gen_data_samples(model, input_value=1, base_value=-1, p=40, n=30000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = [np.random.choice([input_value, base_value], p) for _ in range(n)]
    X = np.stack(X)

    Y = model(X).squeeze()
    return X, Y


def proprocess_data(
        X, Y, valid_size=500, test_size=500, std_scale=False, std_scale_X=False
):
    n, p = X.shape
    ## Make dataset splits
    ntrain, nval, ntest = n - valid_size - test_size, valid_size, test_size

    Xs = {
        "train": X[:ntrain],
        "val": X[ntrain: ntrain + nval],
        "test": X[ntrain + nval: ntrain + nval + ntest],
    }
    Ys = {
        "train": np.expand_dims(Y[:ntrain], axis=1),
        "val": np.expand_dims(Y[ntrain: ntrain + nval], axis=1),
        "test": np.expand_dims(Y[ntrain + nval: ntrain + nval + ntest], axis=1),
    }

    for k in Xs:
        if len(Xs[k]) == 0:
            assert k != "train"
            del Xs[k]
            del Ys[k]

    if std_scale:
        scaler = StandardScaler()
        scaler.fit(Ys["train"])
        for k in Ys:
            Ys[k] = scaler.transform(Ys[k])
        Ys["scaler"] = scaler

    if std_scale_X:
        scaler = StandardScaler()
        scaler.fit(Xs["train"])
        for k in Xs:
            Xs[k] = scaler.transform(Xs[k])

    return Xs, Ys


def set_seed(seed=42):
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)


# def force_float(X_numpy):
#     return torch.from_numpy(X_numpy.astype(np.float32))


# def create_mlp(layer_sizes, out_bias=True, act_func=nn.ReLU()):
#     ls = list(layer_sizes)
#     layers = nn.ModuleList()
#     for i in range(1, len(ls) - 1):
#         layers.append(nn.Linear(int(ls[i - 1]), int(ls[i])))
#         layers.append(act_func)
#     layers.append(nn.Linear(int(ls[-2]), int(ls[-1]), bias=out_bias))
#     return nn.Sequential(*layers)


# def train(
#         net,
#         data_loaders,
#         criterion=nn.MSELoss(reduction="none"),
#         nepochs=100,
#         verbose=False,
#         early_stopping=True,
#         patience=5,
#         l1_const=1e-4,
#         l2_const=0,
#         learning_rate=0.01,
#         opt_func=optim.Adam,
#         device=torch.device("cpu"),
#         **kwargs
# ):
#     optimizer = opt_func(net.parameters(), lr=learning_rate, weight_decay=l2_const)
#
#     def include_sws(loss, sws):
#         assert loss.shape == sws.shape
#         return (loss * sws / sws.sum()).sum()
#
#     def evaluate(net, data_loader, criterion, device):
#         losses = []
#         sws = []
#         for inputs, targets, sws_batch in data_loader:
#             inputs = inputs.to(device)
#             targets = targets.to(device)
#             loss = criterion(net(inputs), targets).cpu().data
#             losses.append(loss)
#             sws.append(sws_batch)
#         return include_sws(torch.stack(losses), torch.stack(sws)).item()
#
#     best_loss = float("inf")
#     best_net = None
#
#     if "val" not in data_loaders:
#         early_stopping = False
#
#     patience_counter = 0
#
#     for epoch in range(nepochs):
#         if verbose:
#             print("epoch", epoch)
#         running_loss = 0.0
#         run_count = 0
#         for i, data in enumerate(data_loaders["train"], 0):
#             inputs, targets, sws = data
#             inputs = inputs.to(device)
#             targets = targets.to(device)
#             sws = sws.to(device)
#             optimizer.zero_grad()
#             outputs = net(inputs)
#             loss = include_sws(criterion(outputs, targets), sws)
#
#             reg_loss = 0
#             for name, param in net.named_parameters():
#                 if "interaction_mlp" in name and "weight" in name:
#                     reg_loss += torch.sum(torch.abs(param))
#
#             (loss + reg_loss * l1_const).backward()
#             optimizer.step()
#             running_loss += loss.item()
#             run_count += 1
#
#         if epoch % 1 == 0:
#             key = "val" if "val" in data_loaders else "train"
#             val_loss = evaluate(net, data_loaders[key], criterion, device)
#
#             if verbose:
#                 print(
#                     "[%d, %5d] train loss: %.4f, val loss: %.4f"
#                     % (epoch + 1, nepochs, running_loss / run_count, val_loss)
#                 )
#             if early_stopping:
#                 if val_loss < best_loss:
#                     best_loss = val_loss
#                     best_net = copy.deepcopy(net)
#                     patience_counter = 0
#                 else:
#                     patience_counter += 1
#                     if patience_counter > patience:
#                         net = best_net
#                         val_loss = best_loss
#                         if verbose:
#                             print("early stopping!")
#                         break
#
#             prev_loss = running_loss
#             running_loss = 0.0
#
#     if "test" in data_loaders:
#         key = "test"
#     elif "val" in data_loaders:
#         key = "val"
#     else:
#         key = "train"
#     test_loss = evaluate(net, data_loaders[key], criterion, device)
#
#     if verbose:
#         print("Finished Training. Test loss: ", test_loss)
#
#     return net, test_loss


def get_sample_weights(Xs, kernel_width=0.25, enable=True, **kwargs):
    def kernel(d):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

    if enable:
        Dd = get_sample_distances(Xs)

    Wd = {}
    for k in Xs:
        if k == "scaler":
            continue
        Wd[k] = kernel(Dd[k]) if enable else np.ones(Xs[k].shape[0])
    return Wd

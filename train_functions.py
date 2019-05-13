
from core import *
from utils import lr_loss_plot


# loss_function = F.binary_cross_entropy_with_logits

################################
######## 1st approach #########
################################

def get_optimizer(model, lr:float = .01, wd:float = 0.):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.Adam(parameters, lr=lr, weight_decay=wd)
    return optim

######### FIND LEARNING RATE & STORE/LOAD history #####

def lr_finder(model, train_dl, p:(Path,str)=None, lr_low:float=1e-5, lr_high:float=1, epochs:int=2):
    '''
    Lr finder with the first approach 
    
    
    :param model: 
    :param train_dl: 
    :param p: 
    :param lr_low: 
    :param lr_high: 
    :param epochs: 
    :return: 
    '''
    losses = []
    if p: save_model(model, str(p))
        
    iterations = epochs * len(train_dl)
    delta = (lr_high - lr_low) / iterations
    lrs = [lr_low + i * delta for i in range(iterations)]
    model.train()
    ind = 0
    for i in range(epochs):
        train_dl.set_random_choices()
        for x, y in train_dl:
            optim = get_optimizer(model, lr=lrs[ind])
            x = x.cuda().float()
            y = y.cuda()
            out = model(x)
            loss = loss_function(out, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            ind += 1
    if p: load_model(model, str(p))

    return lrs, losses

######### Store/Load lr finder output #####

def to_csv(lrs,losses, file='lrs_losses.csv'):
    with open(str(file), 'w') as f:
        for i in range(len(lrs)):
            f.write(f'{lrs[i]},{losses[i]}\n')


def from_csv(path):
    if not isinstance(path, str):
        path = str(path)

    df = pd.read_csv(path, header=None)
    lr = df[0].tolist()
    losses = df[1].tolist()
    return lr, losses

######### Define LR policy and training #####

def train_regular_policy(model, path, train_dl, valid_dl,
                         loss_function=F.binary_cross_entropy_with_logits,
                         lr_low=1e-6, lr_high=0.001, epochs=50, epsilon=.01, compute_metric=True, data=None):
    if data is None:
        data = train_dl.data
    lr = lr_high
    prev_loss, min_loss = np.inf, np.inf
    for i in range(epochs):
        model.train()
        train_dl.set_random_choices()
        total = 0
        sum_loss = 0
        optim = get_optimizer(model, lr=lr, wd=0)
        for x, y in train_dl:
            batch = y.shape[0]
            out = model(x)
            loss = loss_function(out, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += batch
            sum_loss += batch * (loss.item())
        print("lr %.7f train loss %.5f" % (lr, sum_loss / total))

        if data == 'chest':
            val_loss, measure = val_metrics_chest(model, valid_dl, compute_metric)

        elif data == 'chest-PvsNP' or data == 'binary_task':
            val_loss, measure = val_metrics_chest_PvsNP(model, valid_dl, compute_metric)

        elif data == 'hands':
            val_loss, measure = val_metrics_hands(model, valid_dl,
                                                  loss_function, compute_metric)
        elif data == 'MURA':
            val_loss, measure = val_metrics_MURA(model, valid_dl, compute_metric)

        print("lr %.7f train loss %.5f" % (lr, sum_loss / total, measure))

        if val_loss - prev_loss > epsilon:
            lr = lr / 10.0
        if val_loss < min_loss:
            save_model(model, path)
            min_loss = val_loss
        prev_loss = val_loss
        if lr < lr_low:
            break

    return sum_loss / total




################################
######## 2nd approach #########
################################

### Annealings ###
def exp_annealing(start_lr, end_lr, n):
    ptg = np.linspace(0, 1, n)
    return start_lr * (end_lr / start_lr) ** ptg


def cos_annealing(start_lr, end_lr, n_iterations):
    i = np.arange(n_iterations)
    c_i = 1 + np.cos(i * np.pi / n_iterations)
    return end_lr + (start_lr - end_lr) / 2 * c_i


### Diff lr ###

def diff_range(val, alpha=1.3):
    return [val * alpha ** i for i in range(2, -1, -1)]

#### POLICIES (Finder and Training) ######

class FinderPolicy:

    def __init__(self, n_epochs, dl, min_lr=1e-7, max_lr=10):
        '''
        Implements exponential annealing policy from min_lr to max_lr
        '''
        total_iterations = n_epochs * len(dl)
        self.lr_schedule = exp_annealing(min_lr, max_lr, total_iterations)
        self.mom = .9  # constant momentum policy with default value
        self.idx = -1

    def step(self):
        self.idx = self.idx + 1
        return self.lr_schedule[self.idx], self.mom


# LR finder loop
def lr_finder(model, n_epochs, train_dl, min_lr=1e-4, max_lr=1e-1, save_path=None, early_stopping=200, plot_every=200):
    if save_path: save_model(model, save_path)
    model.train()

    policy = FinderPolicy(n_epochs=n_epochs, dl=train_dl, min_lr=min_lr, max_lr=max_lr)
    optimizer = OptimizerWrapper(model, policy)

    lrs = optimizer.policy.lr_schedule

    losses = []
    cnt = 0

    for _ in tqdm_notebook(range(n_epochs)):
        train_dl.set_random_choices()
        for it, (x, y) in enumerate(tqdm_notebook(train_dl, leave=False)):

            optimizer.zero_grad()

            out = model(x)
            loss = F.binary_cross_entropy_with_logits(input=out.squeeze(), target=y)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if cnt % plot_every == (plot_every-1): lr_loss_plot(lrs, losses)
            if cnt == early_stopping: return lrs[:cnt], losses
            cnt += 1

    if save_path: load_model(model, p)

    return lrs, losses


class TrainingPolicy:
    '''Cretes the lr and momentum policy'''

    def __init__(self, n_epochs, dl, max_lr, pctg=.3, moms=(.95, .85),
                 delta=1e-4, div_factor=25.):
        total_iterations = n_epochs * len(dl)

        iter1 = int(total_iterations * pctg)
        iter2 = total_iterations - int(total_iterations * pctg)
        iterations = (iter1, iter2)

        min_start = max_lr / div_factor
        min_end = min_start * delta

        lr_segments = ((min_start, max_lr), (max_lr, min_end))
        mom_segments = (moms, (moms[1], moms[0]))

        self.lr_schedule = self._create_schedule(lr_segments, iterations)
        self.mom_schedule = self._create_schedule(mom_segments, iterations)

        self.idx = -1

    def _create_schedule(self, segments, iterations):
        '''
        Creates a schedule given a function, behaviour and size
        '''
        stages = [cos_annealing(start, end, n) for ((start, end), n) in zip(segments, iterations)]
        return np.concatenate(stages)

    def step(self):
        self.idx += 1
        return self.lr_schedule[self.idx], self.mom_schedule[self.idx]






#### OPTIMIZER WRAPPERS ######
class OptimizerWrapper:
    '''Without using the momentum policy'''

    def __init__(self, model, policy, wd=0, alpha=1. / 3):

        self.policy = policy  # TrainingPolicy(n_epochs=n_epochs, dl=dl, max_lr=max_lr)

        self.model = model
        self.alpha = alpha
        self.wd = wd

        # This assumes the model is defined by groups.
        param_groups = [group.parameters() for group in list(self.model.children())[0]]
        lr_0 = self.policy.lr_schedule[0]
        mom_0 = self.policy.mom_schedule[0] if hasattr(self.policy, 'mom_schedule') else .9

        groups = zip(param_groups, diff_range(lr_0, alpha=alpha), diff_range(mom_0, alpha=1))

        self.optimizer = optim.Adam(
            [{'params': p, 'lr': lr, 'mom': (mom, .999)} for p, lr, mom in groups]
        )

    def _update_optimizer(self):
        lr_i, mom_i = self.policy.step()
        groups = zip(self.optimizer.param_groups,
                     diff_range(lr_i, alpha=self.alpha),
                     diff_range(mom_i, alpha=1))

        for param_group, lr, mom in groups:
            param_group['lr'] = lr
            param_group['mom'] = (mom, .999)

    def _weight_decay(self):
        for group in self.optimizer.param_group:
            for p in group['params']: p.data.mul_(group['lr'] * self.wd)

    def step(self):
        self._update_optimizer()
        if self.wd != 0: self._weight_decay()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

########## METRICS #################

def R2L1(y, out):
    y_bar = np.mean(y)
    numerator = np.sum(np.absolute(out-y))
    denominator = np.sum(np.absolute(y-y_bar))
    return 1 - numerator/denominator

def ave_auc(probs, ys):
    aucs = [roc_auc_score(ys[:, i], probs[:, i]) for i in range(probs.shape[1])]
    return np.mean(aucs), aucs

########## VALIDATION #################

def cuda2cpu_classification(y): return y.long().cpu().numpy()


def cuda2cpu_regression(y): return y.cpu().numpy()


def validate_loop(model, valid_dl, task):
    if task=='binary' or task=='multilabel':
        cuda2cpu = cuda2cpu_classification
        loss_fun = F.binary_cross_entropy_with_logits
    elif task=='regression':
        cuda2cpu = cuda2cpu_regression
        loss_fun = F.l1_loss

    model.eval()
    total = 0
    sum_loss = 0
    ys = []
    preds = []

    for x, y in valid_dl:
        out = model(x)
        loss = loss_fun(out.squeeze(), y)

        batch = y.shape[0]
        sum_loss += batch * (loss.item())
        total += batch

        preds.append(out.squeeze().detach().cpu().numpy())
        ys.append(cuda2cpu(y))

    return sum_loss/total, preds, ys


def validate_multilabel(model, valid_dl):
    loss, preds, ys = validate_loop(model, valid_dl, 'multilabel')

    preds = np.vstack(preds)
    ys = np.vstack(ys)

    mean_auc, aucs = ave_auc(preds, ys)

    return loss, mean_auc, aucs

def validate_binary(model, valid_dl):
    loss, preds, ys = validate_loop(model, valid_dl, 'binary')

    preds = np.concatenate(preds)
    ys = np.concatenate(ys)

    auc = roc_auc_score(ys, preds)
    accuracy = accuracy_score(ys, (preds>.5).astype(np.int))

    return loss, auc, accuracy

def validate_regression(model, valid_dl):
    loss, preds, ys = validate_loop(model, valid_dl, 'regression')

    preds = np.concatenate(preds)
    ys = np.concatenate(ys)

    R2 = R2L1(y=ys,out=preds)
    return loss, R2

########## TTA #################

def TTA_loop(model, valid_dl, task, ndl=4):
    if task=='binary' or task=='multilabel':
        cuda2cpu = cuda2cpu_classification
        loss_fun = F.binary_cross_entropy_with_logits
    elif task=='regression':
        cuda2cpu = cuda2cpu_regression
        loss_fun = F.l1_loss

    model.eval()
    total = 0
    sum_loss = 0
    ys = []
    preds = [[] for _ in range(ndl)]

    for i in range(ndl - 1):
        valid_dl.set_random_choices()
        for x, y in valid_dl:
            out = model(x)
            loss = loss_fun(out.squeeze(), y)

            batch = y.shape[0]
            sum_loss += batch * (loss.item())
            total += batch
            preds[i].append(out.squeeze().detach().cpu().numpy())

    for x, y in valid_dl:
        out = model(x)
        loss = loss_fun(out.squeeze(), y)

        batch = y.shape[0]
        sum_loss += batch * (loss.item())
        total += batch

        preds[ndl - 1].append(out.squeeze().detach().cpu().numpy())
        ys.append(cuda2cpu(y))

    return sum_loss / total, preds, ys


def TTA_multilabel(model, valid_dl, ndl=4):
    loss, preds, ys = TTA_loop(model, valid_dl, 'multilabel', ndl)

    preds = [np.vstack(pred) for pred in preds]
    preds = np.mean(preds, axis=0)
    ys = np.vstack(ys)

    mean_auc, aucs = ave_auc(preds, ys)

    print("TTA loss %.4f and auc %.4f" % (loss, mean_auc))
    return loss, mean_auc, aucs


def TTA_binary(model, valid_dl, ndl=4):
    loss, preds, ys = TTA_loop(model, valid_dl, 'binary', ndl)

    preds = [np.concatenate(pred) for pred in preds]
    preds = np.mean(preds, axis=0)
    ys = np.concatenate(ys)

    auc = roc_auc_score(ys, preds)
    accuracy = accuracy_score(ys, (preds>0).astype(int))
    print("TTA loss %.4f  auc %.4f  accuracy %.4f" % (loss, auc, accuracy))
    return loss, auc, accuracy


def TTA_regression(model, valid_dl, ndl=4):
    loss, preds, ys = TTA_loop(model, valid_dl, 'regression', ndl)

    preds = [np.concatenate(pred) for pred in preds]
    preds = np.mean(preds, axis=0)
    ys = np.concatenate(ys)

    R2 = R2L1(y=ys, out=preds)
    print("TTA loss %.4f pseudo R2 (L1) %.4f " % (loss, R2))
    return loss, R2



#### LR FINDER AND TRAINING WITH POLICY ####

# loss_functions = {'binary': F.binary_cross_entropy_with_logits,
#                   'multilabel': F.binary_cross_entropy_with_logits,
#                   'multiclass': F.cross_entropy,
#                   'regression': F.l1_loss
#                   }
#
# def lr_finder(model, n_epochs, train_dl, min_lr=1e-7, max_lr=10, save_path=None,
#               mode='exponential', bar=tqdm_notebook, early_stopping=200):
#
#     if save_path: save_model(model, save_path)
#
#     optimizer = FinderOptimizerWrapper(model, n_epochs, train_dl, min_lr=min_lr, max_lr=max_lr, wd=0, mode=mode)
#
#     lrs = optimizer.policy.lr_schedule
#     losses = []
#     cnt = 0
#
#     for _ in bar(range(n_epochs)):
#         model.train()
#         train_dl.set_random_choices()
#         for it, (x, y) in enumerate(bar(train_dl)):
#
#             optimizer.zero_grad()
#
#             out = model(x)
#             loss = F.binary_cross_entropy_with_logits(input=out, target=y)
#
#             loss.backward()
#             optimizer.step()
#
#             losses.append(loss.item())
#
#             if it%200 == 199:
#                 plt.plot(lrs[:len(losses)], losses)
#                 plt.xticks(rotation=45)
#                 plt.show()
#
#             if cnt==early_stopping: return lrs[:cnt], losses
#             cnt +=1
#
#     if save_path: load_model(model, p)
#
#     return lrs, losses
#
#
# def train(n_epochs, train_dl, valid_dl, model, div_factor=25., max_lr=.01, wd=0, alpha=1./ 3, classification_type='binary',
#           save_path=None, bar=tqdm_notebook, val_function=None, unfreeze_during_loop:tuple=None):
#
#     model.train()
#
#     best_loss = np.inf
#
#     loss_f = loss_functions[classification_type]
#
#     validate = val_function if val_function else get_val_metric(train_dl)
#
#     optimizer = OptimizerWrapper(model, n_epochs, train_dl, div_factor=div_factor, max_lr=max_lr, wd=wd, alpha=alpha)
#
#     if unfreeze_during_loop:
#         if not isinstance(unfreeze_during_loop, (list, tuple)): raise ValueError('unfreeze_during_loop requires to  be None, list or a tuple')
#         total_iter = n_epochs*len(train_dl)
#         first_unfreeze = int(total_iter*unfreeze_during_loop[0])
#         second_unfreeze = int(total_iter*unfreeze_during_loop[1])
#
#     for epoch in bar(range(n_epochs)):
#         div = 0
#         agg_loss = 0
#         if hasattr(train_dl, 'set_random_choices'): train_dl.set_random_choices()
#         for i, (x, y) in enumerate(train_dl):
#
#             if unfreeze_during_loop:
#                 if i == first_unfreeze: model.unfreeze(1)
#                 if i == second_unfreeze: model.unfreeze(0)
#
#             out = model(x)
#             optimizer.zero_grad()
#             loss = loss_f(input=out, target=y)
#             loss.backward()
#             optimizer.step()
#
#             agg_loss += loss.item()
#             div += 1
#
#
#         val_loss, measure = validate(model, valid_dl, True)
#         print(f'Ep. {epoch+1} - train loss {agg_loss/div:.4f} -  val loss {val_loss:.4f} AUC {measure:.4f}')
#
#
#
#         if save_path and val_loss < best_loss:
#             save_model(model, save_path)
#             best_loss = val_loss
#
#

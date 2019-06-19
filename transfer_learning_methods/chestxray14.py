
import sys; sys.path.append("/data/miguel/practicum/DL-Medical-Physics")

from core import *
from data_manipulation import DataBatches, RandomRotation, Flip, RandomCrop
from utils import save_model, load_model, lr_loss_plot
from architectures import DenseNet121
from train_functions import get_optimizer, FinderPolicy, OptimizerWrapper, validate_multilabel, TTA_multilabel
import json

BATCH_SIZE = 16
EPOCHS = 10
TRANSFORMATIONS = [RandomRotation(arc_width=20), Flip(), RandomCrop(r_pix=8)]
DATA = '14diseases'

SAMPLE_AMOUNTS = [50,100,200,400,600,800,1000,1200,1400,1600,1800,2000]

BASE_PATH = Path('/data/miguel/practicum/')
PATH = BASE_PATH/'data'
SAVE_DIRECTORY = BASE_PATH/'DL-Medical-Physics/transfer_learning_methods/models'
SAVE_DATA = BASE_PATH/'DL-Medical-Physics/transfer_learning_methods/results'
IMG_FOLDER = PATH/'ChestXRay-250'
PRETRAINED = True # Imagenet


def cos_annealing(start_lr, end_lr, n_iterations):
    i = np.arange(n_iterations)
    c_i = 1 + np.cos(i * np.pi / n_iterations)
    return end_lr + (start_lr - end_lr) / 2 * c_i

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
    
def one_cycle_train(n_epochs, train_dl, valid_dl, model, max_lr=.01, wd=0, alpha=1./ 3,
          save_path=None, unfreeze_during_loop:tuple=None):
    
    if unfreeze_during_loop:
        total_iter = n_epochs*len(train_dl)
        first_unfreeze = int(total_iter*unfreeze_during_loop[0])
        second_unfreeze = int(total_iter*unfreeze_during_loop[1])

    best_loss = np.inf
    cnt = 0
    
    policy = TrainingPolicy(n_epochs=n_epochs, dl=train_dl, max_lr=max_lr)
    optimizer = OptimizerWrapper(model, policy, wd=wd, alpha=alpha)

    for epoch in range(n_epochs):
        model.train()
        agg_div = 0
        agg_loss = 0
        train_dl.set_random_choices()
        for x, y in train_dl:

            if unfreeze_during_loop:
                if cnt == first_unfreeze: model.unfreeze(1)
                if cnt == second_unfreeze: model.unfreeze(0)

            out = model(x)
            loss = F.binary_cross_entropy_with_logits(input=out, target=y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch = y.shape[0]
            agg_loss += batch*loss.item()
            agg_div += batch
            cnt += 1


        val_loss, measure, _ = validate_multilabel(model, valid_dl)
        print(f'Ep. {epoch+1} - train loss {agg_loss/agg_div:.4f} -  val loss {val_loss:.4f} AUC {measure:.4f}')

        if save_path and val_loss < best_loss:
            save_model(model, save_path)
            best_loss = val_loss

# Training            
train_df = pd.read_csv(PATH/"train_df.csv")
valid_df = pd.read_csv(PATH/"val_df.csv")

valid_dl = DataBatches(valid_df,img_folder_path=IMG_FOLDER,transforms=False, 
                       shuffle=False, data=DATA, batch_size=BATCH_SIZE, normalize=PRETRAINED)

train_df = train_df.sample(frac=1)

for N in SAMPLE_AMOUNTS:

    df = train_df[:N]

    train_dl = DataBatches(df, img_folder_path=IMG_FOLDER, transforms=TRANSFORMATIONS, 
                           shuffle=True, data=DATA, batch_size=BATCH_SIZE, normalize=PRETRAINED)

    freeze = True
    gradual_unfreezing = False    

    model = DenseNet121(14, pretrained=PRETRAINED, freeze=freeze).cuda()

    save_path = SAVE_DIRECTORY/f"chestxray14-feature-extractor-{N}.pth"

    one_cycle_train(EPOCHS, train_dl, valid_dl, model, max_lr=.001, save_path=save_path, unfreeze_during_loop=(.1, .2) if gradual_unfreezing else None, alpha=1)

    freeze = False
    gradual_unfreezing = False
    
    model = DenseNet121(14, pretrained=PRETRAINED, freeze=freeze).cuda()

    save_path = SAVE_DIRECTORY/f"chestxray14-fine-tune-all-{N}.pth"
    
    one_cycle_train(EPOCHS, train_dl, valid_dl, model, max_lr=.001, save_path=save_path, unfreeze_during_loop=(.1, .2) if gradual_unfreezing else None, alpha=1)
    
    freeze = True
    gradual_unfreezing = True
    
    model = DenseNet121(14, pretrained=PRETRAINED, freeze=freeze).cuda()

    save_path = SAVE_DIRECTORY/f"chestxray14-grad-unfr-diff-lr-{N}.pth"
    
    one_cycle_train(EPOCHS, train_dl, valid_dl, model, max_lr=.001, save_path=save_path, unfreeze_during_loop=(.1, .2) if gradual_unfreezing else None, alpha=1./3)
    
# Evaluation

feature_extractor = {
    'losses':[],
    'aucs':[]
}
fine_tune_all = {
    'losses':[],
    'aucs':[]
}
grad_unfr_diff_lr = {
    'losses':[],
    'aucs':[]
}

test_df = pd.read_csv(PATH/"test_df.csv")

test_dl = DataBatches(test_df,img_folder_path=IMG_FOLDER, transforms=TRANSFORMATIONS, 
                      shuffle=False, data=DATA, batch_size=BATCH_SIZE, normalize=PRETRAINED)

for i, N in enumerate(SAMPLE_AMOUNTS):

    model = DenseNet121(14, pretrained=PRETRAINED, freeze=False).cuda()

    load_path = SAVE_DIRECTORY/f"chestxray14-feature-extractor-{N}.pth"

    load_model(model, load_path)

    loss, mean_auc, _ = TTA_multilabel(model, test_dl, ndl=4)

    feature_extractor['losses'].append(loss)
    feature_extractor['aucs'].append(mean_auc)

    load_path = SAVE_DIRECTORY/f"chestxray14-fine-tune-all-{N}.pth"

    load_model(model, load_path)

    loss, mean_auc, _ = TTA_multilabel(model, test_dl, ndl=4)

    fine_tune_all['losses'].append(loss)
    fine_tune_all['aucs'].append(mean_auc)
    
    save_path = SAVE_DIRECTORY/f"chestxray14-grad-unfr-diff-lr-training-{N}.pth"

    load_model(model, load_path)

    loss, mean_auc, _ = TTA_multilabel(model, test_dl, ndl=4)

    grad_unfr_diff_lr['losses'].append(loss)
    grad_unfr_diff_lr['aucs'].append(mean_auc)

feature_extractor = json.dumps(feature_extractor)
with open('results/chestxray14_feature_extractor.json', 'w') as f:
    f.write(feature_extractor)

fine_tune_all = json.dumps(fine_tune_all)
with open('results/chestxray14_fine_tune_all.json', 'w') as f:
    f.write(fine_tune_all)
    
grad_unfr_diff_lr = json.dumps(grad_unfr_diff_lr)
with open('results/chestxray14_grad_unfr_diff_lr.json', 'w') as f:
    f.write(grad_unfr_diff_lr)


import sys; sys.path.append("..")
import warnings; warnings.filterwarnings('ignore')

from core import * 
from data_manipulation import Transform, RandomRotation, Flip, RandomCrop, balance_obs, multi_label_2_binary, DataBatches
from utils import save_model, load_model, lr_loss_plot
from architectures import DenseNet121
from train_functions import OptimizerWrapper, TrainingPolicy, FinderPolicy, validate_multilabel, lr_finder, validate_binary, TTA_binary
import json

SEED = 42
R_PIX = 8
IDX = 10
BATCH_SIZE = 16
EPOCHS = 30
TRANSFORMATIONS = [RandomRotation(arc_width=20), Flip(), RandomCrop(r_pix=R_PIX)]
NORMALIZE = True # ImageNet
FREEZE = True
GRADUAL_UNFREEZING = True
n_samples = [50,100,200,400,600,800,1000,1200,1400,1600,1800, 2000]
# n_samples = [6, 12, 24, 36, 48]

print(n_samples)
BASE_PATH = Path('/data/miguel/practicum/')
PATH = BASE_PATH/'data'
IMG_FOLDER = PATH/'ChestXRay-250'
DATA = 'Pneumonia' # will take advantage of the dataset structure for pneumonia 
DISEASE = 'Hernia'

idx2tgt = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
tgt2idx = {disease: i for i, disease in enumerate(idx2tgt)}

# To balance validation and testing
def decode_labels(df_col):
    return np.array(list(map(np.array, df_col.str.split(' ')))).astype(int)

def subset_df(df, amt, idx=IDX):
    
    lbls = decode_labels(df.Label)
    
    pos_idxs = lbls[:,idx].astype(bool)

    neg = df[~pos_idxs].sample(n=amt//2, replace=False)
    pos = df[pos_idxs].sample(n=amt//2, replace=False)

    return pd.concat([neg, pos]).reset_index(drop=True)

def train(n_epochs, train_dl, valid_dl, model, max_lr=.01, wd=0, alpha=1./ 3,
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
            loss = F.binary_cross_entropy_with_logits(input=out.squeeze(), target=y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch = y.shape[0]
            agg_loss += batch*loss.item()
            agg_div += batch
            cnt += 1


        val_loss, measure, _ = validate_binary(model, valid_dl)
        print(f'Ep. {epoch+1} - train loss {agg_loss/agg_div:.4f} -  val loss {val_loss:.4f} AUC {measure:.4f}')

        if save_path and val_loss < best_loss:
            save_model(model, save_path)
            best_loss = val_loss
            
            
train_df = pd.read_csv(PATH/'train_df.csv')
valid_df = pd.read_csv(PATH/"val_df.csv")
test_df = pd.read_csv(PATH/"test_df.csv")

train_df = multi_label_2_binary(train_df, tgt2idx[DISEASE])

valid_df = multi_label_2_binary(valid_df, tgt2idx[DISEASE])
valid_df = balance_obs(valid_df, amt=2*len(valid_df[valid_df['Label']==1]))

test_df = multi_label_2_binary(test_df, tgt2idx[DISEASE])
test_df = balance_obs(test_df, amt=2*len(test_df[test_df['Label']==1]))

no_pretrained = {'loss': [],
           'auc': [],
           'accuracy': []}

imagenet = {'loss': [],
           'auc': [],
           'accuracy': []}

MURA = {'loss': [],
           'auc': [],
           'accuracy': []}

chexpert = {'loss': [],
           'auc': [],
           'accuracy': []}

for N in n_samples:
    
    train_df_balanced = balance_obs(train_df, amt=N)

    train_dl = DataBatches(df=train_df_balanced, transforms=TRANSFORMATIONS, shuffle=True,
                           img_folder_path=IMG_FOLDER, batch_size=BATCH_SIZE, data=DATA,
                           r_pix=R_PIX, normalize=NORMALIZE, seed=SEED)

    valid_dl = DataBatches(df=valid_df, transforms=None, shuffle=False,
                           img_folder_path=IMG_FOLDER, batch_size=BATCH_SIZE, data=DATA,
                           r_pix=R_PIX, normalize=NORMALIZE, seed=SEED)

    test_dl = DataBatches(df=test_df, transforms=TRANSFORMATIONS, shuffle=False,
                          img_folder_path=IMG_FOLDER, batch_size=BATCH_SIZE, data=DATA,
                          r_pix=R_PIX, normalize=NORMALIZE, seed=SEED)
    
    print('ImageNet...')
    pretrained = True
    model = DenseNet121(1, pretrained=pretrained, freeze=FREEZE).cuda()
    model_p = f'models/best_{DISEASE.lower()}_{N}_imagenet.pth'
    train(EPOCHS, train_dl, valid_dl, model, max_lr=.001, save_path=model_p, 
          unfreeze_during_loop=(.1, .2) if GRADUAL_UNFREEZING else None)
    
    print('Testing with TTA ....')
    load_model(model, model_p)
    loss, auc, accuracy = TTA_binary(model, test_dl)
    imagenet['loss'].append(loss)
    imagenet['auc'].append(auc)
    imagenet['accuracy'].append(accuracy)
    
    print('MURA...')
    pretrained = 'MURA'
    model = DenseNet121(1, pretrained=pretrained, freeze=FREEZE).cuda()
    model_p = f'models/best_{DISEASE.lower()}_{N}_MURA.pth'
    train(EPOCHS, train_dl, valid_dl, model, max_lr=.001, save_path=model_p, 
          unfreeze_during_loop=(.1, .2) if GRADUAL_UNFREEZING else None)

    print('Testing with TTA ....')
    load_model(model, model_p)
    loss, auc, accuracy = TTA_binary(model, test_dl)
    MURA['loss'].append(loss)
    MURA['auc'].append(auc)
    MURA['accuracy'].append(accuracy)
    
    print('CheXPert...')
    pretrained = 'chexpert'
    model = DenseNet121(1, pretrained=pretrained, freeze=FREEZE).cuda()
    model_p = f'models/best_{DISEASE.lower()}_{N}_chexpert.pth'
    train(EPOCHS, train_dl, valid_dl, model, max_lr=.001, save_path=model_p, 
          unfreeze_during_loop=(.1, .2) if GRADUAL_UNFREEZING else None)

    print('Testing with TTA ....')
    load_model(model, model_p)
    loss, auc, accuracy = TTA_binary(model, test_dl)
    chexpert['loss'].append(loss)
    chexpert['auc'].append(auc)
    chexpert['accuracy'].append(accuracy)
    
    train_dl = DataBatches(df=train_df_balanced, transforms=TRANSFORMATIONS, shuffle=True,
                           img_folder_path=IMG_FOLDER, batch_size=BATCH_SIZE, data=DATA,
                           r_pix=R_PIX, normalize=False, seed=SEED)

    valid_dl = DataBatches(df=valid_df, transforms=None, shuffle=False,
                           img_folder_path=IMG_FOLDER, batch_size=BATCH_SIZE, data=DATA,
                           r_pix=R_PIX, normalize=False, seed=SEED)

    test_dl = DataBatches(df=test_df, transforms=TRANSFORMATIONS, shuffle=False,
                          img_folder_path=IMG_FOLDER, batch_size=BATCH_SIZE, data=DATA,
                          r_pix=R_PIX, normalize=False, seed=SEED)

    print('No pretrained...')
    pretrained = True
    model = DenseNet121(1, pretrained=False, freeze=False).cuda()
    model_p = f'models/best_{DISEASE.lower()}_{N}_no_pretrained.pth'
    train(EPOCHS, train_dl, valid_dl, model, max_lr=.001, save_path=model_p,
          unfreeze_during_loop=None)

    print('Testing with TTA ....')
    load_model(model, model_p)
    loss, auc, accuracy = TTA_binary(model, test_dl)
    no_pretrained['loss'].append(loss)
    no_pretrained['auc'].append(auc)
    no_pretrained['accuracy'].append(accuracy)

imagenet = json.dumps(imagenet)
with open(f'data_plots/imagenet_{DISEASE.lower()}.json', 'w') as f:
# with open('data_plots/imagenet_small.json', 'w') as f:
    f.write(imagenet)

MURA = json.dumps(MURA)
with open(f'data_plots/MURA_{DISEASE.lower()}.json', 'w') as f:
# with open('data_plots/MURA_small.json', 'w') as f:
    f.write(MURA)
    
chexpert = json.dumps(chexpert)
with open(f'data_plots/chexpert_{DISEASE.lower()}.json', 'w') as f:
# with open('data_plots/chexpert_small.json', 'w') as f:
    f.write(chexpert)

no_pretrained = json.dumps(no_pretrained)
with open(f'data_plots/no_pretrained_{DISEASE.lower()}.json', 'w') as f:
# with open('data_plots/no_pretrained_small.json', 'w') as f:
    f.write(no_pretrained)

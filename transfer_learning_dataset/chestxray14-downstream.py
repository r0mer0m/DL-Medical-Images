
import sys; sys.path.append("..")
import warnings; warnings.filterwarnings('ignore')

from core import * 
from data_manipulation import Transform, RandomRotation, Flip, RandomCrop, balance_obs, multi_label_2_binary, DataBatches
from utils import save_model, load_model, lr_loss_plot
from architectures import DenseNet121
from train_functions import OptimizerWrapper, TrainingPolicy, FinderPolicy, validate_multilabel, lr_finder, validate_multilabel, TTA_multilabel
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
DATA = '14diseases'

# idx2tgt = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
#                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
# tgt2idx = {disease: i for i, disease in enumerate(idx2tgt)}

# To balance validation and testing
def decode_labels(df_col):
    return np.array(list(map(np.array, df_col.str.split(' ')))).astype(int)

def subset_df(df, amt, idx=IDX):
    
    lbls = decode_labels(df.Label)
    
    pos_idxs = lbls[:,idx].astype(bool)

    neg = df[~pos_idxs].sample(n=amt//2, replace=False)
    pos = df[pos_idxs].sample(n=amt//2, replace=False)

    return pd.concat([neg, pos]).reset_index(drop=True)

# class ChestXray1DataSet(Dataset):
#     """
#     Basic Images DataSet

#     Args:
#         dataframe with data: image_file, label
#     """

#     def __init__(self, df, image_path, idx):
#         self.image_files = df["ImageIndex"].values
#         self.lables = np.array([obs.split(" ")[idx]
#                                 for obs in df.Label]).astype(np.float32)
#         self.image_path = image_path

#     def __getitem__(self, index):
#         path = self.image_path / self.image_files[index]
#         x = cv2.imread(str(path)).astype(np.float32)
#         x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255
#         y = self.lables[index]
#         return x, y

#     def __len__(self):
#         return len(self.image_files)
    
# class DataBatches:
#     '''
#     Creates a dataloader using the specificed data frame with the dataset corresponding to "data".
#     '''

#     def __init__(self, df, idx, transforms, shuffle, img_folder_path, batch_size=16, num_workers=8,
#                  drop_last=False, r_pix=8, normalize=True, seed=42):

#         self.dataset = Transform(ChestXray1DataSet(df, image_path=img_folder_path, idx=idx),
#                                  transforms=transforms, normalize=normalize, seed=seed, r_pix=r_pix)
#         self.dataloader = DataLoader(
#             self.dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
#             shuffle=shuffle, drop_last=drop_last
#         )
       

#     def __iter__(self): return ((x.cuda().float(), y.cuda().float()) for (x, y) in self.dataloader)

#     def __len__(self): return len(self.dataloader)

#     def set_random_choices(self):
#         if hasattr(self.dataset, "set_random_choices"): self.dataset.set_random_choices()
            
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


        val_loss, measure, _ = validate_multilabel(model, valid_dl)
        print(f'Ep. {epoch+1} - train loss {agg_loss/agg_div:.4f} -  val loss {val_loss:.4f} AUC {measure:.4f}')

        if save_path and val_loss < best_loss:
            save_model(model, save_path)
            best_loss = val_loss
            
            
train_df = pd.read_csv(PATH/'train_df.csv')
valid_df = pd.read_csv(PATH/"val_df.csv")
test_df = pd.read_csv(PATH/"test_df.csv")

# train_df = multi_label_2_binary(train_df, tgt2idx['Pneumonia'])

# valid_df = multi_label_2_binary(valid_df, tgt2idx['Pneumonia'])
# valid_df = balance_obs(valid_df, amt=2*len(valid_df[valid_df['Label']==1]))

# test_df = multi_label_2_binary(test_df, tgt2idx['Pneumonia'])
# test_df = balance_obs(test_df, amt=2*len(test_df[test_df['Label']==1]))

train_df = train_df.sample(frac=1)

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
    
    train_df_balanced = train_df[:N]

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
    model = DenseNet121(14, pretrained=pretrained, freeze=FREEZE).cuda()
    model_p = f'models/best_14diseases_{N}_imagenet.pth'
    train(EPOCHS, train_dl, valid_dl, model, max_lr=.001, save_path=model_p, 
          unfreeze_during_loop=(.1, .2) if GRADUAL_UNFREEZING else None)
    
    print('Testing with TTA ....')
    load_model(model, model_p)
    loss, auc, accuracy = TTA_multilabel(model, test_dl)
    imagenet['loss'].append(loss)
    imagenet['auc'].append(auc)
    imagenet['accuracy'].append(accuracy)
    
    print('MURA...')
    pretrained = 'MURA'
    model = DenseNet121(14, pretrained=pretrained, freeze=FREEZE).cuda()
    model_p = f'models/best_14diseases_{N}_MURA.pth'
    train(EPOCHS, train_dl, valid_dl, model, max_lr=.001, save_path=model_p, 
          unfreeze_during_loop=(.1, .2) if GRADUAL_UNFREEZING else None)

    print('Testing with TTA ....')
    load_model(model, model_p)
    loss, auc, accuracy = TTA_multilabel(model, test_dl)
    MURA['loss'].append(loss)
    MURA['auc'].append(auc)
    MURA['accuracy'].append(accuracy)
    
    print('CheXPert...')
    pretrained = 'chexpert'
    model = DenseNet121(14, pretrained=pretrained, freeze=FREEZE).cuda()
    model_p = f'models/best_14diseases_{N}_chexpert.pth'
    train(EPOCHS, train_dl, valid_dl, model, max_lr=.001, save_path=model_p, 
          unfreeze_during_loop=(.1, .2) if GRADUAL_UNFREEZING else None)

    print('Testing with TTA ....')
    load_model(model, model_p)
    loss, auc, accuracy = TTA_multilabel(model, test_dl)
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
    model = DenseNet121(14, pretrained=False, freeze=False).cuda()
    model_p = f'models/best_14diseases_{N}_no_pretrained.pth'
    train(EPOCHS, train_dl, valid_dl, model, max_lr=.001, save_path=model_p,
          unfreeze_during_loop=None)

    print('Testing with TTA ....')
    load_model(model, model_p)
    loss, auc, accuracy = TTA_multilabel(model, test_dl)
    no_pretrained['loss'].append(loss)
    no_pretrained['auc'].append(auc)
    no_pretrained['accuracy'].append(accuracy)

imagenet = json.dumps(imagenet)
with open('data_plots/imagenet_14diseases.json', 'w') as f:
# with open('data_plots/imagenet_small.json', 'w') as f:
    f.write(imagenet)

MURA = json.dumps(MURA)
with open('data_plots/MURA_14diseases.json', 'w') as f:
# with open('data_plots/MURA_small.json', 'w') as f:
    f.write(MURA)
    
chexpert = json.dumps(chexpert)
with open('data_plots/chexpert_14diseases.json', 'w') as f:
# with open('data_plots/chexpert_small.json', 'w') as f:
    f.write(chexpert)

no_pretrained = json.dumps(no_pretrained)
with open('data_plots/no_pretrained_14diseases.json', 'w') as f:
# with open('data_plots/no_pretrained_small.json', 'w') as f:
    f.write(no_pretrained)

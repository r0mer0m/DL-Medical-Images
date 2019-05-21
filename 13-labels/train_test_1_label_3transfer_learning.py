
import sys; sys.path.append("..")
import warnings; warnings.filterwarnings('ignore')

from core import * 
from data_manipulation import Transform, RandomRotation, Flip, RandomCrop
from utils import save_model, load_model, lr_loss_plot
from architectures import DenseNet121
from train_functions import OptimizerWrapper, TrainingPolicy, FinderPolicy, validate_multilabel, lr_finder, validate_binary, TTA_binary
import json

SEED = 42
R_PIX = 8
IDX = 10
BATCH_SIZE = 8
EPOCHS = 30
TRANSFORMATIONS = [RandomRotation(arc_width=20), Flip(), RandomCrop(r_pix=R_PIX)]
NORMALIZE = True # ImageNet
FREEZE = True
GRADUAL_UNFREEZING = True
# n_samples = [50,100,200,400,600,800,1000,1200,1400,1600,1800, 2000]
n_samples = [8,  16, 24, 32, 40, 48]
print(n_samples)
BASE_PATH = Path('../..')
PATH = BASE_PATH/'data'
IMG_FOLDER = PATH/'ChestXRay-250'

# To balance validation and testing
def decode_labels(df_col):
    return np.array(list(map(np.array, df_col.str.split(' ')))).astype(int)

def subset_df(df, amt, idx=IDX):
    
    lbls = decode_labels(df.Label)
    
    pos_idxs = lbls[:,idx].astype(bool)

    neg = df[~pos_idxs].sample(n=amt//2, replace=False)
    pos = df[pos_idxs].sample(n=amt//2, replace=False)

    return pd.concat([neg, pos]).reset_index(drop=True)

class ChestXray1DataSet(Dataset):
    """
    Basic Images DataSet

    Args:
        dataframe with data: image_file, label
    """

    def __init__(self, df, image_path, idx):
        self.image_files = df["ImageIndex"].values
        self.lables = np.array([obs.split(" ")[idx]
                                for obs in df.Label]).astype(np.float32)
        self.image_path = image_path

    def __getitem__(self, index):
        path = self.image_path / self.image_files[index]
        x = cv2.imread(str(path)).astype(np.float32)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255
        y = self.lables[index]
        return x, y

    def __len__(self):
        return len(self.image_files)
    
class DataBatches:
    '''
    Creates a dataloader using the specificed data frame with the dataset corresponding to "data".
    '''

    def __init__(self, df, idx, transforms, shuffle, img_folder_path, batch_size=16, num_workers=8,
                 drop_last=False, r_pix=8, normalize=True, seed=42):

        self.dataset = Transform(ChestXray1DataSet(df, image_path=img_folder_path, idx=idx),
                                 transforms=transforms, normalize=normalize, seed=seed, r_pix=r_pix)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
            shuffle=shuffle, drop_last=drop_last
        )
       

    def __iter__(self): return ((x.cuda().float(), y.cuda().float()) for (x, y) in self.dataloader)

    def __len__(self): return len(self.dataloader)

    def set_random_choices(self):
        if hasattr(self.dataset, "set_random_choices"): self.dataset.set_random_choices()
            
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
            
            
train_df = pd.read_csv('train_df_small.csv')
valid_df = pd.read_csv(PATH/"val_df.csv")
test_df = pd.read_csv(PATH/"test_df.csv")

val_amt = 2*decode_labels(valid_df.Label)[:,IDX].sum()
test_amt = 2*decode_labels(test_df.Label)[:,IDX].sum()

valid_df_balanced = subset_df(valid_df, val_amt, idx=IDX)
test_df_balanced = subset_df(test_df, test_amt, idx=IDX)


no_pretrained = {'loss': [],
           'auc': [],
           'accuracy': []}

imagenet = {'loss': [],
           'auc': [],
           'accuracy': []}

MURA = {'loss': [],
           'auc': [],
           'accuracy': []}

diseases13 = {'loss': [],
           'auc': [],
           'accuracy': []}

for N in n_samples:
    
    train_df_balanced = subset_df(train_df, N, idx=IDX)

    train_dl = DataBatches(df=train_df_balanced, idx=IDX, transforms=TRANSFORMATIONS, shuffle=True,
                           img_folder_path=IMG_FOLDER, batch_size=BATCH_SIZE, 
                           r_pix=R_PIX, normalize=NORMALIZE, seed=SEED)

    valid_dl = DataBatches(df=valid_df_balanced, idx=IDX, transforms=None, shuffle=False,
                           img_folder_path=IMG_FOLDER, batch_size=BATCH_SIZE,
                           r_pix=R_PIX, normalize=NORMALIZE, seed=SEED)

    test_dl = DataBatches(df=test_df_balanced, idx=IDX, transforms=TRANSFORMATIONS, shuffle=False,
                          img_folder_path=IMG_FOLDER, batch_size=BATCH_SIZE,
                          r_pix=R_PIX, normalize=NORMALIZE, seed=SEED)
    
    print('ImageNet...')
    pretrained = True
    model = DenseNet121(1, pretrained=pretrained, freeze=FREEZE).cuda()
    model_p = f'models/best_{N}_imagenet.pth'
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
    model_p = f'models/best_{N}_MURA.pth'
    train(EPOCHS, train_dl, valid_dl, model, max_lr=.001, save_path=model_p, 
          unfreeze_during_loop=(.1, .2) if GRADUAL_UNFREEZING else None)

    print('Testing with TTA ....')
    load_model(model, model_p)
    loss, auc, accuracy = TTA_binary(model, test_dl)
    MURA['loss'].append(loss)
    MURA['auc'].append(auc)
    MURA['accuracy'].append(accuracy)
    
    print('13 diseases...')
    pretrained = '13diseases'
    model = DenseNet121(1, pretrained=pretrained, freeze=FREEZE).cuda()
    model_p = f'models/best_{N}_13diseases.pth'
    train(EPOCHS, train_dl, valid_dl, model, max_lr=.001, save_path=model_p, 
          unfreeze_during_loop=(.1, .2) if GRADUAL_UNFREEZING else None)

    print('Testing with TTA ....')
    load_model(model, model_p)
    loss, auc, accuracy = TTA_binary(model, test_dl)
    diseases13['loss'].append(loss)
    diseases13['auc'].append(auc)
    diseases13['accuracy'].append(accuracy)

    train_dl = DataBatches(df=train_df_balanced, idx=IDX, transforms=TRANSFORMATIONS, shuffle=True,
                           img_folder_path=IMG_FOLDER, batch_size=BATCH_SIZE,
                           r_pix=R_PIX, normalize=False, seed=SEED)

    valid_dl = DataBatches(df=valid_df_balanced, idx=IDX, transforms=None, shuffle=False,
                           img_folder_path=IMG_FOLDER, batch_size=BATCH_SIZE,
                           r_pix=R_PIX, normalize=False, seed=SEED)

    test_dl = DataBatches(df=test_df_balanced, idx=IDX, transforms=TRANSFORMATIONS, shuffle=False,
                          img_folder_path=IMG_FOLDER, batch_size=BATCH_SIZE,
                          r_pix=R_PIX, normalize=False, seed=SEED)

    print('No pretrained...')
    pretrained = True
    model = DenseNet121(1, pretrained=False, freeze=False).cuda()
    model_p = f'models/best_{N}_no_pretrained.pth'
    train(EPOCHS, train_dl, valid_dl, model, max_lr=.001, save_path=model_p,
          unfreeze_during_loop=None)

    print('Testing with TTA ....')
    load_model(model, model_p)
    loss, auc, accuracy = TTA_binary(model, test_dl)
    no_pretrained['loss'].append(loss)
    no_pretrained['auc'].append(auc)
    no_pretrained['accuracy'].append(accuracy)

imagenet = json.dumps(imagenet)
# with open('data_plots/imagenet.json', 'w') as f:
with open('data_plots/imagenet_small.json', 'w') as f:
    f.write(imagenet)

MURA = json.dumps(MURA)
# with open('data_plots/MURA.json', 'w') as f:
with open('data_plots/MURA_small.json', 'w') as f:
    f.write(MURA)
    
diseases13 = json.dumps(diseases13)
# with open('data_plots/13diseases.json', 'w') as f:
with open('data_plots/13diseases_small.json', 'w') as f:
    f.write(diseases13)

no_pretrained = json.dumps(no_pretrained)
# with open('data_plots/no_pretrained.json', 'w') as f:
with open('data_plots/no_pretrained_small.json', 'w') as f:
    f.write(no_pretrained)

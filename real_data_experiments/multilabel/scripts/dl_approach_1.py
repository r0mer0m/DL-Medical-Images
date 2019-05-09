
import sys; sys.path.append("../../..")

from core import *
from data_manipulation import DataBatches, RandomRotation, Flip, RandomCrop
from utils import save_model, load_model, lr_loss_plot
from architectures import DenseNet121
from train_functions import get_optimizer, FinderPolicy, OptimizerWrapper, validate_multilabel

BATCH_SIZE = 16
EPOCHS = 50
TRANSFORMATIONS = [RandomRotation(arc_width=20), Flip(), RandomCrop(r_pix=8)]
DATA = '14diseases'

# (pretrained, freeze first blocks, prog_unfreezing)
METHODS = [(False, False, False), 
           (True, True, False), (True, False, False), (True, True, True),
           ('MURA', True, False), ('MURA', False, False), ('MURA', True, True)] 

RANDOM_STATES = range(10)
SAMPLE_AMOUNTS = [50,100,200,400,600,800,1000,1200,1400,1600,1800,2000]

BASE_PATH = Path('../../../../')
PATH = BASE_PATH/'data'
SAVE_DIRECTORY = BASE_PATH/'output/real_data_experiments/multilabel/models'
SAVE_DATA = BASE_PATH/'output/real_data_experiments/multilabel/results'
IMG_FOLDER = PATH/'ChestXRay-250'


def train(epochs, train_dl, valid_dl, model, save_path=None,min_lr=1e-6,
          max_lr=0.001, epsilon=.001, unfreeze_during_loop:tuple=None):
    
    lr = max_lr
    prev_loss, min_loss = np.inf, np.inf
    cnt = 0
    
    if unfreeze_during_loop:
        total_iter = n_epochs*len(train_dl)
        first_unfreeze = int(total_iter*unfreeze_during_loop[0])
        second_unfreeze = int(total_iter*unfreeze_during_loop[1])
    
    for epoch in range(epochs):
        model.train()
        train_dl.set_random_choices()
        total = 0
        sum_loss = 0
        optim = get_optimizer(model, lr=lr, wd=0)
        for x, y in train_dl:
            
            if unfreeze_during_loop:
                if cnt == first_unfreeze: model.unfreeze(1)
                if cnt == second_unfreeze: model.unfreeze(0)
            
            batch = y.shape[0]
            out = model(x)
            loss = F.binary_cross_entropy_with_logits(out, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += batch
            sum_loss += batch * (loss.item())
            
            cnt += 1
                
        val_loss, measure, _ = validate_multilabel(model, valid_dl)
        print(f'Ep. {epoch+1} - lr {lr:.7f} train loss {sum_loss/total:.4f} -  val loss {val_loss:.4f} AUC {measure:.4f}')

        if val_loss - prev_loss > epsilon:
            lr = lr / 10.0
        if val_loss < min_loss:
            if save_path: save_model(model, save_path)
            min_loss = val_loss
        
        prev_loss = val_loss
        if lr < min_lr:
            break

# Training            
train_df = pd.read_csv(PATH/"train_df.csv")
valid_df = pd.read_csv(PATH/"val_df.csv")

for pretrained, freeze, grad_unfreez in METHODS:
    
    valid_dl = DataBatches(valid_df,img_folder_path=IMG_FOLDER,transforms=False, 
                           shuffle=False, data=DATA, batch_size=BATCH_SIZE, normalize=pretrained)

    for rs in RANDOM_STATES:

        train_df = train_df.sample(frac=1)

        for N in SAMPLE_AMOUNTS:

            df = train_df[:N]

            train_dl = DataBatches(df, img_folder_path=IMG_FOLDER, transforms=TRANSFORMATIONS, 
                                   shuffle=True, data=DATA, batch_size=BATCH_SIZE, normalize=pretrained)

            model = DenseNet121(14, pretrained=pretrained, freeze=freeze).cuda()

            save_path = SAVE_DIRECTORY/f"app1_{pretrained}-{freeze}-{grad_unfreez}-{N}-{rs}.pth"

            train(EPOCHS, train_dl, valid_dl, model, max_lr=.001, save_path=save_path, unfreeze_during_loop=(.1, .2) if grad_unfreez else None)

# Evaluation

test_df = pd.read_csv(PATH/"test_df.csv")

for pretrained, freeze, grad_unfreeze in METHODS:
    
    losses = [[] for _ in n_samples]
    aucs = [[] for _ in n_samples]
    
    test_dl = DataBatches(test_df,img_folder_path=IMG_FOLDER, transforms=TRANSFORMATIONS, 
                          shuffle=False, data=DATA, batch_size=BATCH_SIZE, normalize=pretrained)
    
    for i, N in enumerate(SAMPLE_AMOUNTS):
        
        for rs in RANDOM_STATES:
            
            model = DenseNet121(14, pretrained=pretrained, freeze=freeze).cuda()

            load_path = SAVE_DIRECTORY/f"app1_{pretrained}-{freeze}-{grad_unfreeze}-{N}-{rs}.pth"
            
            load_model(model, load_path)
            
            loss, mean_auc, _ = TTA_multilabel(model, test_dl, ndl=4)
            
            losses[i].append(loss)
            aucs[i].append(mean_auc)
    
    loss_path = SAVE_DATA/f"losses_{pretrained}_{freeze}_{grad_unfreeze}"
    aucs_path = SAVE_DATA/f"aucs_{pretrained}_{freeze}_{grad_unfreeze}"
    
    numpy.save(loss_path, np.array(losses))
    numpy.save(aucs_path, np.array(aucs))
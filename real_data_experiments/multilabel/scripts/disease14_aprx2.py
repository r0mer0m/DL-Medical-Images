
from core import * # basic imports
from data_manipulation import DataBatches, RandomRotation, Flip, RandomCrop
from utils import save_model, load_model, lr_loss_plot
from architectures import DenseNet121
from train_functions import OptimizerWrapper, TrainingPolicy, FinderPolicy, validate_multilabel, TTA_multilabel
import warnings; warnings.filterwarnings('ignore')

batch_size = 16
epochs = 10
methods = [(False, False, False),
           (True, True, False),
           (True, False, False),
           (True, True, True),
           ('MURA', True, False),
           ('MURA', False, False),
           ('MURA', True, True)] # pretrained / freeze first blocks / prog_unfreezing

random_states = range(10)
n_samples = [50,100,200,400,600,800,1000,1200,1400,1600,1800, 2000]

PATH = Path('../data')
SAVE_DIRECTORY = Path('../latest_models/14diseases-app2')

img_folder_path = PATH/'ChestXRay-250'
data = '14diseases'


def train(n_epochs, train_dl, valid_dl, model, div_factor=25., max_lr=.01, wd=0, alpha=1./ 3,
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


transforms=[RandomRotation(arc_width=20), Flip(), RandomCrop(r_pix=8)]

train_df = pd.read_csv(PATH/"train_df.csv")
valid_df = pd.read_csv(PATH/"val_df.csv")


for pretrained, freeze, grad_unfreeze in methods:
    
    valid_dl = DataBatches(valid_df,img_folder_path=img_folder_path,
                     transforms = False, shuffle = False, data= data,
                     batch_size = batch_size, normalize=pretrained)

    for rs in random_states:

        train_df = train_df.sample(frac=1)

        for N in n_samples:

            df = train_df[:N]

            train_dl = DataBatches(df, img_folder_path=img_folder_path, transforms=transforms, shuffle=True, data=data,
                                   batch_size=batch_size, normalize=pretrained)

            model = DenseNet121(14, pretrained=pretrained, freeze=freeze).cuda()

            save_path = SAVE_DIRECTORY/f"{pretrained}-{freeze}-{grad_unfreeze}-{N}-{rs}.pth"

            train(epochs, train_dl, valid_dl, model, max_lr=.001, save_path=save_path, unfreeze_during_loop=(.1, .2) if grad_unfreeze else None)

            
            
SAVE_DATA = Path('../latest_data/14diseases-app2')         
            

test_df = pd.read_csv(PATH/"test_df.csv")



for pretrained, freeze, grad_unfreeze in methods:
    
    test_dl = DataBatches(test_df,img_folder_path=img_folder_path,
                  transforms = True, shuffle = False, data=data,
                  batch_size = batch_size, normalize=pretrained)
    
    losses = [[] for _ in n_samples]
    aucs = [[] for _ in n_samples]
    
    loss_path = SAVE_DATA/f"losses_{pretrained}_{freeze}_{grad_unfreeze}"
    aucs_path = SAVE_DATA/f"aucs_{pretrained}_{freeze}_{grad_unfreeze}"
    
    for i, N in enumerate(n_samples):
        
        for rs in random_states:
            
            model = DenseNet121(14, pretrained=pretrained, freeze=freeze).cuda()

            load_path = SAVE_DIRECTORY/f"{pretrained}-{freeze}-{grad_unfreeze}-{N}-{rs}.pth"
            
            load_model(model, load_path)
            
            loss, mean_auc, _ = TTA_multilabel(model, test_dl, ndl=4)
            
            losses[i].append(loss)
            aucs[i].append(mean_auc)
    
    losses = np.array(losses)
    aucs = np.array(aucs)
    
    numpy.save(loss_path, losses)
    numpy.save(aucs_path, aucs)
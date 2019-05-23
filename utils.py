from core import *

# __all__=["read_image","center_crop","crop","resize_crop_image", "normalize", "normalize_image",
#          "normalize_all_images","random_crop","rotate_cv","compute_AUCs", "val_metrics", "TTA_val_metrics",
#          "TTA_val_metrics_chest", "TTA_val_metrics_hands","save_model", "load_model",
#          "class_name","class_name2id", "N_CLASSES"]

# PATH = Path('../data')
#sz=200
# Variables for the chest data set.
class_name = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
class_name2id = {n:i for i,n in enumerate(class_name)}
N_CLASSES = len(class_name)

# def sigmoid(x):
#     return 1. / (1. + np.exp(-x))

def read_image(path):
    im = cv2.imread(str(path)).astype(np.float32)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)/255

def resize(x, sz:(int, tuple)):
    out = cv2.resize(x, sz if isinstance(sz, tuple) else (sz, sz))
    return out

def read_resize(img_path, sz=250):
    im = read_image(img_path)
    return resize(im, sz)

# def resize_crop_image(path, sz=(250, 250)):
#     im = read_image(path)
#     im = center_crop(im)
#     return cv2.resize(im, sz)


###### BALANCE DATA #####
# TODO: Change to upsampling rather than downsampling

# def sample_dl(df, K, img_folder_path, rs=42, data='chest'):
#     if K<len(df):
#         dl = DataBatches(df.sample(K, random_state=rs).reset_index(drop=True),
#                          img_folder_path=img_folder_path, data=data,
#                          transform = True, shuffle = True,
#                          batch_size = batch_size)
#     elif len(df)==K:
#         dl = DataBatches(df,
#                          img_folder_path=img_folder_path,
#                          transform = True, shuffle = True, data=data,
#                          batch_size = batch_size)
#     return dl


def save_model(m, p): torch.save(m.state_dict(), p)


def load_model(m, p): m.load_state_dict(torch.load(p))


########## Plotting ################

def lr_loss_plot(lrs, losses):
    plt.plot(lrs[:len(losses)], losses)
    plt.xticks(rotation=45)
    plt.show()



##################################################
########## Synthetic data experiments ############
##################################################

def soft2hard(array, threshold=.5):
    """
    Given the label column of the dataframe will return 1/0 depending on > threshold or < threshold respectively.
    """
    if isinstance(array[0],str):
        seq_int = np.array([[int(float(item) > threshold) for item in record.split(' ')] for record in array])
    else:
        seq_int = np.array([int(float(record) > threshold) for record in array])
    return seq_int


def percentage_over_threshold(array, threshold=.5, printit=True):
    """
    Computes the total number of positive cases over the total number of labels:

    Careful: Total number  of labels, not records!
    """
    array = soft2hard(array, threshold=threshold)
    p_s = np.sum(array) / (array.size)
    if printit:
        print(f"Amount of positive cases: {p_s*100:.3f}%")
    else:
        return f"{p_s*100:.3f}%"


def num2str(array, sep=' '):
    """
    Given the output of `soft2hard` changes the column formatting keeping the changed values (hard predictions)
    so the upcoming functions can use it seemlessly.
    """
    if len(array.shape) > 1:
        return [sep.join([str(item) for item in record]) for record in array.tolist()]
    else:
        return [str(record) for record in array.tolist()]


#### Transform initial data frames
def create_new_labels_by_median(df, median=None, old_col='boneage', new_col='boneage'):
    """
    From month to category: 1/0 if over/under (training) median.

    """
    if median is None:
        median = df[old_col].median()
        df[new_col] = (df[old_col] > median).astype(int)
        return df, median
    else:
        df[new_col] = (df[old_col] > median).astype(int)
        return df


def load_numpy(filename, dtype, shape):
    return np.array(np.memmap(filename, dtype=dtype, mode='r', shape=shape))


def load_all_data_from_linear_generator():
    df = pd.read_csv(f'../hand_data/synthetic_experiment/generator/linear/load_table.csv', index_col='data')
    df['shape'] = [ast.literal_eval(t) for t in df['shape']]
    arrays = []
    for data in ['X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test']:
        fname = str(f'../hand_data/synthetic_experiment/generator/linear/{data}.dat')
        row = df.loc[data]
        arrays.append(load_numpy(filename = fname, dtype = row['dtype'], shape = row['shape']))
    return tuple(arrays)


def get_mean_and_boundaries(array):
    # Normality Assumption  made
    wide = 1.96*np.std(array, axis=1)
    mean = np.mean(array, axis=1)
    upper = mean + wide
    lower = mean - wide
    return mean, upper, lower


from core import *
#from __utils__ import *

#### Image  Normalization ####
def normalize_imagenet(im):
    """Normalizes images with Imagenet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0])/imagenet_stats[1]

def normalize_mura(im):
    """Normalizes images with MURA stats."""
    mura_stats = np.array([[0.204, 0.204, 0.204], [0.121, 0.121, 0.121]])
    return (im - mura_stats[0])/mura_stats[1]


######### TRANSFORMATIONS ##########


def crop(im, r, c, target_r, target_c): return im[r:r+target_r, c:c+target_c]


class RandomCrop:
    """
    Returns a random crop

    Those are the arguments used when specifiying the transformations.

    Args:
        target_r:int Target Width
        target_c:int Target Hight

    """

    def __init__(self, r_pix=8):
        self.r_pix = r_pix

    def __call__(self, x, rand_r, rand_c):
        """To be called in  the transform
        """

        r, c, *_ = x.shape

        c_pix = round(self.r_pix * c / r)

        start_r = np.floor(2 * rand_r * self.r_pix).astype(int)
        start_c = np.floor(2 * rand_c * c_pix).astype(int)

        # print(start_r, start_c, r-2*self.r_pix, c-2*c_pix)
        return crop(x, start_r, start_c, r - 2 * self.r_pix, c - 2 * c_pix)

    def options(self, x_shape):
        """Specify the random arguments to be generated every epoch.
        Images must be have same dimensions !
        """
        r, c, *_ = x_shape
        return {"rand_r": -1,
                "rand_c": -1}

    def set_random_choices(self, N, x_shape):
        return {k: (v * np.random.uniform(0, 1, size=N)).astype(int)
                for k, v in self.options(x_shape).items()}


class RandomRotation:
    """ Rotates an image by deg degrees

    Args: -
    """

    def __init__(self, arc_width: float = 20): self.arc_width = arc_width

    def __call__(self, im, deg,
                 mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
        r, c, *_ = im.shape
        M = cv2.getRotationMatrix2D((c / 2, r / 2), deg, 1)
        return cv2.warpAffine(im, M, (c, r), borderMode=mode,
                              flags=cv2.WARP_FILL_OUTLIERS + interpolation)

    def options(self, x_shape):
        """Specify the random arguments to be generated every epoch.
        Images must be have same dimensions !
        """
        return {"deg": -1}

    def set_random_choices(self, N, x_shape):
        return {k: ((np.random.random(size=N) - .50) * self.arc_width) for k, v in self.options(x_shape).items()}


class Flip:
    """ Rotates an image by deg degrees

    Args: -
    """

    def __init__(self): pass

    def __call__(self, im, flip):
        if flip > .5:
            im = np.fliplr(im).copy()
        return im

    def options(self, x_shape):
        """Specify the random arguments to be generated every epoch.
        Images must be have same dimensions !
        """
        return {"flip": -1}

    def set_random_choices(self, N, x_shape):
        return {k: np.random.random(size=N) for k, v in self.options(x_shape).items()}


def center_crop(im, r_pix=8):
    """ Returns a center crop of an image"""
    r, c, *_ = im.shape
    c_pix = round(r_pix * c / r)
    return crop(im, r_pix, c_pix, r - 2 * r_pix, c - 2 * c_pix)


###### Dataset by set of data ########

# DataBunches creation
class ChestXrayDataSet(Dataset):
    """
    Basic Images DataSet

    Args:
        dataframe with data: image_file, label
    """

    def __init__(self, df, image_path):
        self.image_files = df["ImageIndex"].values
        self.labels = df["Label"].values
        self.image_path = image_path

    def __getitem__(self, index):
        path = self.image_path / self.image_files[index]
        x = cv2.imread(str(path)).astype(np.float32)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255
        y = np.array([int(i) for i in self.labels[index].split(" ")]).astype(np.float32)
        return x, y

    def __len__(self):
        return len(self.image_files)

class ChestXrayDataSet_PvsNP(Dataset):
    """
    Basic Images DataSet

    Args:
        dataframe with data: image_file, label
    """

    def __init__(self, df, image_path):
        self.image_files = df["ImageIndex"].values
        self.labels = df["Label"].values
        self.image_path = image_path

    def __getitem__(self, index):
        path = self.image_path / self.image_files[index]
        x = cv2.imread(str(path)).astype(np.float32)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255
        y = np.array([self.labels[index]]).astype(np.float32)
        return x, y

    def __len__(self):
        return len(self.image_files)

class HandXrayDataSet(Dataset):
    """
    Basic Images DataSet

    Args:
        dataframe with data: image_file, label
    """

    def __init__(self, df, image_path=Path('../hand_data/boneage-550')):
        self.image_files = df["id"].apply(lambda x: str(x)+'.png').values
        self.labels = df["boneage"].values
        self.image_path = image_path

    def __getitem__(self, index):
        path = self.image_path / self.image_files[index]
        x = cv2.imread(str(path)).astype(np.float32)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255
        # y = self.labels[index]
        y = np.array([self.labels[index]]).astype(np.float32)

        return x, y

    def __len__(self):
        return len(self.labels)

class MURAXrayDataSet(Dataset):
    """
    Basic Images DataSet

    Args:
        dataframe with data: image_file, label
    """

    def __init__(self, df, image_path=Path('/data/miguel/practicum/')):
        self.image_files = df["img_path"].values
        self.labels = df["label"].astype(int).values
        self.image_path = image_path

    def __getitem__(self, index):
        path = self.image_path / self.image_files[index]
        x = cv2.imread(str(path)).astype(np.float32)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255
        y = self.labels[index]
        y = np.expand_dims(y, axis=-1)
        return x, y

    def __len__(self):
        return len(self.labels)

###### Wrapper: Apply transformations over any dataset ########

class Transform():
    """ Rotates an image by deg degrees

    Args:

        dataset: A base torch.utils.data.Dataset of images
        transforms: list with all the transformations involving randomnes

        Ex:
            ds_transform = Transform(ds, [random_crop(240, 240), rotate_cv()])

    """

    def __init__(self, dataset, transforms=None, normalize=True, seed=42, r_pix=8):
        self.dataset, self.transforms = dataset, transforms

        if normalize is True: self.normalize = normalize_imagenet
        elif normalize=='MURA': self.normalize = normalize_mura
        else: self.normalize = False

        self.center_crop = partial(center_crop, r_pix=r_pix)

        np.random.seed(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Do transformation when image is called.
        We are assuming the trainingvalidation set is read from a folder of images already
        noramlized and resized to before random-crop and after random-crop sizes respectively.

        """
        data, label = self.dataset[index]

        if self.transforms:
            for choices, f in list(zip(self.choices, self.transforms)):
                args = {k: v[index] for k, v in choices.items()}
                data = f(data, **args)
        else:
            data=self.center_crop(im=data)

        if self.normalize: data = self.normalize(data)

        return np.rollaxis(data, 2), label


    def set_random_choices(self):
        """
        To be called at the begining of every epoch to generate the random numbers
        for all iterations and transformations.
        """
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)

        for t in self.transforms:
            self.choices.append(t.set_random_choices(N, x_shape))

########## DATASET MANAGER -WITH TRANSFORMATIONS- ######

BASE_DATASETS = {'hands': HandXrayDataSet,
                 '14diseases': ChestXrayDataSet,
                 'chest-PvsNP': ChestXrayDataSet_PvsNP,
                 'MURA': MURAXrayDataSet
                 }


class DataBatches:
    '''
    Creates a dataloader using the specificed data frame with the dataset corresponding to "data".
    '''

    def __init__(self, df, transforms, shuffle, img_folder_path, data, batch_size=16, num_workers=8,
                 drop_last=False, r_pix=8, normalize=True, seed=42):

        self.data = data

        if transforms == True: transforms = [RandomRotation(arc_width=20), Flip(), RandomCrop(r_pix)] # default tfm
        elif transforms == False: transforms = None

        if data in BASE_DATASETS.keys():
            self.dataset = Transform(BASE_DATASETS[data](df, image_path=img_folder_path),
                                     transforms=transforms, normalize=normalize, seed=seed, r_pix=r_pix)
            self.dataloader = DataLoader(
                self.dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                shuffle=shuffle, drop_last=drop_last
            )
        else: ValueError("This dataset is not contemplated")

    def __iter__(self): return ((x.cuda().float(), y.cuda().float()) for (x, y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def set_random_choices(self):
        if hasattr(self.dataset, "set_random_choices"):
            self.dataset.set_random_choices()

######### CREATE BALANCED DL (Binary data) #########

def multi_label_2_binary(df, idx=6):
    '''
    Transforms the dataframe:


            |...|      Label    |              |...| Label |
            |...| 0 1 .. 1 .. 1 |              |...|   1   |
            |...| 1 0 .. 0 .. 1 |    -->       |...|   0   |
            |...| 0 1 .. 0 .. 0 |              |...|   0   |
            |...| 0 0 .. 1 .. 1 |              |...|   1   |
                        idx

    Assumes: labels under the Label column, in a str format separated with a space.

    :param df: data frame where Labels column is a string of 1/0 separated by spaces.
    :param idx: Index of the disease we aim to use. 6 - Pneumonia by default
    :return: dataframe with 1/0 in Labels.
    '''
    df_out = df.copy(deep=True)
    df_out.Label = [int(y.split(' ')[idx]) for y in df_out.Label]
    return df_out


def balance_obs(df:pd.DataFrame, amt:int =None, random_state:int=42):
    '''
    Balance the data for a binary classification task.
    
    Assumes that:
         > The amount of ones is smaller than the amount of zeros.
         > `amt` is always smaller than the amount of zeros.
         
    
    :param df: dataframe with a binary task with integer labels in the Label column. 
    :param amt: Absolute amount of desired amount of samples of each type.
    :param random_state: Random state fore reploducibility.
    :return: dataframe with balanced data and 2*`amt` rows.
    
    '''
    
    if amt is None: amt = len(df[df.Label==1])
    else: amt = amt//2
    
    pos_df = df[df.Label==1]
    neg_df = df[df.Label==0]
    
    if amt > len(df[df.Label==1]):
        pos = pos_df
        pos_upsampling = pos_df.sample(f = amt-len(df[df.Label==1]), random_state=random_state, replace=True)
        neg = neg_df.sample(n = amt, random_state=random_state, replace=False)
        data = [pos, pos_upsampling, neg]
    else:
        pos = pos_df.sample(f=amt, random_state=random_state, replace=False)
        neg = neg_df.sample(n=amt, random_state=random_state, replace=False)
        data = [pos,  neg]

    return pd.concat(data).reset_index(drop=True)


# def balance_dl(unbalanced_df, random_state, img_folder_path, data, batch_size, amt=None, transforms=True, shuffle=True):
#     
#     balanced_df = balance_obs(df=unbalanced_df, amt= amt if amt is None else amt//2, random_state=random_state)
#     
#     balanced_dl = DataBatches(balanced_df,img_folder_path=img_folder_path,
#                            transforms=transforms, shuffle=shuffle, data=data,
#                            batch_size=batch_size, half=False, normalize = True, seed = random_state)
#     return balanced_dl



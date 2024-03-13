import numpy as np
import numpy.testing as npt
import torch
import torch.utils.data as Data
from PIL import Image
from torchvision import transforms
import itertools
import random

######################################### When data is in [channel*batch, height, weight] format
def data_loader(BATCH_SIZE, transform_flag = True, split_flag=True):
    train_dataset = mnist_dataset(train=True, transform = transform_flag, splitted=split_flag)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Batch size is ",BATCH_SIZE, "Num_Of_Iter_TrainLoader", len(train_loader))

    val_dataset = mnist_dataset(train=False, transform = transform_flag, splitted=split_flag)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Batch size is ",BATCH_SIZE, "Num_Of_Iter_TestLoader", len(val_loader))
    
    return train_dataset, train_loader, val_dataset, val_loader


######################################### When data is in [channel*batch, height, weight] format
class mnist_dataset(Data.Dataset):
    def __init__(self, train, transform, splitted):
        ### self, train=True, transform=None, splitted=True, random_seed=1    
        self.transform = transform
        self.train = train
        self.splitted=splitted
        
        if self.splitted:
            self.images_train = np.load('data/mnist/train/train_data.npy')
            self.gt_labels_train = np.load('data/mnist/train/train_gt_labels.npy')
            self.ann1_labels_train = np.load('data/mnist/train/train_ann1_labels.npy')
            self.ann2_labels_train = np.load('data/mnist/train/train_ann2_labels.npy')
            self.ann3_labels_train = np.load('data/mnist/train/train_ann3_labels.npy')

            self.images_val = np.load('data/mnist/val/val_data.npy')
            self.gt_labels_val = np.load('data/mnist/val/val_gt_labels.npy')
            self.ann1_labels_val = np.load('data/mnist/val/val_ann1_labels.npy')
            self.ann2_labels_val = np.load('data/mnist/val/val_ann2_labels.npy')
            self.ann3_labels_val = np.load('data/mnist/val/val_ann3_labels.npy')
        else:
            self.images_train = np.load('data/mnist/train_whole/train_data.npy')
            self.gt_labels_train = np.load('data/mnist/train_whole/train_gt_labels.npy')
            self.ann1_labels_train = np.load('data/mnist/train_whole/train_ann1_labels.npy')
            self.ann2_labels_train = np.load('data/mnist/train_whole/train_ann2_labels.npy')
            self.ann3_labels_train = np.load('data/mnist/train_whole/train_ann3_labels.npy')

            self.images_val = np.load('data/mnist/test/test_data.npy')
            self.gt_labels_val = np.load('data/mnist/test/test_gt_labels.npy')
            self.ann1_labels_val = np.load('data/mnist/test/test_ann1_labels.npy')
            self.ann2_labels_val = np.load('data/mnist/test/test_ann2_labels.npy')
            self.ann3_labels_val = np.load('data/mnist/test/test_ann3_labels.npy')

    
    def __getitem__(self, index):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32,32)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        tr = transforms.ToTensor()
        
        if self.train:
            if self.transform:
                img = transform_train(self.images_train[index,:,:])
            else:
                img = tr(self.images_train[index,:,:])
                
            gt_lbl = self.gt_labels_train[index]
            ann1_lbl = self.ann1_labels_train[index]
            ann2_lbl = self.ann2_labels_train[index]
            ann3_lbl = self.ann3_labels_train[index]

        else:
            if self.transform:
                img = transform_train(self.images_val[index,:,:])
            else:
                img = tr(self.images_val[index,:,:])
                
            gt_lbl = self.gt_labels_val[index]
            ann1_lbl = self.ann1_labels_val[index]
            ann2_lbl = self.ann2_labels_val[index]
            ann3_lbl = self.ann3_labels_val[index]
            
#         img = Image.fromarray(img)       

#         img = torch.squeeze(img, dim=0)
   
#         if self.target_transform is not None:
#             label = self.target_transform(label)
        
        return img, gt_lbl, ann1_lbl, ann2_lbl, ann3_lbl, index  
    
    def __len__(self):
        if self.train:
            return len(self.gt_labels_train)
        else:
            return len(self.gt_labels_val)


########################################## When data is in [channel, batch, height, weight] format and using TensorDataset
import torch
from torch.utils.data import Dataset

class CustomTensorDataset(Dataset):
  def __init__(self, train, transform, splitted):
    self.transform = transform
    self.train = train
    self.splitted=splitted

    if self.splitted:
        self.images_train = torch.tensor(np.load('data/mnist/train/train_data.npy'))
        self.gt_labels_train = torch.tensor(np.load('data/mnist/train/train_gt_labels.npy'))
        self.ann1_labels_train = torch.tensor(np.load('data/mnist/train/train_ann1_labels.npy'))
        self.ann2_labels_train = torch.tensor(np.load('data/mnist/train/train_ann2_labels.npy'))
        self.ann3_labels_train = torch.tensor(np.load('data/mnist/train/train_ann3_labels.npy'))

        self.images_val = torch.tensor(np.load('data/mnist/val/val_data.npy'))
        self.gt_labels_val = torch.tensor(np.load('data/mnist/val/val_gt_labels.npy'))
        self.ann1_labels_val = torch.tensor(np.load('data/mnist/val/val_ann1_labels.npy'))
        self.ann2_labels_val = torch.tensor(np.load('data/mnist/val/val_ann2_labels.npy'))
        self.ann3_labels_val = torch.tensor(np.load('data/mnist/val/val_ann3_labels.npy'))
        
        self.train_tensors = (self.images_train, self.gt_labels_train, self.ann1_labels_train, self.ann2_labels_train, self.ann3_labels_train)
        self.val_tensors = (self.images_val, self.gt_labels_val, self.ann1_labels_val, self.ann2_labels_val, self.ann3_labels_val)
    else:
        self.images_train = torch.tensor(np.load('data/mnist/train_whole/train_data.npy'))
        self.gt_labels_train = torch.tensor(np.load('data/mnist/train_whole/train_gt_labels.npy'))
        self.ann1_labels_train = torch.tensor(np.load('data/mnist/train_whole/train_ann1_labels.npy'))
        self.ann2_labels_train = torch.tensor(np.load('data/mnist/train_whole/train_ann2_labels.npy'))
        self.ann3_labels_train = torch.tensor(np.load('data/mnist/train_whole/train_ann3_labels.npy'))

        self.images_val = torch.tensor(np.load('data/mnist/test/test_data.npy'))
        self.gt_labels_val = torch.tensor(np.load('data/mnist/test/test_gt_labels.npy'))
        self.ann1_labels_val = torch.tensor(np.load('data/mnist/test/test_ann1_labels.npy'))
        self.ann2_labels_val = torch.tensor(np.load('data/mnist/test/test_ann2_labels.npy'))
        self.ann3_labels_val = torch.tensor(np.load('data/mnist/test/test_ann3_labels.npy'))  
        
        self.train_tensors = (self.images_train, self.gt_labels_train, self.ann1_labels_train, self.ann2_labels_train, self.ann3_labels_train)
        self.val_tensors = (self.images_val, self.gt_labels_val, self.ann1_labels_val, self.ann2_labels_val, self.ann3_labels_val)
            
    
    assert all(self.train_tensors[0].size(0) == tensor.size(0) for tensor in tensors)
    

  def __getitem__(self, index):
    transform_train = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    x_train = self.train_tensors[0]/255
    x_val = self.val_tensors[0]/255
    if self.train:
        if self.transform:
            img = transform_train(x_train)[index,:,:]
        else:
            img = x_train[index,:,:]

        gt_lbl = self.train_tensors[1][index]
        ann1_lbl = self.train_tensors[2][index]
        ann2_lbl = self.train_tensors[3][index]
        ann3_lbl = self.train_tensors[4][index]

    else:
        if self.transform:
            img = transform_train(x_val)[index,:,:]
        else:
            img = x_val[index,:,:]

        gt_lbl = self.val_tensors[1][index]
        ann1_lbl = self.val_tensors[2][index]
        ann2_lbl = self.val_tensors[3][index]
        ann3_lbl = self.val_tensors[4][index]

        return img, gt_lbl, ann1_lbl, ann2_lbl, ann3_lbl, index  
    
    def __len__(self):
        if self.train:
            return self.train_tensors[0].size(0)    
        else:
            return self.val_tensors[0].size(0) 


########################################## When data is in [channel, batch, height, weight] format
# class mnist_dataset(Data.Dataset):
#     def __init__(self, train=True, transform=None):
            
#         self.transform = transform
#         self.train = train 
        
#         self.images_train = np.load('data/mnist/train/train_data.npy')
#         self.gt_labels_train = np.load('data/mnist/train/train_gt_labels.npy')
#         self.ann1_labels_train = np.load('data/mnist/train/train_ann1_labels.npy')
#         self.ann2_labels_train = np.load('data/mnist/train/train_ann2_labels.npy')
#         self.ann3_labels_train = np.load('data/mnist/train/train_ann3_labels.npy')
        
#         self.images_val = np.load('data/mnist/val/val_data.npy')
#         self.gt_labels_val = np.load('data/mnist/val/val_gt_labels.npy')
#         self.ann1_labels_val = np.load('data/mnist/val/val_ann1_labels.npy')
#         self.ann2_labels_val = np.load('data/mnist/val/val_ann2_labels.npy')
#         self.ann3_labels_val = np.load('data/mnist/val/val_ann3_labels.npy')
    
#     def __getitem__(self, index):
#         transform_train = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize((32,32)),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
        
#         tr = transforms.ToTensor()
        
#         if self.train:
#             if self.transform:
#                 gt_img = transform_train(self.images_train[0, index,:,:])
#                 thin_img = transform_train(self.images_train[1, index,:,:])
#                 thick_img = transform_train(self.images_train[2, index,:,:])
#                 swell_img = transform_train(self.images_train[3, index,:,:])
#                 frac_img = transform_train(self.images_train[4, index,:,:])
#             else:
#                 gt_img = tr(self.images_train[0, index,:,:])
#                 thin_img = tr(self.images_train[1, index,:,:])
#                 thick_img = tr(self.images_train[2, index,:,:])
#                 swell_img = tr(self.images_train[3, index,:,:])
#                 frac_img = tr(self.images_train[4, index,:,:])

#             gt_lbl = self.gt_labels_train[:, index]
#             ann1_lbl = self.ann1_labels_train[:, index]
#             ann2_lbl = self.ann2_labels_train[:, index]
#             ann3_lbl = self.ann3_labels_train[:, index]

#         else:
#             if self.transform:
#                 gt_img = transform_train(self.images_val[0, index,:,:])
#                 thin_img = transform_train(self.images_val[1, index,:,:])
#                 thick_img = transform_train(self.images_val[2, index,:,:])
#                 swell_img = transform_train(self.images_val[3, index,:,:])
#                 frac_img = transform_train(self.images_val[4, index,:,:])
#             else:
#                 gt_img = tr(self.images_val[0, index,:,:])
#                 thin_img = tr(self.images_val[1, index,:,:])
#                 thick_img = tr(self.images_val[2, index,:,:])
#                 swell_img = tr(self.images_val[3, index,:,:])
#                 frac_img = tr(self.images_val[4, index,:,:])

#             gt_lbl = self.gt_labels_val[:, index]
#             ann1_lbl = self.ann1_labels_val[:, index]
#             ann2_lbl = self.ann2_labels_val[:, index]
#             ann3_lbl = self.ann3_labels_val[:, index]
            
# #         img = Image.fromarray(img)
        
#         imgs = gt_img
#         imgs = torch.vstack((imgs, thin_img))
#         imgs = torch.vstack((imgs, thick_img))
#         imgs = torch.vstack((imgs, swell_img))
#         imgs = torch.vstack((imgs, frac_img))
# #         gt_img = torch.squeeze(gt_img, dim=0)
        
    
# #         if self.target_transform is not None:
# #             label = self.target_transform(label)
        
#         return imgs, gt_lbl, ann1_lbl, ann2_lbl, ann3_lbl, index  
    
#     def __len__(self):
#         if self.train:
#             return len(self.gt_labels_train[0])
#         else:
#             return len(self.gt_labels_val[0])
 


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1. Code reproduced using [1] K. Yi and J. Wu, “Probabilistic end-to-end noise correction for learning with noisy labels,” Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit., vol. 2019-June, pp. 7010–7018, 2019, doi: 10.1109/CVPR.2019.00718."
    """
    assert P.shape[0] == P.shape[1]
    assert torch.max(y) < P.shape[0]
    
    npt.assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()
#     print("this is y:", y)
    m = y.shape[0]
    
    new_y = np.copy(y)
    flipper = np.random.RandomState(random_state)
    
    for idx in np.arange(m):
        i = y[idx]
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state, nb_classes=10):
    """mistakes: flip in the pair"""
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = np.mean(y_train_noisy != y_train)

        assert actual_noise > 0.0
        y_train = y_train_noisy
        y_train = torch.tensor(y_train)
    return y_train, P

def noisify_pairflip_permutation(y_train, noise, random_state, nb_classes=10):
    """mistakes: flip in the pair"""
#     torch.manual_seed(random_state)
#     y_train=y_train[torch.randperm(y_train.size()[0])]
    
#     P = np.eye(nb_classes)
#     n = noise

#     if n > 0.0:
#         # 0 -> 1
#         P[0, 0], P[0, 1] = 1. - n, n
#         for i in range(1, nb_classes-1):
#             P[i, i], P[i, i + 1] = 1. - n, n
#         P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n
    
    ori = list(range(10))
#     ori = random.choice(list(itertools.permutations(ori)))     #should be this, but to trace the Sms I am setting the below
    ori = list(itertools.permutations(ori))[random_state+1500000]
    print(ori, random_state)

    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[ori[0], ori[0]], P[ori[0], ori[1]] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[ori[i], ori[i]], P[ori[i], ori[i+1]] = 1. - n, n
        P[ori[nb_classes-1], ori[nb_classes-1]], P[ori[nb_classes-1], ori[0]] = 1. - n, n

    
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = np.mean(y_train_noisy != y_train)

        assert actual_noise > 0.0
        y_train = y_train_noisy
        y_train = torch.tensor(y_train)
    return y_train, P


def noisify_multiclass_symmetric(y_train, noise, random_state, nb_classes=10):
    """mistakes: flip in the symmetric way"""
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P
    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n
        
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = np.mean(y_train_noisy != y_train)

        assert actual_noise > 0.0
        y_train = y_train_noisy
        y_train = torch.tensor(y_train)
    return y_train, P


def noisify_asymmetric(y_train, noise, random_state, nb_classes=10):
    """mistakes: flip in the asymmetric way: random flip between two random classes"""
    P = np.eye(nb_classes)
    n = noise
#     cls1, cls2 = np.random.choice(range(nb_classes), size=2, replace=False)
    
    if n > 0.0:
        # 0 -> 1
        A = np.ones((5,5))*noise/4
        for i in range(5):
            A[i,i] = 1.0-noise
            P = np.block([
                [A,               np.zeros((5, 5))],
                [np.zeros((5, 5)), A               ]
            ])
    
#         P[cls1, cls2] = noise
#         P[cls2, cls1] = noise
#         P[cls1, cls1] = 1.0 - noise
#         P[cls2, cls2] = 1.0 - noise
        
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = np.mean(y_train_noisy != y_train)

        assert actual_noise > 0.0
        y_train = y_train_noisy
        y_train = torch.tensor(y_train)
    return y_train, P


def noisify(train_labels, noise_type, noise_rate, random_state, nb_classes=10):
    if noise_type == 'pairflip':
        train_noisy_labels, SM = noisify_pairflip(train_labels, noise_rate, random_state, nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, SM = noisify_multiclass_symmetric(train_labels, noise_rate, random_state, nb_classes)
    if noise_type == 'asymmetric':
        train_noisy_labels, SM = noisify_asymmetric(train_labels, noise_rate, random_state, nb_classes)
    if noise_type == 'pairflip_permut':
        train_noisy_labels, SM = noisify_pairflip_permutation(train_labels, noise_rate, random_state, nb_classes)
    return train_noisy_labels, SM

                                               

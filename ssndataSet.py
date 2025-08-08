import pandas as pd
from sklearn.preprocessing import StandardScaler 
from torch.utils.data import Dataset
import random
import torch


class MyDateSet(Dataset):

    def __init__(self, root_dir, model, transform=None):
        super(MyDateSet, self).__init__()

        self.root_dir = root_dir
        self.model = model
        self.transform = transform

        myAnnotion = pd.read_excel(self.root_dir).values[:, 1:]

        ratio = [8, 1, 1] 
        the_index_random = [i for i in range(myAnnotion.shape[0])]
        random.shuffle(the_index_random)

        the_index_random_train = the_index_random[:int(myAnnotion.shape[0]/sum(ratio)*ratio[0])]
        the_index_random_val = the_index_random[int(myAnnotion.shape[0]/sum(ratio)*ratio[0]):int(myAnnotion.shape[0]/sum(ratio)*(ratio[0]+ratio[1]))]
        the_index_random_test = the_index_random[int(myAnnotion.shape[0]/sum(ratio)*(ratio[0]+ratio[1])):]

        if self.model == 'Train':
            myAnnotion = myAnnotion[the_index_random_train]
        elif self.model == 'Val':
            myAnnotion = myAnnotion[the_index_random_val]
        elif self.model == 'Test':
            myAnnotion = myAnnotion[the_index_random_test]
        else:
            raise('不存在'+str(self.model))
        
        self.num_data = myAnnotion.shape[0]
        
        self.label_huangtong = myAnnotion[:, 2]
        self.label_zaogan = myAnnotion[:, 3]
        self.label_duotang = myAnnotion[:, 4]
        
        import numpy as np
      
        self.data_leaf = myAnnotion[:, 5:5+204]
        scaler = StandardScaler()
        self.data_leaf = scaler.fit_transform(self.data_leaf)

        x = np.linspace(397.32, 1003.58, 204)
        dx = np.diff(x)
        self.derivative_spectra = np.diff(self.data_leaf) / dx[0] 
        
        from scipy.interpolate import UnivariateSpline
        wavelength = [i for i in range(self.data_leaf.shape[1])]
        for i, info_i in enumerate(self.data_leaf):
            spline = UnivariateSpline(wavelength, info_i, k=3)
            continuum = spline(wavelength)
            self.data_leaf[i] = info_i - continuum

        import pywt
        self.cA, self.cD = pywt.dwt(self.data_leaf, 'haar')
        self.cA1, self.cD1 = pywt.dwt(self.derivative_spectra, 'haar')

    def __getitem__(self, index):
        label = torch.FloatTensor([self.label_huangtong[index]/12, self.label_zaogan[index]/4.1, self.label_duotang[index]/60])
        cA = torch.FloatTensor(self.cA[index])
        cD = torch.FloatTensor(self.cD[index])
        cA1 = torch.FloatTensor(self.cA1[index])
        cD1 = torch.FloatTensor(self.cD1[index])

        return cA, cD, cA1, cD1, label


    def __len__(self):
        return self.num_data
    



    
           
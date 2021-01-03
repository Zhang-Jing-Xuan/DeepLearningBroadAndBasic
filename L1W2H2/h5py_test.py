import h5py
import numpy as np
f=h5py.File('/Users/admin/Desktop/CL/Python/DeapLeaning/L1W2H2/datasets/train_catvnoncat2.h5','r')
print('-----根目录-----')
for key in f.keys():
    print(f[key].name)
print('-----/train_set_x目录-----')
g1=f['/train_set_x'][:]
print(g1.shape)
print('-----/train_set_y目录-----')
g1=f['/train_set_y'][:]
print(g1)
np.insert(g1,1)
print(g1)
import os
import glob
from data import common
import numpy as np
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.scale = args.scale
        # self.arr = self.get_arr()
        self.arr = np.fromfile('data/arr.dat', dtype=int)

        data_range = [r.split('-') for r in args.data_range.split('/')] # ex: self.data_range: 1-400/401-432
        data_range = data_range[0] if train else data_range[1]
        self.begin, self.end = list(map(lambda x: int(x), data_range))

        self._set_filesystem(args.dir_data_root)  # args.dir_data_root: /home/xxx/xxx/data/
        self.images_hr, self.images_lr = self._scan() # get file list

        if train:
            n_patches = args.batch_size * args.test_every  # args.batch_size: 16, args.test_every: 1000
            self.repeat = max(n_patches // len(self.images_hr), 1) if len(self.images_hr) > 0 else 0


    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(glob.glob(os.path.join(self.dir_hr, '*.dat')))
        names_lr = sorted(glob.glob(os.path.join(self.dir_lr, '*.dat')))
        
        names_hr = np.array(names_hr)[self.arr] # shuffle
        names_hr = names_hr[self.begin - 1: self.end]

        if self.args.apply_field_data:
            names_lr = np.array(names_lr)
        else:
            names_lr = np.array(names_lr)[self.arr]
            names_lr = names_lr[self.begin - 1: self.end]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data_root):
        self.root_path = dir_data_root
        self.dir_hr = os.path.join(self.root_path, self.args.dir_hr)
        if not self.args.apply_field_data:
            self.dir_lr = os.path.join(self.root_path, self.args.dir_lr)
        else:
            self.dir_lr = self.args.dir_lr

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair, params = common.normal(lr, hr)
        pair = common.set_channel(*pair)

        pair = self.get_patch(*pair)
        pair_t = common.np2Tensor(*pair)

        return pair_t[0], pair_t[1], filename, params

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _load_file(self, idx):
        idx = idx % len(self.images_hr) if self.train else idx
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_lr)) # without suffix

        lr = np.fromfile(f_lr, dtype=np.float32)
        hr = np.fromfile(f_hr, dtype=np.float32)
        if not self.args.apply_field_data:
            lr = lr.reshape((128,128))
        else:
            shape = [int(x) for x in filename.split('_')[1].split('x')]
            lr = lr.reshape(shape)
        hr = hr.reshape((256,256))

        lr = np.rot90(lr, 3)
        hr = np.rot90(hr, 3)

        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale
            )
            lr, hr = common.augment(lr, hr)

        return lr, hr
    
    def get_arr(self):
        names_hr = sorted(glob.glob(os.path.join(self.dir_hr, '*.dat')))
        l = len(names_hr)

        arr = np.arange(l)
        np.random.shuffle(arr)
        return arr


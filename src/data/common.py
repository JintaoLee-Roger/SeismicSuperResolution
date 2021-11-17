import random
import numpy as np
import torch

def get_patch(*args, patch_size=48, scale=2):
    ih, iw = args[0].shape[:2]

    ip = patch_size
    tp = ip * scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    tx, ty = scale * ix, scale * iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret

def set_channel(*args):
    return [np.expand_dims(a, axis=2) for a in args]

def np2Tensor(*args):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args]

def normal(lr, hr):
    ma = max(hr.max(), lr.max())
    mi = min(hr.min(), lr.min())
    lr = (lr - mi) / (ma - mi)
    hr = (hr - mi) / (ma - mi)

    return [lr, hr], [ma, mi]


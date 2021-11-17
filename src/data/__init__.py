from data import srdata
from torch.utils.data import dataloader

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            train_set = srdata.SRData(args, train=True)
            self.loader_train = dataloader.DataLoader(
                    train_set, batch_size=args.batch_size,
                    shuffle=True, pin_memory=not args.cpu,
                    num_workers=args.n_threads
                    )

        test_set = srdata.SRData(args, train=False)
        self.loader_test =  dataloader.DataLoader(
                test_set, batch_size=1,
                shuffle=False, pin_memory=not args.cpu,
                num_workers=args.n_threads
                )


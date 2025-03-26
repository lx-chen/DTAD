from datasets import mvtecad
from torch.utils.data import DataLoader
from dataloaders.utlis import worker_init_fn_seed, BalancedBatchSampler

class initDataloader():

    @staticmethod
    def build(args, **kwargs):
        if args.dataset == "mvtecad":
            normal_set = mvtecad.MVTecAD(args, train_type="normal_set", train=True)
            train_set = mvtecad.MVTecAD(args, train_type="train_set", train=True)
            test_set = mvtecad.MVTecAD(args, train_type="test_set", train=False)
            
            normal_loader = DataLoader(normal_set,
                                      worker_init_fn = worker_init_fn_seed,
                                      batch_sampler = BalancedBatchSampler(args, normal_set, train_type="normal_set"),
                                      **kwargs)
            train_loader = DataLoader(train_set,
                                      worker_init_fn = worker_init_fn_seed,
                                      batch_sampler = BalancedBatchSampler(args, train_set, train_type="train_set"),
                                      **kwargs)
            test_loader = DataLoader(test_set,
                                     batch_size=args.batch_size,
                                    #  batch_size = 1,
                                     shuffle=False,
                                     worker_init_fn= worker_init_fn_seed,
                                     **kwargs)
            return normal_loader, train_loader, test_loader


        else:
            raise NotImplementedError



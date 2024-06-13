from data_provider.data_loader import Dataset_ETT_hour, Dataset_Custom, Dataset_M4, Dataset_Solar, Dataset_TSF, Dataset_TSF_ICL, Dataset_ECG, Dataset_ECG_multi
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

data_dict = {
    'ECG': Dataset_ECG
}

def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size 
    elif flag == 'val':
        shuffle_flag = args.val_set_shuffle
        drop_last = False
        batch_size = args.batch_size 
    else:
        shuffle_flag = True
        drop_last = args.drop_last
        batch_size = args.batch_size

    if flag in ['train', 'val']:
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.token_len]
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            size=[args.test_seq_len, args.test_label_len, args.test_pred_len]
        )


    if args.use_multi_gpu:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(data_set, 
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last,
            )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
    return data_set, data_loader

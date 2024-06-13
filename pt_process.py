import argparse
import os
import torch
from models.Preprocess_Llama import Model
from data_provider.data_loader import Dataset_PT_process
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoTimes Preprocess')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--llm_ckp_dir', type=str, default='/home/rongqin/dragonfly/Llama3_ecg', help='llm checkpoints dir')
    parser.add_argument('--dataset', type=str, default='ECG_CSV', 
                        help='dataset to preprocess, options:[ETTh1, electricity, weather, traffic, ECG_CSV]')
    args = parser.parse_args()
    print(args.dataset)
    
    model = Model(args)

    seq_len = 576
    label_len = 512
    pred_len = 64

    root_path = './dataset/ECG_CSV/'
    data_files = [f for f in os.listdir(root_path) if f.endswith('.csv')]



    for data_file in tqdm(data_files):
        data_set = Dataset_PT_process(
            root_path=root_path,
            data_path=data_file,
            size=[seq_len, label_len, pred_len]
        )

        data_loader = DataLoader(
            data_set,
            batch_size=128,
            shuffle=False,
        )

        output_list = []

        for idx, data in tqdm(enumerate(data_loader), desc=f'Processing {data_file}'):
            output = model(data)
            output_list.append(output.detach().cpu())

        result = torch.cat(output_list, dim=0)
        print(f'{data_file} result shape: {result.shape}')
        ecg_file = 'ECG'
        torch.save(result, os.path.join(root_path, f'{ecg_file}.pt'))

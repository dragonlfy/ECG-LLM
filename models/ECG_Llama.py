import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from layers.mlp import MLP

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)
        
        self.llama = LlamaForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
        )
        self.hidden_dim_of_llama = 4096
        self.mix = configs.mix_embeds
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))
        
        for name, param in self.llama.named_parameters():
            param.requires_grad = False

        self.encoder = nn.Sequential(
                nn.Conv1d(in_channels=self.token_len, out_channels=self.hidden_dim_of_llama, kernel_size=1)
            )
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim_of_llama, out_channels=self.token_len, kernel_size=1)
        )
    
        # self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_llama)
        # self.projection = nn.Linear(self.hidden_dim_of_llama, self.token_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()    
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        bs, _, n_vars = x_enc.shape
        # x_enc: [bs x nvars x seq_len]
        x_enc = x_enc.permute(0, 2, 1)
        # x_enc: [bs * nvars x seq_len]
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
        # fold_out: [bs * n_vars x token_num x token_len]
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]
        # times_embeds: [bs * n_vars x token_num x hidden_dim_of_llama]
        fold_out = fold_out.permute(0, 2, 1)  # [bs * n_vars x token_len x token_num] for Conv1d
        times_embeds = self.encoder(fold_out)
        times_embeds = times_embeds.permute(0, 2, 1) 
        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * x_mark_enc
        # outputs: [bs * n_vars x token_num x hidden_dim_of_llama]
        outputs = self.llama.model(
            inputs_embeds=times_embeds)[0]
        # dec_out: [bs * n_vars x token_num x token_len]
        dec_out = self.projection(outputs.permute(0, 2, 1))  # [bs * n_vars x hidden_dim_of_llama x token_num] for Conv1d
        dec_out = dec_out.permute(0, 2, 1)  # back to [bs * n_vars x token_num x token_len]
        dec_out = dec_out.reshape(bs, n_vars, -1)
        # dec_out: [bs x token_num * token_len x n_vars]
        dec_out = dec_out.permute(0, 2, 1)
        
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

import torch
from transformers import OpenAIGPTModel, OpenAIGPTConfig
import torch.nn as nn
import numpy as np

class GPT(nn.Module):
    def __init__(self, input_size=(2560),
                 action_shape=0,
                 concat=False,
                 num_atoms=1,
                 n_embd=768,
                 n_layer=12,
                 n_head=12,
                 device='cpu'):
        '''
        Args:
            input_size: input size
            n_embd: embedding size
            n_layer: num of transformer layer
            n_head: num of self attention head
        '''
        super().__init__()
        self.device = device
        input_size= int(np.prod(input_size))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_size += action_dim

        self.emb_size = n_embd
        config = OpenAIGPTConfig(n_embd=n_embd, n_layer=n_layer, n_head=n_head)
        self.model = OpenAIGPTModel(config)
        self.embedding = nn.Linear(input_size, n_embd)
        self.output = nn.Linear(n_embd, 40)
        self.output_dim=40
        self.to(self.device)


    def forward(self, x,
                state= None,
    ):
        x = self.embedding(x)
        res = self.model(inputs_embeds=x, return_dict=True)
        emb = res['last_hidden_state']
        logits = self.output(emb)
        return logits,state

if __name__ == '__main__':
    gpt = GPT()
    gpt.eval()
    state1 = torch.randn([3, 32, 2560])
    state2 = state1[:, 0, :].unsqueeze(1)
    print(gpt(state1))
    print(gpt(state2))




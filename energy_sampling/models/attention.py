import torch
from torch import nn


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False, target='velocity'):
        super().__init__()
        self.time_varying = time_varying
        self.target = target
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)



class AtomAttention(nn.Module):
    def __init__(self, embeddings_dim=3, t_dim=32, output_dim=11, hidden_dim=32, kernel_size=9):
        super(AtomAttention, self).__init__()
        self.atom_num = kernel_size
        self.atom_type_proj = nn.Embedding(100, embeddings_dim)
        self.feature_convolution = nn.Conv1d(embeddings_dim*2+t_dim, hidden_dim, 1, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim*2+t_dim, hidden_dim, 1, stride=1,
                                               padding=kernel_size // 2)
        self.circular_pad = nn.CircularPad1d(padding = kernel_size // 2)
        self.softmax = nn.Softmax(dim=-1)

        
        self.linear = MLP(dim=2*hidden_dim, out_dim=2*hidden_dim, w=2*hidden_dim)
        self.output = MLP(dim=2*hidden_dim, out_dim=output_dim, w=2*hidden_dim)

    def forward(self, x: torch.Tensor, t, atom_types, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        atom_emb = self.atom_type_proj(atom_types)
        atom_emb = atom_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = x.view(-1, self.atom_num, 3).permute(0, 2, 1).to(self.feature_convolution.weight.device)

        t = t.unsqueeze(2)
        x = torch.cat([x, t.repeat(1, 1, x.shape[2]), atom_emb.permute(0, 2, 1)], dim=1)
        
        x_padded = self.circular_pad(x)
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]


        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]


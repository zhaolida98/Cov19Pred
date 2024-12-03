import torch
import torch.nn.functional as F
from torch import nn


class RnnModel(nn.Module):
    """
    An RNN model using either RNN, LSTM or GRU cells.
    """

    def __init__(self, input_dim, output_dim, hidden_size, dropout_p, cell_type):
        super(RnnModel, self).__init__()

        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.cell_type = cell_type

        self.dropout = nn.Dropout(dropout_p)

        if cell_type == 'LSTM':
            self.encoder = nn.LSTM(input_dim, hidden_size)
        elif cell_type == 'GRU':
            self.encoder = nn.GRU(input_dim, hidden_size)
        elif cell_type == 'RNN':
            self.encoder = nn.RNN(input_dim, hidden_size)

        self.out = nn.Linear(hidden_size, output_dim)

    def forward(self, input_seq, hidden_state):
        input_seq = self.dropout(input_seq)
        encoder_outputs, _ = self.encoder(input_seq, hidden_state)
        score_seq = self.out(encoder_outputs[-1, :, :])

        dummy_attn_weights = torch.zeros(input_seq.shape[1], input_seq.shape[0])
        return score_seq, dummy_attn_weights  # No attention weights

    def init_hidden(self, batch_size):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.cell_type == 'LSTM':
            h_init = torch.zeros(1, batch_size, self.hidden_size).to(device)
            c_init = torch.zeros(1, batch_size, self.hidden_size).to(device)

            return (h_init, c_init)
        elif self.cell_type == 'GRU':
            return torch.zeros(1, batch_size, self.hidden_size).to(device)
        elif self.cell_type == 'RNN':
            return torch.zeros(1, batch_size, self.hidden_size).to(device)


class AttentionModel(nn.Module):
    """
    A temporal attention model using an LSTM encoder.
    """

    def __init__(self, seq_length, input_dim, output_dim, hidden_size, dropout_p):
        super(AttentionModel, self).__init__()

        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.output_dim = output_dim

        self.encoder = nn.LSTM(input_dim, hidden_size)
        self.attn = nn.Linear(hidden_size, seq_length)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_size, output_dim)

    def forward(self, input_seq, hidden_state):
        input_seq = self.dropout(input_seq)
        encoder_outputs, (h, _) = self.encoder(input_seq, hidden_state)
        attn_applied, attn_weights = self.attention(encoder_outputs, h)
        score_seq = self.out(attn_applied.reshape(-1, self.hidden_size))

        return score_seq, attn_weights

    def attention(self, encoder_outputs, hidden):
        attn_weights = F.softmax(torch.squeeze(self.attn(hidden)), dim=1)
        attn_weights = torch.unsqueeze(attn_weights, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)

        return attn_applied, torch.squeeze(attn_weights)

    def init_hidden(self, batch_size):
        h_init = torch.zeros(1, batch_size, self.hidden_size)
        c_init = torch.zeros(1, batch_size, self.hidden_size)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h_init = h_init.to(device)
        c_init = c_init.to(device)
        return (h_init, c_init)


class DaRnnModel(nn.Module):
    """
    A Dual-Attention RNN model, attending over both the input at each timestep
    and all hidden states of the encoder to make the final prediction.
    """

    def __init__(self, seq_length, input_dim, output_dim, hidden_size, dropout_p):
        super(DaRnnModel, self).__init__()

        self.n = input_dim
        self.m = hidden_size
        self.T = seq_length
        self.output_dim = output_dim

        self.dropout = nn.Dropout(dropout_p)

        self.encoder = nn.LSTM(self.n, self.m)

        self.We = nn.Linear(2 * self.m, self.T)
        self.Ue = nn.Linear(self.T, self.T)
        self.ve = nn.Linear(self.T, 1)

        self.Ud = nn.Linear(self.m, self.m)
        self.vd = nn.Linear(self.m, 1)
        self.out = nn.Linear(self.m, output_dim)

    def forward(self, x, hidden_state):
        x = self.dropout(x)
        h_seq = []
        for t in range(self.T):
            x_tilde, _ = self.input_attention(x, hidden_state, t)
            ht, hidden_state = self.encoder(x_tilde, hidden_state)
            h_seq.append(ht)

        h = torch.cat(h_seq, dim=0)
        c, beta = self.temporal_attention(h)
        logits = self.out(c)

        return logits, torch.squeeze(beta)

    def input_attention(self, x, hidden_state, t):
        x = x.permute(1, 2, 0)
        h, c = hidden_state
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)
        hc = torch.cat([h, c], dim=2)

        e = self.ve(torch.tanh(self.We(hc) + self.Ue(x)))
        e = torch.squeeze(e)
        alpha = F.softmax(e, dim=1)
        xt = x[:, :, t]

        x_tilde = alpha * xt
        x_tilde = torch.unsqueeze(x_tilde, 0)

        return x_tilde, alpha

    def temporal_attention(self, h):
        h = h.permute(1, 0, 2)
        l = self.vd(torch.tanh((self.Ud(h))))
        l = torch.squeeze(l)
        beta = F.softmax(l, dim=1)
        beta = torch.unsqueeze(beta, 1)
        c = torch.bmm(beta, h)
        c = torch.squeeze(c)

        return c, beta

    def init_hidden(self, batch_size):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h_init = torch.zeros(1, batch_size, self.m).to(device)
        c_init = torch.zeros(1, batch_size, self.m).to(device)

        return (h_init, c_init)


class TransformerModel(nn.Module):
    """
    A temporal attention model using an Transformer encoder.
    """

    def __init__(self, input_dim, output_dim, dropout_p, nhead=5):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim  # 100
        self.output_dim = output_dim  # 2
        self.hidden_size = 128
        self.dropout = nn.Dropout(dropout_p)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fnn = nn.Linear(input_dim, output_dim)

    def forward(self, input_seq, hidden_state):
        out = self.dropout(input_seq)
        out = self.transformer_encoder(out)

        # out = torch.sum(out, 0)
        out = out[-1, :, :]

        out = self.fnn(out)
        return out, out

    def init_hidden(self, batch_size):
        h_init = torch.zeros(1, batch_size, self.hidden_size)
        c_init = torch.zeros(1, batch_size, self.hidden_size)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h_init = h_init.to(device)
        c_init = c_init.to(device)
        return (h_init, c_init)

class PosTrans(nn.Module):
    class Head(nn.Module):
        def __init__(self, n_embed, head_size, dropout):
            super().__init__()
            self.key = nn.Linear(n_embed, head_size)
            self.query = nn.Linear(n_embed, head_size)
            self.value = nn.Linear(n_embed, head_size)
            self.dropout = nn.Dropout(dropout)

        def forward(self, idx):
            B, T, C = idx.shape
            k = self.key(idx)  # (B, T, C)
            q = self.query(idx)  # (B, T, C)
            weight = q @ k.transpose(-2, -1) / (C ** 0.5)  # (B, T, T)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            tril = torch.tril(torch.ones(T, T)).to(device)
            weight = weight.masked_fill(tril == 0, float('-inf'))
            weight = F.softmax(weight, dim=-1)
            weight = self.dropout(weight)
            v = self.value(idx)  # (B, T, C)
            out = weight @ v  # (B, T, C)
            return out

    class MultiHead(nn.Module):
        def __init__(self, n_embed, head_size, head_num, dropout):
            super().__init__()
            self.heads = nn.ModuleList(
                [PosTrans.Head(n_embed, head_size, dropout) for _ in range(head_num)]
            )
            self.proj = nn.Linear(n_embed, n_embed)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.proj(out)
            out = self.dropout(out)
            return out

    class Block(nn.Module):
        def __init__(self, n_embed, head_num, head_size, dropout):
            super().__init__()
            self.sa = PosTrans.MultiHead(n_embed, head_size, head_num, dropout)
            self.ffw = PosTrans.FFwd(n_embed, dropout)
            self.ln1 = nn.LayerNorm(n_embed)
            self.ln2 = nn.LayerNorm(n_embed)

        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffw(self.ln2(x))
            return x

    class FFwd(nn.Module):
        def __init__(self, n_embed, dropout):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Linear(n_embed, 4 * n_embed),
                nn.GELU(),
                nn.Linear(4 * n_embed, n_embed),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.seq(x)

    def __init__(self, input_dim, output_dim, dropout, head_num=4, n_layer=6):
        super().__init__()
        self.n_embed = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.head_size = self.n_embed // head_num
        self.blocks = nn.Sequential(
            *[
                PosTrans.Block(
                    n_embed=self.n_embed,
                    head_num=head_num,
                    head_size=self.head_size,
                    dropout=self.dropout,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(self.n_embed)
        self.ffwd = PosTrans.FFwd(self.n_embed, self.dropout)
        self.lm_head = nn.Linear(self.n_embed, self.output_dim)

    def forward(self, idx, hidden, targets=None):
        x = self.blocks(idx)  # (B, T, n_embed)
        x = self.ln_f(x)
        x = self.ffwd(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        logits = logits[-1, :, :]  # Get the last token's prediction
        logits = logits.squeeze(-1)
        if targets is not None:
            loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        else:
            loss = None
        return logits, loss

    def init_hidden(self, batch_size):
        h_init = torch.zeros(1, batch_size, self.n_embed)
        c_init = torch.zeros(1, batch_size, self.n_embed)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h_init = h_init.to(device)
        c_init = c_init.to(device)
        return (h_init, c_init)
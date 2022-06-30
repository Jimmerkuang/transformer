import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEncoding(nn.Module):

    def __init__(self, embed_dim, seq_len):
        super(PositionEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.position_encoding = self._position_encoding()

    def _position_encoding(self):
        position_encoding = torch.zeros((self.seq_len, self.embed_dim))
        for i in range(position_encoding.shape[0]):
            for j in range(position_encoding.shape[1]):
                pos = i / (10000 ** (2 * j / self.embed_dim))
                position_encoding[i][j] = math.sin(pos) if j % 2 == 0 else math.cos(pos)
        return position_encoding

    def forward(self, x):
        x = x + self.position_encoding.to(x.device)
        return x


class AddNorm(nn.Module):

    def __init__(self, embed_dim, seq_len):
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm([seq_len, embed_dim])

    def forward(self, x, sub_layer, **kwargs):
        sub_output = sub_layer(x, **kwargs)
        x = self.layer_norm(x + sub_output)
        return x


class FeedForward(nn.Module):

    def __init__(self, embed_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EncoderMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(EncoderMultiheadAttention, self).__init__()
        self.multihead_atten = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x, _ = self.multihead_atten(x, x, x)
        return x.permute(1, 0, 2)


class DecoderMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(DecoderMultiheadAttention, self).__init__()
        self.multihead_atten = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    @staticmethod
    def _generate_mark(dim):
        matrix = np.ones((dim, dim))
        mask = torch.Tensor(np.tril(matrix))
        return mask == 0

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x, _ = self.multihead_atten(x, x, x, attn_mask=self.mask.to(x.device))
        return x.permute(1, 2, 0)


class CrossMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(CrossMultiheadAttention, self).__init__()
        self.cross_atten = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x, encoder_output):
        x, _ = self.cross_atten(encoder_output.permute(1, 0, 2), x.permute(1, 0, 2), x.permute(1, 0, 2))
        return x.permute(1, 0, 2)


class Encoder(nn.Module):

    def __init__(self, embed_dim, seq_len, num_heads):
        super(Encoder, self).__init__()
        self.position_encoding = PositionEncoding(embed_dim, seq_len)
        self.multihead_atten = EncoderMultiheadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim)
        self.add_norm = AddNorm(embed_dim, seq_len)

    def forward(self, x):
        x = self.position_encoding(x)
        x = self.add_norm(x, self.multihead_atten)
        x = self.add_norm(x, self.feed_forward)
        return x


class Decoder(nn.Module):

    def __init__(self, embed_dim, seq_len, num_heads):
        super(Decoder, self).__init__()
        self.position_encoding = PositionEncoding(embed_dim, seq_len)
        self.multihead_atten = EncoderMultiheadAttention(embed_dim, num_heads)
        self.cross_atten = CrossMultiheadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim)
        self.add_norm = AddNorm(embed_dim, seq_len)

    def forward(self, x, encoder_output):
        x = self.position_encoding(x)
        x = self.add_norm(x, self.multihead_atten)
        x = self.add_norm(x, self.cross_atten, encoder_output=encoder_output)
        decoder_output = self.add_norm(x, self.feed_forward)
        return decoder_output


class Transformer(nn.Module):

    def __init__(self, embed_dim=2393, seq_len=20, num_heads=1):
        super(Transformer, self).__init__()
        self.encoder = nn.Sequential(*[Encoder(embed_dim, seq_len, num_heads) for _ in range(2)])
        self.decoder = nn.Sequential(*[Decoder(embed_dim, seq_len, num_heads) for _ in range(2)])
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        encoder_output = self.encoder(x)
        # decoder_output = self.decoder(x, encoder_output)
        x = F.leaky_relu(self.fc1(encoder_output))
        x = self.fc2(x)
        return x.view(-1, 20)


if __name__ == '__main__':
    input = torch.rand(1, 20, 2393)
    transformer = Transformer()
    output = transformer(input)

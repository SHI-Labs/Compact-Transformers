import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformers import TransformerEncoderLayer


__all__ = ['cct_2', 'cct_4', 'cct_6', 'cct_7', 'cct_8',
           'cct_10', 'cct_12', 'cct_24', 'cct_32',
           'cvt_2', 'cvt_4', 'cvt_6', 'cvt_7', 'cvt_8',
           'cvt_10', 'cvt_12', 'cvt_24', 'cvt_32',
           'vit_lite_2', 'vit_lite_4', 'vit_lite_6',
           'vit_lite_7', 'vit_lite_8', 'vit_lite_10',
           'vit_lite_12', 'vit_lite_24', 'vit_lite_32']


class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True):
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=False),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class TransformerClassifier(nn.Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout_rate=0.1,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 positional_embedding='sine',
                 sequence_length=None,
                 *args, **kwargs):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim),
                                          requires_grad=True)
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                   requires_grad=True)
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = self.sinusoidal_embedding(sequence_length, embedding_dim)
        else:
            self.positional_emb = None

        self.dropout = nn.Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = nn.LayerNorm(embedding_dim)

        self.fc = nn.Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class ViTLite(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 patch_size=16,
                 *args, **kwargs):
        super(ViTLite, self).__init__()
        assert img_size % patch_size == 0, f"Image size ({img_size}) has to be" \
                                           f"divisible by patch size ({patch_size})"
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=patch_size,
                                   stride=patch_size,
                                   padding=0,
                                   max_pool=False,
                                   activation=None,
                                   n_conv_layers=1)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            seq_pool=False,
            dropout=0.1,
            attention_dropout=0.,
            stochastic_depth=0.,
            *args, **kwargs)

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


class CVT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 patch_size=16,
                 *args, **kwargs):
        super(CVT, self).__init__()
        assert img_size % patch_size == 0, f"Image size ({img_size}) has to be" \
                                           f"divisible by patch size ({patch_size})"
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=patch_size,
                                   stride=patch_size,
                                   padding=0,
                                   max_pool=False,
                                   activation=None,
                                   n_conv_layers=1)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            seq_pool=True,
            dropout=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs)

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


class CCT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 *args, **kwargs):
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            seq_pool=True,
            dropout=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs)

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


def _cct(num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None, padding=None,
         *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    return CCT(num_layers=num_layers,
               num_heads=num_heads,
               mlp_ratio=mlp_ratio,
               embedding_dim=embedding_dim,
               kernel_size=kernel_size,
               stride=stride,
               padding=padding,
               *args, **kwargs)


def _cvt(num_layers, num_heads, mlp_ratio, embedding_dim,
         patch_size=4, *args, **kwargs):
    return CVT(num_layers=num_layers,
               num_heads=num_heads,
               mlp_ratio=mlp_ratio,
               embedding_dim=embedding_dim,
               patch_size=patch_size,
               *args, **kwargs)


def _vit_lite(num_layers, num_heads, mlp_ratio, embedding_dim,
              patch_size=4, *args, **kwargs):
    return ViTLite(num_layers=num_layers,
                   num_heads=num_heads,
                   mlp_ratio=mlp_ratio,
                   embedding_dim=embedding_dim,
                   patch_size=patch_size,
                   *args, **kwargs)


def cct_2(*args, **kwargs):
    return _cct(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_4(*args, **kwargs):
    return _cct(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_6(*args, **kwargs):
    return _cct(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_7(*args, **kwargs):
    return _cct(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_8(*args, **kwargs):
    return _cct(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_10(*args, **kwargs):
    return _cct(num_layers=10, num_heads=8, mlp_ratio=3, embedding_dim=512,
                *args, **kwargs)


def cct_12(*args, **kwargs):
    return _cct(num_layers=12, num_heads=12, mlp_ratio=4, embedding_dim=768,
                *args, **kwargs)


def cct_24(*args, **kwargs):
    return _cct(num_layers=24, num_heads=16, mlp_ratio=4, embedding_dim=1024,
                *args, **kwargs)


def cct_32(*args, **kwargs):
    return _cct(num_layers=32, num_heads=16, mlp_ratio=4, embedding_dim=1280,
                *args, **kwargs)


def cvt_2(*args, **kwargs):
    return _cvt(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cvt_4(*args, **kwargs):
    return _cvt(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cvt_6(*args, **kwargs):
    return _cvt(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cvt_7(*args, **kwargs):
    return _cvt(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cvt_8(*args, **kwargs):
    return _cvt(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cvt_10(*args, **kwargs):
    return _cvt(num_layers=10, num_heads=8, mlp_ratio=3, embedding_dim=512,
                *args, **kwargs)


def cvt_12(*args, **kwargs):
    return _cvt(num_layers=12, num_heads=12, mlp_ratio=4, embedding_dim=768,
                *args, **kwargs)


def cvt_24(*args, **kwargs):
    return _cvt(num_layers=24, num_heads=16, mlp_ratio=4, embedding_dim=1024,
                *args, **kwargs)


def cvt_32(*args, **kwargs):
    return _cvt(num_layers=32, num_heads=16, mlp_ratio=4, embedding_dim=1280,
                *args, **kwargs)


def vit_lite_2(*args, **kwargs):
    return _vit_lite(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                     *args, **kwargs)


def vit_lite_4(*args, **kwargs):
    return _vit_lite(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                     *args, **kwargs)


def vit_lite_6(*args, **kwargs):
    return _vit_lite(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                     *args, **kwargs)


def vit_lite_7(*args, **kwargs):
    return _vit_lite(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                     *args, **kwargs)


def vit_lite_8(*args, **kwargs):
    return _vit_lite(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                     *args, **kwargs)


def vit_lite_10(*args, **kwargs):
    return _vit_lite(num_layers=10, num_heads=8, mlp_ratio=3, embedding_dim=512,
                     *args, **kwargs)


def vit_lite_12(*args, **kwargs):
    return _vit_lite(num_layers=12, num_heads=12, mlp_ratio=4, embedding_dim=768,
                     *args, **kwargs)


def vit_lite_24(*args, **kwargs):
    return _vit_lite(num_layers=24, num_heads=16, mlp_ratio=4, embedding_dim=1024,
                     *args, **kwargs)


def vit_lite_32(*args, **kwargs):
    return _vit_lite(num_layers=32, num_heads=16, mlp_ratio=4, embedding_dim=1280,
                     *args, **kwargs)

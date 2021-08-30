import torch.nn as nn
from .utils.transformers import TransformerClassifier, MaskedTransformerClassifier
from .utils.tokenizer import Tokenizer, TextTokenizer
from .utils.embedder import Embedder

__all__ = ['cct_2', 'cct_4', 'cct_6', 'cct_7', 'cct_8',
           'cct_14', 'cct_16',
           'text_cct_2', 'text_cct_4', 'text_cct_6'
           ]


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
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs)

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


class TextCCT(nn.Module):
    def __init__(self,
                 seq_len=64,
                 word_embedding_dim=300,
                 embedding_dim=256,
                 kernel_size=2,
                 stride=1,
                 padding=1,
                 pooling_kernel_size=2,
                 pooling_stride=2,
                 pooling_padding=1,
                 *args, **kwargs):
        super(TextCCT, self).__init__()

        self.embedder = Embedder(word_embedding_dim=word_embedding_dim,
                                 *args, **kwargs)

        self.tokenizer = TextTokenizer(n_input_channels=word_embedding_dim,
                                       n_output_channels=embedding_dim,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       pooling_kernel_size=pooling_kernel_size,
                                       pooling_stride=pooling_stride,
                                       pooling_padding=pooling_padding,
                                       max_pool=True,
                                       activation=nn.ReLU)

        self.classifier = MaskedTransformerClassifier(
            seq_len=self.tokenizer.seq_len(seq_len=seq_len, embed_dim=word_embedding_dim),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs)

    def forward(self, x, mask=None):
        x, mask = self.embedder(x, mask=mask)
        x, mask = self.tokenizer(x, mask=mask)
        out = self.classifier(x, mask=mask)
        return out


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


def _text_cct(num_layers, num_heads, mlp_ratio, embedding_dim,
              kernel_size=4, stride=None, padding=None,
              *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))

    return TextCCT(num_layers=num_layers,
                   num_heads=num_heads,
                   mlp_ratio=mlp_ratio,
                   embedding_dim=embedding_dim,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
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


def cct_14(*args, **kwargs):
    return _cct(num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def cct_16(*args, **kwargs):
    return _cct(num_layers=16, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def text_cct_2(*args, **kwargs):
    return _text_cct(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                     *args, **kwargs)


def text_cct_4(*args, **kwargs):
    return _text_cct(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                     *args, **kwargs)


def text_cct_6(*args, **kwargs):
    return _text_cct(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                     *args, **kwargs)

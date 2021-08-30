import torch.nn as nn
from .utils.transformers import TransformerClassifier, MaskedTransformerClassifier
from .utils.tokenizer import Tokenizer, TextTokenizer
from .utils.embedder import Embedder


__all__ = ['vit_lite_2', 'vit_lite_4', 'vit_lite_6',
           'vit_lite_7', 'vit_lite_8',
           'text_vit_lite_2', 'text_vit_lite_4', 'text_vit_lite_6',
           ]


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
                                   n_conv_layers=1,
                                   conv_bias=True)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=False,
            dropout=0.1,
            attention_dropout=0.,
            stochastic_depth=0.,
            *args, **kwargs)

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


class TextViTLite(nn.Module):
    def __init__(self,
                 seq_len=64,
                 word_embedding_dim=300,
                 embedding_dim=300,
                 patch_size=2,
                 *args, **kwargs):
        super(TextViTLite, self).__init__()
        assert seq_len % patch_size == 0, f"sequence length ({seq_len}) has to be" \
                                          f"divisible by patch size ({patch_size})"
        self.embedder = Embedder(word_embedding_dim=word_embedding_dim,
                                 *args, **kwargs)

        self.tokenizer = TextTokenizer(n_input_channels=word_embedding_dim,
                                       n_output_channels=embedding_dim,
                                       kernel_size=patch_size,
                                       stride=patch_size,
                                       padding=0,
                                       max_pool=False,
                                       activation=None)

        self.classifier = MaskedTransformerClassifier(
            seq_len=self.tokenizer.seq_len(seq_len=seq_len, embed_dim=word_embedding_dim),
            embedding_dim=embedding_dim,
            seq_pool=False,
            dropout=0.1,
            attention_dropout=0.,
            stochastic_depth=0.,
            *args, **kwargs)

    def forward(self, x, mask=None):
        x, mask = self.embedder(x, mask=mask)
        x, mask = self.tokenizer(x, mask=mask)
        out = self.classifier(x, mask=mask)
        return out


def _vit_lite(num_layers, num_heads, mlp_ratio, embedding_dim,
              patch_size=4, *args, **kwargs):
    return ViTLite(num_layers=num_layers,
                   num_heads=num_heads,
                   mlp_ratio=mlp_ratio,
                   embedding_dim=embedding_dim,
                   patch_size=patch_size,
                   *args, **kwargs)


def _text_vit_lite(num_layers, num_heads, mlp_ratio, embedding_dim,
                   patch_size=4, *args, **kwargs):
    return TextViTLite(num_layers=num_layers,
                       num_heads=num_heads,
                       mlp_ratio=mlp_ratio,
                       embedding_dim=embedding_dim,
                       patch_size=patch_size,
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


def text_vit_lite_2(*args, **kwargs):
    return _text_vit_lite(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                          *args, **kwargs)


def text_vit_lite_4(*args, **kwargs):
    return _text_vit_lite(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                          *args, **kwargs)


def text_vit_lite_6(*args, **kwargs):
    return _text_vit_lite(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                          *args, **kwargs)

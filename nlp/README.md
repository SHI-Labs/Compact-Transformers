# Compact Transformers in NLP

#### Transformer-Lite: Lightweight Transformer
Just a pure transformer encoder used as a text classifier. The embedded sequences
directly go through the model and are classified.
 
#### ViT-Lite: Lightweight ViT 
Like the transformer, but the word embeddings go through a convolutional layer similar
to the design in ViT, before going through the transformer layers.

#### CVT: Compact Vision Transformers
Similar to ViT-Lite, but lacking the class token and using sequence pooling instead.

#### CCT: Compact Convolutional Transformers
CVT with a more complicated convolutional layer, similar to the vision model.

# How to run

## Install locally

Our base model is in pure PyTorch and Torchvision. No extra packages are required.
Please refer to [PyTorch's Getting Started](https://pytorch.org/get-started/locally/) page for detailed instructions.

For each model (transformer/vit/cvt/cct) sizes 2, 4 and 6 are available.
```python3
from src.text import text_cct_2
model = text_cct_2(kernel_size=1)
```
For kernel size, we have found that sizes 1, 2 and 4 perform best.

You can even go further and create your own custom variant by importing the classes (i.e. `TextCCT`).

# Results

<table style="width:100%">
    <thead>
        <tr>
            <td><b>Model</b></td> 
            <td><b>Kernel size</b></td>
            <td><b>AGNews</b></td>
            <td><b>TREC</b></td>
            <td><b># Params</b></td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>CCT-2</td>
            <td>1</td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/nlp/text_cct2-1_agnews_93.45.pth">93.45%</a></td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/nlp/text_cct2-1_trec_91.00.pth">91.00%</a></td>
            <td>0.238M</td>
        </tr>
        <tr>
            <td>2</td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/nlp/text_cct2-2_agnews_93.51.pth">93.51%</a></td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/nlp/text_cct2-2_trec_91.80.pth">91.80%</a></td>
            <td>0.276M</td>
        </tr>
        <tr>
            <td>4</td>
            <td>93.80%</td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/nlp/text_cct2-4_trec_91.00.pth">91.00%</a></td>
            <td>0.353M</td>
        </tr>
        <tr>
            <td rowspan=3>CCT-4</td>
            <td>1</td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/nlp/text_cct4-1_agnews_93.55.pth">93.55%</a></td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/nlp/text_cct4-1_trec_91.80.pth">91.80%</a></td>
            <td>0.436M</td>
        </tr>
        <tr>
            <td>2</td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/nlp/text_cct4-2_agnews_93.24.pth">93.24%</a></td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/nlp/text_cct4-2_trec_93.60.pth">93.60%</a></td>
            <td>0.475M</td>
        </tr>
        <tr>
            <td>4</td>
            <td>93.09%</td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/nlp/text_cct4-4_trec_93.00.pth">93.00%</a></td>
            <td>0.551M</td>
        </tr>
        <tr>
            <td rowspan=3>CCT-6</td>
            <td>1</td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/nlp/text_cct6-1_agnews_93.78.pth">93.78%</a></td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/nlp/text_cct6-1_trec_91.60.pth">91.60%</a></td>
            <td>3.237M</td>
        </tr>
        <tr>
            <td>2</td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/nlp/text_cct6-2_agnews_93.33.pth">93.33%</a></td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/nlp/text_cct6-2_trec_92.20.pth">92.20%</a></td>
            <td>3.313M</td>
        </tr>
        <tr>
            <td>4</td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/nlp/text_cct6-4_agnews_92.95.pth">92.95%</a></td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/nlp/text_cct6-4_trec_92.80.pth">92.80%</a></td>
            <td>3.467M</td>
        </tr>
    </tbody>
</table>
More models are being uploaded.

# Citation
```bibtex
@article{hassani2021escaping,
	title        = {Escaping the Big Data Paradigm with Compact Transformers},
	author       = {Ali Hassani and Steven Walton and Nikhil Shah and Abulikemu Abuduweili and Jiachen Li and Humphrey Shi},
	year         = 2021,
	url          = {https://arxiv.org/abs/2104.05704},
	eprint       = {2104.05704},
	archiveprefix = {arXiv},
	primaryclass = {cs.CV}
}
```

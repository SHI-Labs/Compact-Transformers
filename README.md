# Compact Transformers

Preprint Link: [Escaping the Big Data Paradigm with Compact Transformers
](https://arxiv.org/abs/2104.05704)

By [Ali Hassani<sup>[1]</sup><span>&#42;</span>](https://alihassanijr.com/),
[Steven Walton<sup>[1]</sup><span>&#42;</span>](https://github.com/stevenwalton),
[Nikhil Shah<sup>[1]</sup>](https://itsshnik.github.io/),
[Abulikemu Abuduweili<sup>[1]</sup>](https://github.com/Walleclipse),
[Jiachen Li<sup>[1,2]</sup>](https://chrisjuniorli.github.io/), 
and
[Humphrey Shi<sup>[1,2,3]</sup>](https://www.humphreyshi.com/)


<small><span>&#42;</span>Ali Hassani and Steven Walton contributed equal work</small>

In association with SHI Lab @ University of Oregon<sup>[1]</sup> and
UIUC<sup>[2]</sup>, and Picsart AI Research (PAIR)<sup>[3]</sup>


![model-sym](images/model_sym.png)

# Abstract
With the rise of Transformers as the standard for language
processing, and their advancements in computer vi-sion, along with their
unprecedented size and amounts of training data, many have come to believe
that they are not suitable for small sets of data. This trend leads
to great concerns, including but not limited to: limited availability of
data in certain scientific domains and the exclusion ofthose with limited
resource from research in the field. In this paper, we dispel the myth that
transformers are “data-hungry” and therefore can only be applied to large
sets of data. We show for the first time that with the right size
and tokenization, transformers can perform head-to-head with state-of-the-art
CNNs on small datasets. Our model eliminates the requirement for class
token and positional embed-dings through a novel sequence pooling
strategy and the use of convolutions. We show that compared to CNNs,
our compact transformers have fewer parameters and MACs,while obtaining
similar accuracies. Our method is flexible in terms of model size, and can
have as little as 0.28M parameters and achieve reasonable results. It can
reach an ac-curacy of 94.72% when training from scratch on CIFAR-10,which is
comparable with modern CNN based approaches,and a significant improvement
over previous Transformer based models. Our simple and compact design
democratizes transformers by making them accessible to those equipped
with basic computing resources and/or dealing with important small
datasets.
 
#### ViT-Lite: Lightweight ViT 
Different from [ViT](https://arxiv.org/abs/2010.11929) we show that <i>an image 
is <b>not always</b> worth 16x16 words</i> and the image patch size matters.
Transformers are not in fact ''data-hungry,'' as the authors proposed, and
smaller patching can be used to train efficiently on smaller datasets.

#### CVT: Compact Vision Transformers
Compact Vision Transformers better utilize information with Sequence Pooling post 
encoder, eliminating the need for the class token while achieving better
accuracy.

#### CCT: Compact Convolutional Transformers
Compact Convolutional Transformers not only use the sequence pooling but also
replace the patch embedding with a convolutional embedding, allowing for better
inductive bias and making positional embeddings optional. CCT achieves better
accuracy than ViT-Lite and CVT and increases the flexibility of the input
parameters.

![Comparison](images/comparison.png)

# How to run 
We recommend starting with our faster version (CCT-2/3x2) which can be run with the
following command. If you are running on a CPU we recommend this model.
```bash
python main.py \
       --model cct_2 \
       --conv-size 3 \
       --conv-layers 2 \
       path/to/cifar10
```


If you would like to run our best running model (CCT-7/3x1) with CIFAR-10 on 
your machine, please use the following command.
```bash
python main.py \
       --model cct_7 \
       --conv-size 3 \
       --conv-layers 1 \
       path/to/cifar10
```

# Results
Type can be read in the format `L/PxC` where `L` is the number of transformer
layers, `P` is the patch/convolution size, and `C` (CCT only) is the number of
convolutional layers.

<table style="width:100%">
    <thead>
        <tr>
            <td><b>Model</b></td> 
            <td><b>Type</b></td> 
            <td><b>CIFAR-10</b></td> 
            <td><b>CIFAR-100</b></td> 
            <td><b># Params</b></td> 
            <td><b>MACs</b></td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>ViT-Lite</td>
            <td>7/4</td>
            <td>91.38%</td>
            <td>69.75%</td>
            <td>3.717M</td>
            <td>0.239G</td>
        </tr>
        <tr>
            <td>6/4</td>
            <td>90.94%</td>
            <td>69.20%</td>
            <td>3.191M</td>
            <td>0.205G</td>
        </tr>
        <tr>
            <td rowspan=2>CVT</td>
            <td>7/4</td>
            <td>92.43%</td>
            <td>73.01%</td>
            <td>3.717M</td>
            <td>0.236G</td>
        </tr>
        <tr>
            <td>6/4</td>
            <td>92.58%</td>
            <td>72.25%</td>
            <td>3.190M</td>
            <td>0.202G</td>
        </tr>
        <tr>
            <td rowspan=5>CCT</td>
            <td>2/3x2</td>
            <td>89.17%</td>
            <td>66.90%</td>
            <td><b>0.284M</b></td>
            <td><b>0.033G</b></td>
        </tr>
        <tr>
            <td>4/3x2</td>
            <td>91.45%</td>
            <td>70.46%</td>
            <td>0.482M</td>
            <td>0.046G</td>
        </tr>
        <tr>
            <td>6/3x2</td>
            <td>93.56%</td>
            <td>74.47%</td>
            <td>3.327M</td>
            <td>0.241G</td>
        </tr>
        <tr>
            <td>7/3x2</td>
            <td>93.65%</td>
            <td>74.77%</td>
            <td>3.853M</td>
            <td>0.275G</td>
        </tr>
        <tr>
            <td>7/3x1</td>
            <td><b>94.72%</b></td>
            <td><b>76.67%</b></td>
            <td>3.760M</td>
            <td>0.947G</td>
        </tr>
    </tbody>
</table>

Model zoo will be available soon.

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

# Compact Transformers

Paper Link: [currently unavailable]()

Authors: 
[Ali Hassani](https://alihassanijr.com/),
[Steven Walton](https://github.com/stevenwalton),
[Nikhil Shah](https://itsshnik.github.io/),
[Abulikemu Adbudweili](https://github.com/Walleclipse),
[Jiachen Li](https://chrisjuniorli.github.io/),
[Humphrey Shi](https://www.humphreyshi.com/)


<small>Note: Ali Hassani and Steven Walton contributed equal work</small>

In association with The University of Oregon and UIUC


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
encoder, eliminating the need for the class token. 

#### CCT: Compact Convolutional Transformers
Compact Convolutional Transformers not only use the sequence pooling but also
replace the patch embedding with a convolutional embedding, allowing for better
inductive bias and making positional embeddings optional.

![Comparison](images/comparison.png)

# How to run 
We recommend starting with our faster version (CCT-2/3x2) which can be run with the
following command. If you are running on a CPU we recommend this model.
```bash
python main.py \
       --dataset-name=cifar10 \
       --model cct \
       --model-size 2 \
       --conv-size 3 \
       --conv-layers 1 \
       --cos \
       --auto-aug \
       path/to/cifar
```


If you would like to run our best running model (CCT-7/3x1) with CIFAR-10 on 
your machine, please use the following command.
```bash
python main.py \
       --dataset-name=cifar10 \
       --model cct \
       --model-size 7 \
       --conv-size 3 \
       --conv-layers 1 \
       --cos \
       --auto-aug \
       path/to/cifar
```


# Results
| Model     | CIFAR-10 | CIFAR-100 | # Params | MACs |
|:---------:|:--------:|:---------:|:--------:|:----:|
| ViT-Lite |
| ViT-Lite-7/4 | 91.38% | 69.74% | 3.717M |0.239G |
| ViT-Lite-6/4 | 90.94% | 69.20% | 3.191M |0.205G |
| CVT |
| CVT-7/4 | 92.43% | 73.01% | 3.717M |0.236G |
| CVT-6/4 | 92.58% | 72.25% | 3.190M |0.202G |
| CCT |
| CCT-2/3x2 | 89.17%   | 66.90%    | 0.284M | 0.033G |
| CCT-4/3x2 | 91.45%   | 70.46%    | 0.482M | 0.046G |
| CCT-6/3x2 | 93.56%   | 74.47%    | 3.327M | 0.241G |
| CCT-7/3x2 | 93.65%   | 74.77%    | 3.853M | 0.275G |
| CCT-7/3x1 | 94.72%   | 76.67%    | 3.760M | 0.947G |

# Citation

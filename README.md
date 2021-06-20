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
reach an accuracy of 94.81% when training from scratch on CIFAR-10,which is
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

## Install locally

Please make sure you're using the following PyTorch version:
```bash
torch==1.8.1
torchvision==0.9.1
```
Refer to [PyTorch's Getting Started](https://pytorch.org/get-started/locally/) page for detailed instructions.

## Using Docker
There's also a `Dockerfile`, which builds off of the PyTorch image (requires CUDA).

## Training

We recommend starting with our faster version (CCT-2/3x2) which can be run with the
following command. If you are running on a CPU we recommend this model.
```bash
python main.py \
       --dataset cifar10 \
       --model cct_2 \
       --conv-size 3 \
       --conv-layers 2 \
       path/to/cifar10
```


If you would like to run our best running models (CCT-6/3x1 or CCT-7/3x1)
with CIFAR-10 on your machine, please use the following command.
```bash
python main.py \
       --dataset cifar10 \
       --model cct_6 \
       --conv-size 3 \
       --conv-layers 1 \
       --warmup 10 \
       --batch-size 64 \
       --checkpoint-path /path/to/checkpoint.pth \
       path/to/cifar10
```
## Evaluation

You can use `evaluate.py` to evaluate the performance of a checkpoint.
```bash
python evaluate.py \
       --dataset cifar10 \
       --model cct_6 \
       --conv-size 3 \
       --conv-layers 1 \
       --checkpoint-path /path/to/checkpoint.pth \
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
	    <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666087/vitlite7-4_cifar10.pth.zip">91.38%</a></td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666088/vitlite7-4_cifar100.pth.zip">69.75%</a></td>
            <td>3.717M</td>
            <td>0.239G</td>
        </tr>
        <tr>
            <td>6/4</td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666085/vitlite6-4_cifar10.pth.zip">90.94%</a></td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666086/vitlite6-4_cifar100.pth.zip">69.20%</a></td>
            <td>3.191M</td>
            <td>0.205G</td>
        </tr>
        <tr>
            <td rowspan=2>CVT</td>
            <td>7/4</td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666077/cvt7-4_cifar10.pth.zip">92.43%</a></td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666078/cvt7-4_cifar100.pth.zip">73.01%</a></td>
            <td>3.717M</td>
            <td>0.236G</td>
        </tr>
        <tr>
            <td>6/4</td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666075/cvt6-4_cifar10.pth.zip">92.58%</a></td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666076/cvt6-4_cifar100.pth.zip">72.25%</a></td>
            <td>3.190M</td>
            <td>0.202G</td>
        </tr>
        <tr>
            <td rowspan=6>CCT</td>
            <td>2/3x2</td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666059/cct2-3x2_cifar10.pth.zip">89.17%</a></td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666060/cct2-3x2_cifar100.pth.zip">66.90%</a></td>
            <td><b>0.284M</b></td>
            <td><b>0.033G</b></td>
        </tr>
        <tr>
            <td>4/3x2</td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666061/cct4-3x2_cifar10.pth.zip">91.45%</a></td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666066/cct4-3x2_cifar100.pth.zip">70.46%</a></td>
            <td>0.482M</td>
            <td>0.046G</td>
        </tr>
        <tr>
            <td>6/3x2</td>
	    <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6658604/cct6-3x2_cifar10_best.pth.zip">93.56%</a></td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6658605/cct6-3x2_cifar100_best.pth.zip">74.47%</a></td>
            <td>3.327M</td>
            <td>0.241G</td>
        </tr>
        <tr>
            <td>7/3x2</td>
	    <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6657152/cct7-3x2_cifar10_best.pth.zip">93.83%</a></td>
	    <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6657154/cct7-3x2_cifar100_best.pth.zip">74.92%</a></td>
            <td>3.853M</td>
            <td>0.275G</td>
        </tr>
        <tr>
            <td>7/3x1</td>
            <td><b><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6644400/cct7-3x1_cifar10_best.pth.zip">94.78%</a></b></td>
            <td><b><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6657226/cct7-3x1_cifar100_best.pth.zip">77.05%</a></b></td>
            <td>3.760M</td>
            <td>0.947G</td>
        </tr>
        <tr>
            <td>6/3x1</td>
            <td><b><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6657185/cct6-3x1_cifar10_best.pth.zip">94.81%</a></b></td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6657221/cct6-3x1_cifar100_best.pth.zip">76.71%</a></td>
            <td>3.168M</td>
            <td>0.813G</td>
        </tr>
    </tbody>
</table>

Click to download checkpoints.

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

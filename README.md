# Compact Transformers

Preprint Link: [Escaping the Big Data Paradigm with Compact Transformers
](https://arxiv.org/abs/2104.05704)

By [Ali Hassani<sup>[1]</sup><span>&#42;</span>](https://alihassanijr.com/),
[Steven Walton<sup>[1]</sup><span>&#42;</span>](https://stevenwalton.github.io/),
[Nikhil Shah<sup>[1]</sup>](https://itsshnik.github.io/),
[Abulikemu Abuduweili<sup>[1]</sup>](https://github.com/Walleclipse),
[Jiachen Li<sup>[1,2]</sup>](https://chrisjuniorli.github.io/), 
and
[Humphrey Shi<sup>[1,2,3]</sup>](https://www.humphreyshi.com/)


<small><span>&#42;</span>Ali Hassani and Steven Walton contributed equal work</small>

In association with SHI Lab @ University of Oregon<sup>[1]</sup> and
UIUC<sup>[2]</sup>, and Picsart AI Research (PAIR)<sup>[3]</sup>


![model-sym](images/model_sym.png)

## Other implementations & resources
**[PyTorch blog]**: check out our [official blog post with PyTorch](https://medium.com/pytorch/training-compact-transformers-from-scratch-in-30-minutes-with-pytorch-ff5c21668ed5) to learn more about our work and vision transformers in general.

**[Keras]**: check out [Compact Convolutional Transformers on keras.io](https://keras.io/examples/vision/cct/) by [Sayak Paul](https://github.com/sayakpaul).

**[vit-pytorch]**: CCT is also available through [Phil Wang](https://github.com/lucidrains)'s [vit-pytorch](https://github.com/lucidrains/vit-pytorch), simply use ```pip install vit-pytorch```


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
reach an accuracy of 95.29% when training from scratch on CIFAR-10,which is
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

## CIFAR-10 and CIFAR-100

<table style="width:100%">
    <thead>
        <tr>
            <td><b>Model</b></td> 
            <td><b>Type</b></td> 
            <td><b>Epochs</b></td> 
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
            <td>200</td>
	    <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666087/vitlite7-4_cifar10.pth.zip">91.38%</a></td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666088/vitlite7-4_cifar100.pth.zip">69.75%</a></td>
            <td>3.717M</td>
            <td>0.239G</td>
        </tr>
        <tr>
            <td>6/4</td>
            <td>200</td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666085/vitlite6-4_cifar10.pth.zip">90.94%</a></td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666086/vitlite6-4_cifar100.pth.zip">69.20%</a></td>
            <td>3.191M</td>
            <td>0.205G</td>
        </tr>
        <tr>
            <td rowspan=2>CVT</td>
            <td>7/4</td>
            <td>200</td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666077/cvt7-4_cifar10.pth.zip">92.43%</a></td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666078/cvt7-4_cifar100.pth.zip">73.01%</a></td>
            <td>3.717M</td>
            <td>0.236G</td>
        </tr>
        <tr>
            <td>6/4</td>
            <td>200</td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666075/cvt6-4_cifar10.pth.zip">92.58%</a></td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666076/cvt6-4_cifar100.pth.zip">72.25%</a></td>
            <td>3.190M</td>
            <td>0.202G</td>
        </tr>
        <tr>
            <td rowspan=7>CCT</td>
            <td>2/3x2</td>
            <td>200</td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666059/cct2-3x2_cifar10.pth.zip">89.17%</a></td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666060/cct2-3x2_cifar100.pth.zip">66.90%</a></td>
            <td><b>0.284M</b></td>
            <td><b>0.033G</b></td>
        </tr>
        <tr>
            <td>4/3x2</td>
            <td>200</td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666061/cct4-3x2_cifar10.pth.zip">91.45%</a></td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6666066/cct4-3x2_cifar100.pth.zip">70.46%</a></td>
            <td>0.482M</td>
            <td>0.046G</td>
        </tr>
        <tr>
            <td>6/3x2</td>
            <td>200</td>
	    <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6658604/cct6-3x2_cifar10_best.pth.zip">93.56%</a></td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6658605/cct6-3x2_cifar100_best.pth.zip">74.47%</a></td>
            <td>3.327M</td>
            <td>0.241G</td>
        </tr>
        <tr>
            <td>7/3x2</td>
            <td>200</td>
	    <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6657152/cct7-3x2_cifar10_best.pth.zip">93.83%</a></td>
	    <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6657154/cct7-3x2_cifar100_best.pth.zip">74.92%</a></td>
            <td>3.853M</td>
            <td>0.275G</td>
        </tr>
        <tr>
            <td>7/3x1</td>
            <td>200</td>
            <td><b><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6644400/cct7-3x1_cifar10_best.pth.zip">94.78%</a></b></td>
            <td><b><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6657226/cct7-3x1_cifar100_best.pth.zip">77.05%</a></b></td>
            <td>3.760M</td>
            <td>0.947G</td>
        </tr>
        <tr>
            <td>6/3x1</td>
            <td>200</td>
            <td><b><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6657185/cct6-3x1_cifar10_best.pth.zip">94.81%</a></b></td>
            <td><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6657221/cct6-3x1_cifar100_best.pth.zip">76.71%</a></td>
            <td>3.168M</td>
            <td>0.813G</td>
        </tr>
        <tr>
            <td>6/3x1</td>
            <td>500</td>
            <td><b><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6730781/cct6-3x1_cifar10_500.pth.zip">95.29%</a></b></td>
            <td><b><a href="https://github.com/SHI-Labs/Compact-Transformers/files/6730783/cct6-3x1_cifar100_500.pth.zip">77.31%</a></b></td>
            <td>3.168M</td>
            <td>0.813G</td>
        </tr>
    </tbody>
</table>

### Randaugment + Mixup + CutMix
We trained the following using [timm](https://github.com/rwightman/pytorch-image-models).

<table style="width:100%">
    <thead>
        <tr>
            <td><b>Model</b></td>
            <td><b>Epochs</b></td> 
            <td><b>PE</b></td>
            <td><b>CIFAR-10</b></td> 
            <td><b>CIFAR-100</b></td> 
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>CCT-7/3x1</td>
            <td>300</td>
            <td>Learnable</td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/cct7-3x1_timm_cifar10_300epochs_96.53.pth">96.53%</a></td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/cct7-3x1_timm_cifar100_300epochs_80.92.pth">80.92%</a></td>
        </tr>
        <tr>
            <td>1500</td>
            <td>Sinusoidal</td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/cct7-3x1_timm_cifar10_1500epochs_97.48.pth">97.48%</a></td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/cct7-3x1_timm_cifar100_1500epochs_82.72.pth">82.72%</a></td>
        </tr>
        <tr>
            <td>5000</td>
            <td>Sinusoidal</td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/cct7-3x1_timm_cifar10_5000epochs_98.00.pth">98.00%</a></td>
            <td>-</td>
        </tr>
    </tbody>
</table>

## ImageNet

<table style="width:100%">
    <thead>
        <tr>
            <td><b>Model</b></td> 
            <td><b>Type</b></td> 
            <td><b>Resolution</b></td> 
            <td><b>Epochs</b></td> 
            <td><b>Top-1 Accuracy</b></td>
            <td><b># Params</b></td> 
            <td><b>MACs</b></td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=1><a href="https://github.com/google-research/vision_transformer/">ViT</a></td>
            <td>12/16</td>
	        <td>384</td>
	        <td>300</td>
            <td>77.91%</td>
            <td>86.8M</td>
            <td>17.6G</td>
        </tr>
        <tr>
            <td rowspan=2>CCT</td>
            <td>14t/7x2</td>
	        <td>224</td>
            <td>310</td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/cct14t-7x2_imagenet_80.67.pth">80.67%</a></td>
            <td>22.36M</td>
            <td>5.11G</td>
        </tr>
        <tr>
            <td>14t/7x2</td>
	        <td>384</td>
            <td>310</td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/cct14t-7x2_imagenet384_finetune_82.71.pth">82.71%</a></td>
            <td>22.51M</td>
            <td>15.02G</td>
        </tr>
    </tbody>
</table>

Please note that we used [Ross Wightman's ImageNet training script](https://github.com/rwightman/pytorch-image-models) to train these.

## NLP Results

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
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/nlp/text_cct2-1_agnews_93.45.pth">93.45%</a></td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/nlp/text_cct2-1_trec_91.00.pth">91.00%</a></td>
            <td>0.238M</td>
        </tr>
        <tr>
            <td>2</td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/nlp/text_cct2-2_agnews_93.51.pth">93.51%</a></td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/nlp/text_cct2-2_trec_91.80.pth">91.80%</a></td>
            <td>0.276M</td>
        </tr>
        <tr>
            <td>4</td>
            <td>93.80%</td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/nlp/text_cct2-4_trec_91.00.pth">91.00%</a></td>
            <td>0.353M</td>
        </tr>
        <tr>
            <td rowspan=3>CCT-4</td>
            <td>1</td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/nlp/text_cct4-1_agnews_93.55.pth">93.55%</a></td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/nlp/text_cct4-1_trec_91.80.pth">91.80%</a></td>
            <td>0.436M</td>
        </tr>
        <tr>
            <td>2</td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/nlp/text_cct4-2_agnews_93.24.pth">93.24%</a></td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/nlp/text_cct4-2_trec_93.60.pth">93.60%</a></td>
            <td>0.475M</td>
        </tr>
        <tr>
            <td>4</td>
            <td>93.09%</td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/nlp/text_cct4-4_trec_93.00.pth">93.00%</a></td>
            <td>0.551M</td>
        </tr>
        <tr>
            <td rowspan=3>CCT-6</td>
            <td>1</td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/nlp/text_cct6-1_agnews_93.78.pth">93.78%</a></td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/nlp/text_cct6-1_trec_91.60.pth">91.60%</a></td>
            <td>3.237M</td>
        </tr>
        <tr>
            <td>2</td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/nlp/text_cct6-2_agnews_93.33.pth">93.33%</a></td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/nlp/text_cct6-2_trec_92.20.pth">92.20%</a></td>
            <td>3.313M</td>
        </tr>
        <tr>
            <td>4</td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/nlp/text_cct6-4_agnews_92.95.pth">92.95%</a></td>
            <td><a href="http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/nlp/text_cct6-4_trec_92.80.pth">92.80%</a></td>
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

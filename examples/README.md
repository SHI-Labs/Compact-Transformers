# Compact Transformers
**This subdirectory contains a bare-minimum training and evaluation script.**

## Install PyTorch
Please refer to [PyTorch's Getting Started](https://pytorch.org/get-started/locally/) page for detailed instructions.

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

The following results can be reached with this training script.

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

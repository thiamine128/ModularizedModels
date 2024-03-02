# Modularized Models with SeaM

## Introduction

This is a project of modularized models using SeaM algorithm (https://github.com/qibinhang/SeaM). Beside the models in the SeaM project, it includes MLP Mixer and Densenet121 for multi-class classification.

### Modularizing Your Own Model with SeaM

- Replace some layers with masked layers in the architecture (e.g. from Linear to MaskLinear, from Conv2D to MaskConv. You can write your own masked layer if neccessary.)
- Append module head
- Reengineer your model

## Requirements

+ advertorch 0.2.3<br>
+ fvcore 0.1.5.post20220512<br>
+ matplotlib 3.4.2<br>
+ numpy 1.19.2<br>
+ python 3.8.10<br>
+ pytorch 1.8.1<br>
+ torchvision 0.9.0<br>
+ tqdm 4.61.0<br>
+ GPU with CUDA support is also needed

<br>

## Structure of the directories

```powershell
  |--- README.md                        :  user guidance
  |--- data/                            :  experimental data
  |--- src/                             :  source code of SeaM and models to be modularized
  |------ global_config.py              :  setting the path
  |------ binary_class/                 :  direct reuse on binary classification problems
  |--------- model_reengineering.py     :  re-engineering a trained model and then reuse the re-engineered model
  |--------- calculate_flop.py          :  calculating the number of FLOPs required by reusing the re-engineered and original models
  |--------- calculate_time_cost.py     :  calculating the inference time required by reusing the re-engineered and original models
  |--------- ......                 
  |------ multi_class/                  :  direct reuse on multi-class classification problems
  |--------- ...... 
  |------ defect_inherit/               :  indirect reuse 
  |--------- reengineering_finetune.py  :  re-engineering a trained model and then fine-tuning the re-engineered model
  |--------- standard_finetune.py       :  using standard fine-tuning approach to fine-tune a trained model
  |--------- eval_robustness.py         :  calculating the defect inheritance rate
  |--------- ......
```

<br>

## Downloading experimental data

1. We provide the trained models and datasets used in the experiments, as well as the corresponding re-engineered models.<br>
   One can download `data/` from [here](https://mega.nz/file/tX91ACpR#CSbQ2Xariha7_HLavE_6pKg4FoO5axOPemlv5J0JYwY) and then move it to `SeaM/`.<br>
   The trained models will be downloaded automatically by PyTorch when running our project. If the download fails, please move our provided trained models to the folder according to the failure information given by PyTorch.<br>
   Due to the huge size of ImageNet, please download it from its [webpage](https://www.image-net.org/).
2. Modify `self.root_dir` in `src/global_config.py`.

## Direct model reuse  

### Re-engineering on multi-class classification problems

1. Go to the directory of experiments related to the multi-class classification problems.

```commandline
cd src/multi_class
```

2. Re-engineer MLPMixer-ImageNet1K on a multi-class classification problem.

```commandline
python model_reengineering.py --model mlpmixer --dataset imagenet --target_superclass_idx 0 --superclass_type predefined --lr_mask 0.05 --alpha 2
```

3. Compute the number of FLOPs required by the original and re-engineered ResNet20-CIFAR100, respectively. This command also presents the accuracy of models. 

```commandline
python calculate_flop.py --model mlpmixer --dataset imagenet --target_superclass_idx 0 --superclass_type predefined --lr_mask 0.05 --alpha 2
```

4. Compute the time cost for inference required by the original and re-engineered ResNet20-CIFAR100, respectively. This command also presents the number of a model's weights. 

```commandline
python calculate_time_cost.py --model mlpmixer --dataset imagenet --target_superclass_idx 0 --superclass_type predefined --lr_mask 0.05 --alpha 2
```

***NOTE***: When computing the time cost for inference, DeepSparse runs a model on several CPUs.
The inference process would be interfered with other active processes, leading to fluctuations in inference time cost.
In our experiments, we manually kill as many other processes as possible and enable the inference process to occupy the CPUs exclusively.


## Supplementary experimental results

### The values of major parameters

The following table shows the default hyperparameters. The details of settings for re-engineering each trained model on each target problem can be obtained according to the experimental result files. <br>
For instance, the values of *learning rate* and *alpha* for the re-engineered model file `SeaM/data/multi_class_classification/resnet20_cifar100/predefined/tsc_0/lr_head_mask_0.1_0.05_alpha_1.0.pth` are 0.05 and 1.0, respectively.

<table>
<thead>
  <tr>
    <th>Target Problem</th>
    <th>Model Name</th>
    <th>Learning Rate</th>
    <th>Alpha</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">Multi-class<br>Classification</td>
    <td>MLPMixer-ImageNet</td>
    <td>0.05</td>
    <td>2.00</td>
  </tr>
  <tr>
    <td>Densenet121-ImageNet</td>
    <td>0.10</td>
    <td>2.00</td>
  </tr>
</tbody>
</table>


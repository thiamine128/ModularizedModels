# Modularized Models with SeaM

## Introduction

This is a repository of modularized models (MLP-Mixer, Densenet121, Mobilenet-v3-small) using [SeaM](https://github.com/qibinhang/SeaM) algorithm .

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

## Project Setup

- Download Dataset

​	Due to the huge size of ImageNet, please download it from its [webpage](https://www.image-net.org/).

- Modify Configurations

​	Modify `self.root_dir` in `src/global_config.py`.

## Structure of the directories

```powershell
  |--- README.md                        :  user guidance
  |--- data/                            :  experimental data
  |--- src/                             :  source code of SeaM and models to be modularized
  |------ global_config.py              :  setting the path        
  |------ multi_class/                  :  direct reuse on multi-class classification problems
```



## Direct model reuse  

### Re-engineering on multi-class classification problems

- Go to the directory of experiments related to the multi-class classification problems.

```commandline
cd src/multi_class
```

- Re-engineer specific models on a multi-class classification problem.

  - MLP-Mixer:

    ```shell
    python model_reengineering.py --model mlpmixer --dataset imagenet --target_superclass_idx 0 --superclass_type predefined --lr_mask 0.05 --alpha 2
    ```

  - Alexnet

    ```shell
    python model_reengineering.py --model alexnet --dataset imagenet --target_superclass_idx 0 --superclass_type predefined --lr_mask 0.1 --alpha 2
    ```

    

  - Densenet121

    ```shell
    python model_reengineering.py --model densenet121 --dataset imagenet --target_superclass_idx 0 --superclass_type predefined --lr_mask 0.1 --alpha 2
    ```

  - Mobilenet-v3-small

    ```shell
    python model_reengineering.py --model mobilenet_v3_small --dataset imagenet --target_superclass_idx 0 --superclass_type predefined --lr_mask 0.01 --alpha 2
    ```




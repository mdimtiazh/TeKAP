# TeKAP

This repo:

**(1) covers the implementation of the following ICLR 2025 Submitted paper with ID:6163:**

"SINGLE TEACHER, MULTIPLE PERSPECTIVES: TEACHER KNOWLEDGE AUGMENTATION FOR ENHANCED KNOWLEDGE DISTILLATION":

<div style="text-align:center"><img src="http://hobbitlong.github.io/CRD/CRD_files/teaser.jpg" width="85%" height="85%"></div>  

<p></p>

# Running

1. Fetch the pretrained teacher models by:

    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
   which will download and save the models to `save/models`
   
2. Run distillation by following commands. An example of running our TeKAP logit level (we use KL divergence as the base KD) is given by:

    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1
    ```
    where the flags are explained as:
    - `--path_t`: specify the path of the teacher model
    - `--model_s`: specify the student model, see 'models/\_\_init\_\_.py' to check the available model types.
    - `--distill`: specify the distillation method
    - `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
    - `-a`: the weight of the KD loss, default: `None`
    - `-b`: the weight of other distillation losses, default: `None`
    - `--trial`: specify the experimental id to differentiate between multiple runs.
    
    Therefore, the command for running TeKAP for feature level (we use CRD as the base method) is something like:
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 0 -b 0.8 --trial 1
    ```
    
3. Combining a distillation objective with TeKAP feature level and logit level is simply done by setting `-a` as a non-zero value, which results in the following example (TeKAP* (F+L))
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 1 -b 0.8 --trial 1     
    ```

4. (optional) Train teacher networks from scratch. Example command:
```
	python train_teacher.py --model resnet32x4
```

Note: the default setting is for a single-GPU training. 


## Acknowledgement

Thanks to the authors of CRD (https://arxiv.org/pdf/1910.10699) for providing the code of CRD. Thanks also go to authors of other papers who make their code publicly available.

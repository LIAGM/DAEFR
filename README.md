# Updating
<!-- - **20230915** Update an online demo [![Huggingface Gradio](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/wzhouxiff/RestoreFormerPlusPlus)
- **20230116** For convenience, we further upload the [test datasets](#testset), including CelebA (both HQ and LQ data), LFW-Test, CelebChild-Test, and Webphoto-Test, to OneDrive and BaiduYun.
- **20221003** We provide the link of the [test datasets](#testset).
- **20220924** We add the code for [**metrics**](#metrics) in scripts/metrics. -->
- **20231226** Create project.


# DAEFR

This repo includes the source code of the paper: "[Dual Associated Encoder for Face Restoration](https://arxiv.org/abs/2308.07314)" by Yu-Ju Tsai, Yu-Lun Liu, Lu Qi, Kelvin C.K. Chan, and Ming-Hsuan Yang.

![](assets/DAEFR_results.png)

We propose a novel dual-branch framework named **DAEFR**. Our method introduces an auxiliary LQ branch that extracts crucial information from the LQ inputs. Additionally, we incorporate association training to promote effective synergy between the two branches, enhancing code prediction and output quality. We evaluate the effectiveness of DAEFR on both synthetic and real-world datasets, demonstrating its superior performance in restoring facial details.

![](assets/DAEFR_arch.png)

## Environment

- python>=3.7
- pytorch>=1.7.1
- pytorch-lightning==1.0.8
- omegaconf==2.0.0
- basicsr==1.3.3.4

**Warning** Different versions of pytorch-lightning and omegaconf may lead to errors or different results.

## Preparations of dataset and models

**Dataset**: 
- Training data: **HQ Codebook**, **LQ Codebook** and **DAEFR** are trained with **FFHQ** which attained from [FFHQ repository](https://github.com/NVlabs/ffhq-dataset). The original size of the images in FFHQ are 1024x1024. We resize them to 512x512 with bilinear interpolation in our work. Link this dataset to ./data/FFHQ/image512x512.
- <a id="testset">Test data</a>:
   * CelebA-Test-HQ: [HuggingFace](https://huggingface.co/datasets/LIAGM/DAEFR_test_datasets/blob/main/celeba_512_validation.zip);
   * CelebA-Test-LQ: [HuggingFace](https://huggingface.co/datasets/LIAGM/DAEFR_test_datasets/blob/main/self_celeba_512_v2.zip);
   * LFW-Test: [HuggingFace](https://huggingface.co/datasets/LIAGM/DAEFR_test_datasets/blob/main/lfw_cropped_faces.zip);
   * WIDER-Test: [HuggingFace](https://huggingface.co/datasets/LIAGM/DAEFR_test_datasets/blob/main/Wider-Test.zip);

**Model**: Pretrained models used for training and the trained model of our DAEFR can be attained from [HuggingFace](https://huggingface.co/LIAGM/DAEFR_pretrain_model/tree/main). Link these models to ./experiments.

Make sure the models are stored as follows:
```
experiments/
|-- pretrained_models/
    |-- FFHQ_eye_mouth_landmarks_512.pth
    |-- arcface_resnet18.pth    
    |-- inception_FFHQ_512-f7b384ab.pth
    |-- lpips/
        |-- vgg.pth
|-- HQ_codebook.ckpt
|-- LQ_codebook.ckpt
|-- Association_stage.ckpt
|-- DAEFR_model.ckpt

```

## Test
    sh scripts/test.sh
Or you can use the following command for testing:
```
CUDA_VISIBLE_DEVICES=$GPU python -u scripts/test.py \
--outdir $outdir \
-r $checkpoint \
-c $config \
--test_path $align_test_path \
--aligned
```

## Training
### First stage for codebooks
    sh scripts/run_HQ_codebook_training.sh
    sh scripts/run_LQ_codebook_training.sh
### Second stage for Association
    sh scripts/run_association_stage_training.sh
### Final stage for DAEFR
    sh scripts/run_DAEFR_training.sh

**Note**. 
- The second stage is for model association. You need to add your trained HQ\_Codebook and LQ\_Codebook model to `ckpt_path_HQ` and `ckpt_path_LQ` in config/Association_stage.yaml.
- The final stage is for face restoration. You need to add your trained HQ\_Codebook and Association model to `ckpt_path_HQ` and `ckpt_path_LQ` in config/DAEFR.yaml.
- Our model is trained with 8 A100 GPUs.

## <a id="metrics">Metrics</a>
    sh scripts/metrics/run.sh
    
**Note**. 
- You need to add the path of CelebA-Test dataset in the script if you want get IDD, PSRN, SSIM, LIPIS.

## Citation
    @article{tsai2023dual,
        title={Dual Associated Encoder for Face Restoration},
        author={Tsai, Yu-Ju and Liu, Yu-Lun and Qi, Lu and Chan, Kelvin CK and Yang, Ming-Hsuan},
        journal={arXiv preprint arXiv:2308.07314},
        year={2023}
    }

## Acknowledgement
We thank everyone who makes their code and models available, especially [Taming Transformer](https://github.com/CompVis/taming-transformers), [basicsr](https://github.com/XPixelGroup/BasicSR), and [RestoreFormer](https://github.com/wzhouxiff/RestoreFormer).

## Contact
For any question, feel free to email `louis19950117@gmail.com`.
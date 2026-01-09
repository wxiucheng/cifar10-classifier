<div align="center">
<h1>CIFAR-10 Classifier
</h1>



[Xiucheng Wang](https://wxiucheng.github.io/)&#8224; 

[Beihang University]


<a href="https://wxiucheng.github.io/">
<img src='https://img.shields.io/badge/arxiv-Cifar10Classifier-blue' alt='Paper PDF'></a>
<a href="https://wxiucheng.github.io/cifar10-classifier/">
<img src='https://img.shields.io/badge/Project-Website-orange' alt='Project Page'></a>
</div>

## 📖 Abstract
- 简单 CNN 与 ResNet18 两套模型，支持预训练微调
- 统一的 YAML 配置驱动（模型/数据/训练）
- 训练、验证、测试全流程与最优/最后权重保存
- 评估脚本与 Gradio 可视化 Demo

![DreamText Teaser](demo/teaser.png)

## 🔧 Usage

### Environment Setup

```bash
conda create -n dreamtext python=3.11
conda activate dreamtext
pip install -r requirements.txt
```

### Download our Pre-trained Models
Download our available [checkpoints](https://drive.google.com/file/d/1Q4B0oAnksORsPJS5TwoJU5uPRSFEbwS5/view?usp=sharing) and put them in the corresponding directories in `./checkpoints`.


## 🚀 Gradio Demo
You can run the demo locally by
```
python run_gradio.py
```
<img src=demo/gradio.png style="zoom:30%" />


## 🎨 Preparing Datasets


### LAION-OCR
- Create a data directory `{your data root}/LAION-OCR` in your disk and put your data in it. Then set the **data_root** field in `./configs/dataset/locr.yaml`.
- For the downloading and preprocessing of Laion-OCR dataset, please refer to [TextDiffuser](https://github.com/microsoft/unilm/tree/master/textdiffuser) and `./scripts/preprocess/laion_ocr_pre.ipynb`.

### ICDAR13
- Create a data directory `{your data root}/ICDAR13` in your disk and put your data in it. Then set the **data_root** field in `./configs/dataset/icd13.yaml`.
- Build the tree structure as below:
```
ICDAR13
├── train                  // training set
    ├── annos              // annotations
        ├── gt_x.txt
        ├── ...
    └── images             // images
        ├── img_x.jpg
        ├── ...
└── val                    // validation set
    ├── annos              // annotations
        ├── gt_img_x.txt
        ├── ...
    └── images             // images
        ├── img_x.jpg
        ├── ...
```

### TextSeg
- Create a data directory `{your data root}/TextSeg` in your disk and put your data in it. Then set the **data_root** field in `./configs/dataset/tsg.yaml`.
- Build the tree structure as below:
```
TextSeg
├── train                  // training set
    ├── annotation         // annotations
        ├── x_anno.json    // annotation json file
        ├── x_mask.png     // character-level mask
        ├── ...
    └── image              // images
        ├── x.jpg.jpg
        ├── ...
└── val                    // validation set
    ├── annotation         // annotations
        ├── x_anno.json    // annotation json file
        ├── x_mask.png     // character-level mask
        ├── ...
    └── image              // images
        ├── x.jpg
        ├── ...
```

### SynthText
- Create a data directory `{your data root}/SynthText` in your disk and put your data in it. Then set the **data_root** field in `./configs/dataset/st.yaml`.
- Build the tree structure as below:
```
SynthText
├── 1                      // part 1
    ├── ant+hill_1_0.jpg   // image
    ├── ant+hill_1_1.jpg
    ├── ...
├── 2                      // part 2
├── ...
└── gt.mat                 // annotation file
```



## 💻 Training
Download the [stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/blob/main/512-inpainting-ema.ckpt) and put it in `./checkpoints/pretrained/`.

Set the parameters in `./configs/train.yaml` and run:

```
python train.py
```

## ✨ Evaluation
Set the parameters in `./configs/test.yaml` and run:

```
python test.py
```



## 🎫 License
For non-commercial academic use, this project is licensed under [the 2-clause BSD License](https://opensource.org/license/bsd-2-clause). 
For commercial use, please contact [Cheng Jin](jc@fudan.edu.cn).


## ⭐ BibTeX
If you find our work helpful, please leave us a star and cite our paper.

```bibtex
@inproceedings{DreamText,
      title={High Fidelity Scene Text Synthesis},
      author={Wang, Yibin and Zhang, Weizhong and Honghui, Xu and Jin, Cheng},
      booktitle={CVPR},
      year={2025}
    }
```


## 📧 Contact

If you have any technical comments or questions, please open a new issue or feel free to contact [Yibin Wang](https://codegoat24.github.io).


## 🙏 Acknowledgements

Our work is based on [Stable-Diffusion](https://github.com/Stability-AI/stablediffusion), thanks to all the contributors!

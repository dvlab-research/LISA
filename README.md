# LISA: Reasoning Segmentation via Large Language Model

<font size=10><div align='center'><b>LISA</b>: Large <b>L</b>anguage <b>I</b>nstructed <b>S</b>egmentation <b>A</b>ssistant</div></font>

<font size=10><div align='center' > <a href=https://arxiv.org/pdf/2308.00692.pdf>**Paper**</a> | <a href="https://huggingface.co/xinlai">**Models**</a> | [**Inference**](#inference) | [**Dataset**](#dataset) | <a href="http://103.170.5.190:7860/">**Online Demo**</a></div></font>

<!-- <p align="center"> <img src="imgs/teaser.jpg" width="100%"> </p> -->

<table class="center">
<tr>
  <td style="text-align:center;"><b>Input</b></td>
  <td style="text-align:center;"><b>Output</b></td>
  <td style="text-align:center;"><b>Input</b></td>
  <td style="text-align:center;"><b>Output</b></td>
</tr>
<tr>
  <td><img src="imgs/obama.jpg"></td>
  <td><img src="vis_output/obama.jpg"></td>
  <td><img src="imgs/trump.jpg"></td>              
  <td><img src="vis_output/trump.jpg"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"Who was <u>the president of the US</u> in this image? Please output segmentation mask and explain the reason."</td>
  <td width=25% style="text-align:center;">"Sure, the segmentation result is [SEG]. <u>The President of the United States in the image is President Obama.</u>”</td>
  <td width=25% style="text-align:center;color:gray">"Who was <u>Tthe president of the US</u> in this image? Please output segmentation mask and explain why."</td>
  <td width=25% style="text-align:center;">"Sure, [SEG]. In the image, <u>the President of the United States is President Trump.</u>"</td>
</tr>

<tr>
  <td><img src="imgs/stand_higher.jpg"></td>
  <td><img src="vis_output/stand_higher.jpg"></td>
  <td><img src="imgs/camera_lens.jpg"></td>              
  <td><img src="vis_output/camera_lens.jpg"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"What can <u>make the woman stand higher</u>? Please output segmentation mask and explain why."</td>
  <td width=25% style="text-align:center;">"Sure, [SEG]. The woman is <u>standing higher by using a ladder</u>..."</td>
  <td width=25% style="text-align:center;color:gray">"Can you segment <u>the camera lens that is more suitable for photographing nearby objects</u> in this image?"</td>
  <td width=25% style="text-align:center;">"Sure, [SEG]."</td>
</tr>

<tr>
  <td><img src="imgs/dog_with_horn.jpg"></td>
  <td><img src="vis_output/dog_with_horn.jpg"></td>
  <td><img src="imgs/wash_hands.jpg"></td>              
  <td><img src="vis_output/wash_hands.jpg"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"Can you segment <u>the unusual part</u> in this image and explain why."</td>
  <td width=25% style="text-align:center;">"Sure, it is [SEG]. In the image, the unusual part is <u>the dog wearing a reindeer antler headband</u>..."</td>
  <td width=25% style="text-align:center;color:gray">"Where to wash hands in this image? Please output segmentation mask."</td>
  <td width=25% style="text-align:center;">"Sure, [SEG]."</td>
</tr>
</table>

<p align="center"> <img src="imgs/fig_overview.jpg" width="100%"> </p>

## News
- [x] [2023.8.4] [Online Demo](http://103.170.5.190:7860/) is released! 
- [x] [2023.8.4] [*ReasonSeg* Dataset](https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy?usp=sharing) and the [LISA-13B-llama2-v0-explainatory](https://huggingface.co/xinlai/LISA-13B-llama2-v0-explainatory) model are released! 
- [x] [2023.8.3] Inference code and the [LISA-13B-llama2-v0](https://huggingface.co/xinlai/LISA-13B-llama2-v0) model are released. Welcome to check out!
- [x] [2023.8.2] [Paper](https://arxiv.org/pdf/2308.00692.pdf) is released and GitHub repo is created.

## TODO 
- [ ] Training Code Release

**LISA: Reasoning Segmentation Via Large Language Model [[Paper](https://arxiv.org/abs/2308.00692)]** <br />
[Xin Lai](https://scholar.google.com/citations?user=tqNDPA4AAAAJ&hl=zh-CN),
[Zhuotao Tian](https://scholar.google.com/citations?user=mEjhz-IAAAAJ&hl=en),
[Yukang Chen](https://scholar.google.com/citations?user=6p0ygKUAAAAJ&hl=en),
[Yanwei Li](https://scholar.google.com/citations?user=I-UCPPcAAAAJ&hl=zh-CN),
[Yuhui Yuan](https://scholar.google.com/citations?user=PzyvzksAAAAJ&hl=en),
[Shu Liu](https://scholar.google.com.hk/citations?user=BUEDUFkAAAAJ&hl=zh-CN),
[Jiaya Jia](https://scholar.google.com/citations?user=XPAkzTEAAAAJ&hl=en)<br />

## Abstract
In this work, we propose a new segmentation task --- ***reasoning segmentation***. The task is designed to output a segmentation mask given a complex and implicit query text. We establish a benchmark comprising over one thousand image-instruction pairs, incorporating intricate reasoning and world knowledge for evaluation purposes. Finally, we present LISA: Large-language Instructed Segmentation Assistant, which inherits the language generation capabilities of the multi-modal Large Language Model (LLM) while also possessing the ability to produce segmentation masks.
For more details, please refer to the [paper](https://arxiv.org/abs/2308.00692).

## Highlights
**LISA** unlocks the new segmentation capabilities of multi-modal LLMs, and can handle cases involving: 
1. complex reasoning; 
2. world knowledge; 
3. explanatory answers; 
4. multi-turn conversation. 

**LISA** also demonstrates robust zero-shot capability when trained exclusively on reasoning-free datasets. In addition, fine-tuning the model with merely 239 reasoning segmentation image-instruction pairs results in further performance enhancement.

## Experimental results
<p align="center"> <img src="imgs/table1.jpg" width="80%"> </p>

## Installation
```
pip install -r requirements.txt
```
 
## Inference 
To chat with [LISA-13B-llama2-v0](https://huggingface.co/xinlai/LISA-13B-llama2-v0) or [LISA-13B-llama2-v0-explainatory](https://huggingface.co/xinlai/LISA-13B-llama2-v0-explainatory): (Note that LISA-13B-llama2-v0 currently does not support explanatory answers.)
```
CUDA_VISIBLE_DEVICES=0 python3 chat.py --version='xinlai/LISA-13B-llama2-v0'
```
To use `bf16` or `fp16` data type for inference:
```
CUDA_VISIBLE_DEVICES=0 python3 chat.py --version='xinlai/LISA-13B-llama2-v0' --precision='bf16'
```
To use `8bit` or `4bit` data type for inference (this enables running 13B model on a single 24G or 12G GPU at some cost of generation quality):
```
CUDA_VISIBLE_DEVICES=0 python3 chat.py --version='xinlai/LISA-13B-llama2-v0' --precision='fp16' --load_in_8bit
CUDA_VISIBLE_DEVICES=0 python3 chat.py --version='xinlai/LISA-13B-llama2-v0' --precision='fp16' --load_in_4bit
```

After that, input the text prompt and then the image path. For example，
```
- Please input your prompt: Where can the driver see the car speed in this image? Please output segmentation mask.
- Please input the image path: imgs/example1.jpg

- Please input your prompt: Can you segment the food that tastes spicy and hot?
- Please input the image path: imgs/example2.jpg
```
The results should be like:
<p align="center"> <img src="imgs/example1.jpg" width="22%"> <img src="vis_output/example1_masked_img_0.jpg" width="22%"> <img src="imgs/example2.jpg" width="25%"> <img src="vis_output/example2_masked_img_0.jpg" width="25%"> </p>

## Dataset
In ReasonSeg, we have collected 1218 images (239 train, 200 val, and 779 test). The training and validation sets can be download from <a href="https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy?usp=sharing">**this link**</a>. 

Each image is provided with an annotation JSON file:
```
image_1.jpg, image_1.json
image_2.jpg, image_2.json
...
image_n.jpg, image_n.json
```
Important keys contained in JSON files:
```
- "text": text instructions.
- "is_sentence": whether the text instructions are long sentences.
- "shapes": target polygons.
```

The elements of the "shapes" exhibit two categories, namely **"target"** and **"ignore"**. The former category is indispensable for evaluation, while the latter category denotes the ambiguous region and hence disregarded during the evaluation process. 

We provide a <a href="https://github.com/dvlab-research/LISA/blob/main/utils/data_proc_demo.py">**script**</a> that demonstrates how to process the annotations:
```
python3 utils/data_proc_demo.py
```

Besides, we leveraged GPT-3.5 for rephrasing instructions, so images in the training set may have **more than one instructions (but fewer than six)** in the "text" field. During training, users may randomly select one as the text query to obtain a better model.


## Citation 
If you find this project useful in your research, please consider citing:

```
@article{reason-seg,
  title={LISA: Reasoning Segmentation via Large Language Model},
  author={Xin Lai and Zhuotao Tian and Yukang Chen and Yanwei Li and Yuhui Yuan and Shu Liu and Jiaya Jia},
  journal={arXiv:2308.00692},
  year={2023}
}

```

## Acknowledgement
-  This work is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA) and [SAM](https://github.com/facebookresearch/segment-anything). 

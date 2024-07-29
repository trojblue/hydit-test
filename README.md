<!-- ## **HunyuanDiT** -->

# hydit-test


> testing training of HunYuan-Dit: https://github.com/Tencent/HunyuanDiT/



## Setup

(modified for Lepton usage)


1. get a conda environment:

```bash
cd ~/ && mkdir -p miniconda3 && wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -O ./miniconda3/miniconda.sh --no-check-certificate && bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3 && rm ./miniconda3/miniconda.sh && ./miniconda3/bin/conda init bash && source ~/.bashrc  && python -m pip install unibox ipykernel jupyter && python -m ipykernel sinstall --user --name=conda310
```


2. getting the repo:
```bash
git clone https://github.com/tencent/HunyuanDiT
cd HunyuanDiT
```

3. installing requirements:

> We recommend CUDA versions 11.7 and 12.0+.

```bash
# 1. Prepare conda environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate HunyuanDiT

# 3. Install pip dependencies
python -m pip install -r requirements.txt

# 4. Install flash attention v2 for acceleration (requires CUDA 11.6 or above)
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.1.2.post3

```

4. download pretrained models:

> no need to do this since it's already downloaded at `<dev>/huggingface`

```bash
export HF_HOME=/rmd/huggingface

python -m pip install "huggingface_hub[cli]"

huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.2 
```



## Training Setup


installing requirements:

```bash
pip install -e ./IndexKits
```

setup folder link:

```bash
mkdir -p /lv0/hydit_files/dataset
rm -rf ./dataset
ln -s /lv0/hydit_files/dataset ./dataset
```


## Training Steps


download demo data:

```bash
# (in root dir)
cd dataset && wget -O data_demo.zip https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip && unzip data_demo.zip -d ./ && mkdir ./porcelain/arrows ./porcelain/jsons && cd ..
```

demo data is recorded in csv format:

> ⚠️ Optional fields like MD5, width, and height can be omitted. If omitted, the script below will automatically calculate them. This process can be time-consuming when dealing with large-scale training data.
> 
> We utilize [Arrow](https://github.com/apache/arrow) for training data format, offering a standard and efficient in-memory data representation. A conversion script is provided to transform CSV files into Arrow format.


|    Fields       | Required  |  Description     |   Example   |
|:---------------:| :------:  |:----------------:|:-----------:|
|   `image_path`  | Required  |  image path               |     `./dataset/porcelain/images/0.png`        | 
|   `text_zh`     | Required  |    text               |  青花瓷风格，一只蓝色的鸟儿站在蓝色的花瓶上，周围点缀着白色花朵，背景是白色 | 
|   `md5`         | Optional  |    image md5 (Message Digest Algorithm 5)  |    `d41d8cd98f00b204e9800998ecf8427e`         | 
|   `width`       | Optional  |    image width    |     `1024 `       | 
|   `height`      | Optional  |    image height   |    ` 1024 `       | 




3. convert csv to arrow:

```bash  
# 3 Data conversion 
python ./hydit/data_loader/csv2arrow.py ./dataset/porcelain/csvfile/image_text.csv ./dataset/porcelain/arrows 1
```


  
4. Data Selection and Configuration File Creation 
    
    We configure the training data through YAML files. In these files, you can set up standard data processing strategies for filtering, copying, deduplicating, and more regarding the training data. For more details, see [./IndexKits](IndexKits/docs/MakeDataset.md).

    For a sample file, please refer to [file](./dataset/yamls/porcelain.yaml). For a full parameter configuration file, see [file](./IndexKits/docs/MakeDataset.md).
  
     
5. Create training data index file using YAML file.

> `idk`: the IndexKit installed previously
  
```shell
# Single Resolution Data Preparation
idk base -c dataset/yamls/porcelain.yaml -t dataset/porcelain/jsons/porcelain.json

# Multi Resolution Data Preparation     
idk multireso -c dataset/yamls/porcelain_mt.yaml -t dataset/porcelain/jsons/porcelain_mt.json
```
   
  The directory structure for `porcelain` dataset is:

  ```shell
   cd ./dataset
  
   porcelain
      ├──images/  (image files)
      │  ├──0.png
      │  ├──1.png
      │  ├──......
      ├──csvfile/  (csv files containing text-image pairs)
      │  ├──image_text.csv
      ├──arrows/  (arrow files containing all necessary training data)
      │  ├──00000.arrow
      │  ├──00001.arrow
      │  ├──......
      ├──jsons/  (final training data index files which read data from arrow files during training)
      │  ├──porcelain.json
      │  ├──porcelain_mt.json
   ```

### Full-parameter Training
  
  **Requirement:** 
  1. The minimum requriment is a single GPU with at least 20GB memory, but we recommend to use a GPU with about 30 GB memory to avoid host memory offloading. 
  2. Additionally, we encourage users to leverage the multiple GPUs across different nodes to speed up training on large datasets. 
  
  **Notice:**
  1. Personal users can also use the light-weight Kohya to finetune the model with about 16 GB memory. Currently, we are trying to further reduce the memory usage of our industry-level framework for personal users. 
  2. If you have enough GPU memory, please try to remove  `--cpu-offloading` or `--gradient-checkpointing` for less time costs.

  Specifically for distributed training, you have the flexibility to control **single-node** / **multi-node** training by adjusting parameters such as `--hostfile` and `--master_addr`. For more details, see [link](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node).

  ```shell
  # Single Resolution Training
  PYTHONPATH=./ sh hydit/train.sh --index-file dataset/porcelain/jsons/porcelain.json
  
  # Multi Resolution Training
  PYTHONPATH=./ sh hydit/train.sh --index-file dataset/porcelain/jsons/porcelain_mt.json --multireso --reso-step 64
  
  # Training with old version of HunyuanDiT (<= v1.1)
  PYTHONPATH=./ sh hydit/train_v1.1.sh --index-file dataset/porcelain/jsons/porcelain.json
  ```

  After checkpoints are saved, you can use the following command to evaluate the model.
  ```shell
  # Inference
    #   You should replace the 'log_EXP/xxx/checkpoints/final.pt' with your actual path.
  python sample_t2i.py --infer-mode fa --prompt "青花瓷风格，一只可爱的哈士奇" --no-enhance --dit-weight log_EXP/xxx/checkpoints/final.pt --load-key module
  
  # Old version of HunyuanDiT (<= v1.1)
  #   You should replace the 'log_EXP/xxx/checkpoints/final.pt' with your actual path.
  python sample_t2i.py --infer-mode fa --prompt "青花瓷风格，一只可爱的哈士奇" --model-root ./HunyuanDiT-v1.1 --use-style-cond --size-cond 1024 1024 --beta-end 0.03 --no-enhance --dit-weight log_EXP/xxx/checkpoints/final.pt --load-key module
  ```

### LoRA



We provide training and inference scripts for LoRA, detailed in the [./lora](./lora/README.md). 

  ```shell
  # Training for porcelain LoRA.
  PYTHONPATH=./ sh lora/train_lora.sh --index-file dataset/porcelain/jsons/porcelain.json

  # Inference using trained LORA weights.
  python sample_t2i.py --infer-mode fa --prompt "青花瓷风格，一只小狗"  --no-enhance --lora-ckpt log_EXP/001-lora_porcelain_ema_rank64/checkpoints/0001000.pt
  ```
 We offer two types of trained LoRA weights for `porcelain` and `jade`, see details at [links](https://huggingface.co/Tencent-Hunyuan/HYDiT-LoRA)
  ```shell
  cd HunyuanDiT
  # Use the huggingface-cli tool to download the model.
  huggingface-cli download Tencent-Hunyuan/HYDiT-LoRA --local-dir ./ckpts/t2i/lora
  
  # Quick start
  python sample_t2i.py --infer-mode fa --prompt "青花瓷风格，一只猫在追蝴蝶"  --no-enhance --load-key ema --lora-ckpt ./ckpts/t2i/lora/porcelain
  ```
 <table>
  <tr>
    <td colspan="4" align="center">Examples of training data</td>
  </tr>
  
  <tr>
    <td align="center"><img src="lora/asset/porcelain/train/0.png" alt="Image 0" width="200"/></td>
    <td align="center"><img src="lora/asset/porcelain/train/1.png" alt="Image 1" width="200"/></td>
    <td align="center"><img src="lora/asset/porcelain/train/2.png" alt="Image 2" width="200"/></td>
    <td align="center"><img src="lora/asset/porcelain/train/3.png" alt="Image 3" width="200"/></td>
  </tr>
  <tr>
    <td align="center">青花瓷风格，一只蓝色的鸟儿站在蓝色的花瓶上，周围点缀着白色花朵，背景是白色 （Porcelain style, a blue bird stands on a blue vase, surrounded by white flowers, with a white background.
）</td>
    <td align="center">青花瓷风格，这是一幅蓝白相间的陶瓷盘子，上面描绘着一只狐狸和它的幼崽在森林中漫步，背景是白色 （Porcelain style, this is a blue and white ceramic plate depicting a fox and its cubs strolling in the forest, with a white background.）</td>
    <td align="center">青花瓷风格，在黑色背景上，一只蓝色的狼站在蓝白相间的盘子上，周围是树木和月亮 （Porcelain style, on a black background, a blue wolf stands on a blue and white plate, surrounded by trees and the moon.）</td>
    <td align="center">青花瓷风格，在蓝色背景上，一只蓝色蝴蝶和白色花朵被放置在中央 （Porcelain style, on a blue background, a blue butterfly and white flowers are placed in the center.）</td>
  </tr>
  <tr>
    <td colspan="4" align="center">Examples of inference results</td>
  </tr>
  <tr>
    <td align="center"><img src="lora/asset/porcelain/inference/0.png" alt="Image 4" width="200"/></td>
    <td align="center"><img src="lora/asset/porcelain/inference/1.png" alt="Image 5" width="200"/></td>
    <td align="center"><img src="lora/asset/porcelain/inference/2.png" alt="Image 6" width="200"/></td>
    <td align="center"><img src="lora/asset/porcelain/inference/3.png" alt="Image 7" width="200"/></td>
  </tr>
  <tr>
    <td align="center">青花瓷风格，苏州园林 （Porcelain style,  Suzhou Gardens.）</td>
    <td align="center">青花瓷风格，一朵荷花 （Porcelain style,  a lotus flower.）</td>
    <td align="center">青花瓷风格，一只羊（Porcelain style, a sheep.）</td>
    <td align="center">青花瓷风格，一个女孩在雨中跳舞（Porcelain style, a girl dancing in the rain.）</td>
  </tr>
  
</table>


## 🔑 Inference

### 6GB GPU VRAM Inference
Running HunyuanDiT in under 6GB GPU VRAM is available now based on [diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/hunyuandit). Here we provide instructions and demo for your quick start.

> The 6GB version supports Nvidia Ampere architecture series graphics cards such as RTX 3070/3080/4080/4090, A100, and so on.

The only thing you need do is to install the following library:

```bash
pip install -U bitsandbytes
pip install git+https://github.com/huggingface/diffusers
pip install torch==2.0.0
```

Then you can enjoy your HunyuanDiT text-to-image journey under 6GB GPU VRAM directly!

Here is a demo for you.

```bash
cd HunyuanDiT

# Quick start
model_id=Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled
prompt=一个宇航员在骑马
infer_steps=50
guidance_scale=6
python3 lite/inference.py ${model_id} ${prompt} ${infer_steps} ${guidance_scale}
```

More details can be found in [./lite](lite/README.md).


### Using Gradio

Make sure the conda environment is activated before running the following command.

```shell
# By default, we start a Chinese UI. Using Flash Attention for acceleration.
python app/hydit_app.py --infer-mode fa

# You can disable the enhancement model if the GPU memory is insufficient.
# The enhancement will be unavailable until you restart the app without the `--no-enhance` flag. 
python app/hydit_app.py --no-enhance --infer-mode fa

# Start with English UI
python app/hydit_app.py --lang en --infer-mode fa

# Start a multi-turn T2I generation UI. 
# If your GPU memory is less than 32GB, use '--load-4bit' to enable 4-bit quantization, which requires at least 22GB of memory.
python app/multiTurnT2I_app.py --infer-mode fa
```
Then the demo can be accessed through http://0.0.0.0:443. It should be noted that the 0.0.0.0 here needs to be X.X.X.X with your server IP.

### Using 🤗 Diffusers

Please install PyTorch version 2.0 or higher in advance to satisfy the requirements of the specified version of the diffusers library.  

Install 🤗 diffusers, ensuring that the version is at least 0.28.1:

```shell
pip install git+https://github.com/huggingface/diffusers.git
```
or
```shell
pip install diffusers
```

You can generate images with both Chinese and English prompts using the following Python script:
```py
import torch
from diffusers import HunyuanDiTPipeline

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

# You may also use English prompt as HunyuanDiT supports both English and Chinese
# prompt = "An astronaut riding a horse"
prompt = "一个宇航员在骑马"
image = pipe(prompt).images[0]
```
You can use our distilled model to generate images even faster:

```py
import torch
from diffusers import HunyuanDiTPipeline

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled", torch_dtype=torch.float16)
pipe.to("cuda")

# You may also use English prompt as HunyuanDiT supports both English and Chinese
# prompt = "An astronaut riding a horse"
prompt = "一个宇航员在骑马"
image = pipe(prompt, num_inference_steps=25).images[0]
```
More details can be found in [HunyuanDiT-v1.2-Diffusers-Distilled](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled)

**More functions:** For other functions like LoRA and ControlNet, please have a look at the README of [./diffusers](diffusers).

### Using Command Line

We provide several commands to quick start: 

```shell
# Only Text-to-Image. Flash Attention mode
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚" --no-enhance

# Generate an image with other image sizes.
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚" --image-size 1280 768

# Prompt Enhancement + Text-to-Image. DialogGen loads with 4-bit quantization, but it may loss performance.
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚"  --load-4bit

```

More example prompts can be found in [example_prompts.txt](example_prompts.txt)

### More Configurations

We list some more useful configurations for easy usage:

|    Argument     |  Default  |                     Description                     |
|:---------------:|:---------:|:---------------------------------------------------:|
|   `--prompt`    |   None    |        The text prompt for image generation         |
| `--image-size`  | 1024 1024 |           The size of the generated image           |
|    `--seed`     |    42     |        The random seed for generating images        |
| `--infer-steps` |    100    |          The number of steps for sampling           |
|  `--negative`   |     -     |      The negative prompt for image generation       |
| `--infer-mode`  |   torch   |       The inference mode (torch, fa, or trt)        |
|   `--sampler`   |   ddpm    |    The diffusion sampler (ddpm, ddim, or dpmms)     |
| `--no-enhance`  |   False   |        Disable the prompt enhancement model         |
| `--model-root`  |   ckpts   |     The root directory of the model checkpoints     |
|  `--load-key`   |    ema    | Load the student model or EMA model (ema or module) |
|  `--load-4bit`  |   Fasle   |     Load DialogGen model with 4bit quantization     |

### Using ComfyUI

- Support two workflows: Standard ComfyUI and Diffusers Wrapper, with the former being recommended.
- Support HunyuanDiT-v1.1 and v1.2.
- Support module, lora and clip lora models trained by Kohya.
- Support module, lora models trained by HunyunDiT official training scripts.
- ControlNet is coming soon.

![Workflow](comfyui-hydit/img/workflow_v1.2_lora.png)
More details can be found in [./comfyui-hydit](comfyui-hydit/README.md)

### Using Kohya

We support custom codes for kohya_ss GUI, and sd-scripts training codes for HunyuanDiT.
![dreambooth](kohya_ss-hydit/img/dreambooth.png)
More details can be found in [./kohya_ss-hydit](kohya_ss-hydit/README.md)

### Using Previous versions

* **Hunyuan-DiT <= v1.1**

```shell
# ============================== v1.1 ==============================
# Download the model
huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.1 --local-dir ./HunyuanDiT-v1.1
# Inference with the model
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚" --model-root ./HunyuanDiT-v1.1 --use-style-cond --size-cond 1024 1024 --beta-end 0.03

# ============================== v1.0 ==============================
# Download the model
huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./HunyuanDiT-v1.0
# Inference with the model
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚" --model-root ./HunyuanDiT-v1.0 --use-style-cond --size-cond 1024 1024 --beta-end 0.03
```

## :building_construction: Adapter

### ControlNet

We provide training scripts for ControlNet, detailed in the [./controlnet](./controlnet/README.md). 

  ```shell
  # Training for canny ControlNet.
  PYTHONPATH=./ sh hydit/train_controlnet.sh
  ```
 We offer three types of trained ControlNet weights for `canny` `depth` and `pose`, see details at [links](https://huggingface.co/Tencent-Hunyuan/HYDiT-ControlNet)
  ```shell
  cd HunyuanDiT
  # Use the huggingface-cli tool to download the model.
  # We recommend using distilled weights as the base model for ControlNet inference, as our provided pretrained weights are trained on them.
  huggingface-cli download Tencent-Hunyuan/HYDiT-ControlNet-v1.2 --local-dir ./ckpts/t2i/controlnet
  huggingface-cli download Tencent-Hunyuan/Distillation-v1.2 ./pytorch_model_distill.pt --local-dir ./ckpts/t2i/model
  
  # Quick start
  python3 sample_controlnet.py --infer-mode fa --no-enhance --load-key distill --infer-steps 50 --control-type canny --prompt "在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围" --condition-image-path controlnet/asset/input/canny.jpg --control-weight 1.0
  
  ```
 
 <table>
  <tr>
    <td colspan="3" align="center">Condition Input</td>
  </tr>
  
   <tr>
    <td align="center">Canny ControlNet </td>
    <td align="center">Depth ControlNet </td>
    <td align="center">Pose ControlNet </td>
  </tr>

  <tr>
    <td align="center">在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围<br>（At night, an ancient Chinese-style lion statue stands in front of the hotel, its eyes gleaming as if guarding the building. The background is the hotel entrance at night, with a close-up, eye-level, and centered composition. This photo presents a realistic photographic style, embodies Chinese sculpture culture, and reveals a mysterious atmosphere.） </td>
    <td align="center">在茂密的森林中，一只黑白相间的熊猫静静地坐在绿树红花中，周围是山川和海洋。背景是白天的森林，光线充足。照片采用特写、平视和居中构图的方式，呈现出写实的效果<br>（In the dense forest, a black and white panda sits quietly among the green trees and red flowers, surrounded by mountains and oceans. The background is a daytime forest with ample light. The photo uses a close-up, eye-level, and centered composition to create a realistic effect.） </td>
    <td align="center">在白天的森林中，一位穿着绿色上衣的亚洲女性站在大象旁边。照片采用了中景、平视和居中构图的方式，呈现出写实的效果。这张照片蕴含了人物摄影文化，并展现了宁静的氛围<br>（In the daytime forest, an Asian woman wearing a green shirt stands beside an elephant. The photo uses a medium shot, eye-level, and centered composition to create a realistic effect. This picture embodies the character photography culture and conveys a serene atmosphere.） </td>
  </tr>

  <tr>
    <td align="center"><img src="controlnet/asset/input/canny.jpg" alt="Image 0" width="200"/></td>
    <td align="center"><img src="controlnet/asset/input/depth.jpg" alt="Image 1" width="200"/></td>
    <td align="center"><img src="controlnet/asset/input/pose.jpg" alt="Image 2" width="200"/></td>
    
  </tr>
  
  <tr>
    <td colspan="3" align="center">ControlNet Output</td>
  </tr>

  <tr>
    <td align="center"><img src="controlnet/asset/output/canny.jpg" alt="Image 3" width="200"/></td>
    <td align="center"><img src="controlnet/asset/output/depth.jpg" alt="Image 4" width="200"/></td>
    <td align="center"><img src="controlnet/asset/output/pose.jpg" alt="Image 5" width="200"/></td>
  </tr>
 
</table>

## :art: Hunyuan-Captioner
Hunyuan-Captioner meets the need of text-to-image techniques by maintaining a high degree of image-text consistency. It can generate high-quality image descriptions from a variety of angles, including object description, objects relationships, background information, image style, etc. Our code is based on [LLaVA](https://github.com/haotian-liu/LLaVA) implementation.

### Examples

<td align="center"><img src="./asset/caption_demo.jpg" alt="Image 3" width="1200"/></td>

### Instructions
a. Install dependencies
     
The dependencies and installation are basically the same as the [**base model**](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2).

b. Model download
```shell
# Use the huggingface-cli tool to download the model.
huggingface-cli download Tencent-Hunyuan/HunyuanCaptioner --local-dir ./ckpts/captioner
```

### Inference

Our model supports three different modes including: **directly generating Chinese caption**, **generating Chinese caption based on specific knowledge**, and **directly generating English caption**. The injected information can be either accurate cues or noisy labels (e.g., raw descriptions crawled from the internet). The model is capable of generating reliable and accurate descriptions based on both the inserted information and the image content.

|Mode           | Prompt Template                           |Description                           | 
| ---           | ---                                       | ---                                  |
|caption_zh     | 描述这张图片                               |Caption in Chinese                    | 
|insert_content | 根据提示词“{}”,描述这张图片                 |Caption with inserted knowledge| 
|caption_en     | Please describe the content of this image |Caption in English                    |
|               |                                           |                                      |
 

a. Single picture inference in Chinese

```bash
python mllm/caption_demo.py --mode "caption_zh" --image_file "mllm/images/demo1.png" --model_path "./ckpts/captioner"
```

b. Insert specific knowledge into caption

```bash
python mllm/caption_demo.py --mode "insert_content" --content "宫保鸡丁" --image_file "mllm/images/demo2.png" --model_path "./ckpts/captioner"
```

c. Single picture inference in English

```bash
python mllm/caption_demo.py --mode "caption_en" --image_file "mllm/images/demo3.png" --model_path "./ckpts/captioner"
```

d. Multiple pictures inference in Chinese

```bash
### Convert multiple pictures to csv file. 
python mllm/make_csv.py --img_dir "mllm/images" --input_file "mllm/images/demo.csv"

### Multiple pictures inference
python mllm/caption_demo.py --mode "caption_zh" --input_file "mllm/images/demo.csv" --output_file "mllm/images/demo_res.csv" --model_path "./ckpts/captioner"
```

(Optional) To convert the output csv file to Arrow format, please refer to [Data Preparation #3](#data-preparation) for detailed instructions. 


### Gradio 
To launch a Gradio demo locally, please run the following commands one by one. For more detailed instructions, please refer to [LLaVA](https://github.com/haotian-liu/LLaVA). 
```bash
cd mllm
python -m llava.serve.controller --host 0.0.0.0 --port 10000

python -m llava.serve.gradio_web_server --controller http://0.0.0.0:10000 --model-list-mode reload --port 443

python -m llava.serve.model_worker --host 0.0.0.0 --controller http://0.0.0.0:10000 --port 40000 --worker http://0.0.0.0:40000 --model-path "../ckpts/captioner" --model-name LlavaMistral
```
Then the demo can be accessed through http://0.0.0.0:443. It should be noted that the 0.0.0.0 here needs to be X.X.X.X with your server IP.

## 🚀 Acceleration (for Linux)

- We provide TensorRT version of HunyuanDiT for inference acceleration (faster than flash attention).
See [Tencent-Hunyuan/TensorRT-libs](https://huggingface.co/Tencent-Hunyuan/TensorRT-libs) for more details.

- We provide Distillation version of HunyuanDiT for inference acceleration.
See [Tencent-Hunyuan/Distillation](https://huggingface.co/Tencent-Hunyuan/Distillation) for more details.



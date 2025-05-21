# DEER 🦌: Dynamic Early Exit in Reasoning Models
[![arXiv](https://img.shields.io/badge/arXiv-2504.15895-b31b1b.svg)](https://arxiv.org/abs/2504.15895)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)](https://huggingface.co/)
[![vLLM](https://img.shields.io/badge/vLLM-Efficient%20LLM%20Inference-green)](https://github.com/vllm-project/vllm)

This is the repository of our paper: [Dynamic Early Exit in Reasoning Models](https://arxiv.org/abs/2504.15895).


<p align="center"> <img src="./images/deer.png" style="width: 85%;" id="title-icon">       </p>

**DEER** monitors model behavior at potential reasoning transition points and dynamically terminates the next reasoning chain’s generation when the model exhibits high confidence in a trial answer. It is consistently effective on 11 cutting-edge reasoning LLMs of varying series and sizes, reducing the length of CoT sequences by an average of **19.1% - 80.1%** while improving accuracy by **0.3% - 5.0%**.

---

## 🔥 **Latest Updates**
- **[2025/05/20]** Released DEER code for mathematical reasoning tasks (HuggingFace & vLLM).
- **[Coming Soon]** DEER for code generation tasks & Branch-Parallel Decoding Acceleration.

---

## 🎯 Key Results



| Method                           | GSM8K       | MATH-500    | AMC23       | AIME24      | GPQA-D      | Overall     |
|----------------------------------|-------------|-------------|-------------|-------------|-------------|-------------|
|                                  | Acc↑ Tok↓ CR↓ | Acc↑ Tok↓ CR↓ | Acc↑ Tok↓ CR↓ | Acc↑ Tok↓ CR↓ | Acc↑ Tok↓ CR↓ | Acc↑ CR↓  |
| DeepSeek-R1-Distill-Qwen-7B      |             |             |             |             |             |             |
| Vanilla                          | 89.6 1,484 100% | 87.4 3,858 100% | 78.8 6,792 100% | 41.7 13,765 100% | 23.7 10,247 100% | 64.2 100% |
| DEER                             | 90.6 917 61.8% | 89.8 2,143 55.5% | 85.0 4,451 65.5% | 49.2 9,839 71.5% | 31.3 5,469 53.4% | **69.2** **61.5%** |
| Qwen3-14B                        |             |             |             |             |             |             |
| Vanilla                          | 95.1 2,047 100% | 93.8 4,508 100% | 95.0 7,139 100% | 70.0 10,859 100% | 60.1 7,339 100% | 82.8 100% |
| DEER                             | 95.3 840 41.0% | 94.0 3,074 68.2% | 95.0 4,763 66.7% | 76.7 7,619 70.2% | 57.6 2,898 39.5% | **83.7** **57.1%** |
| QwQ-32B                          |             |             |             |             |             |             |
| Vanilla                          | 96.7 1,427 100% | 93.8 4,508 100% | 92.5 6,792 100% | 66.7 10,821 100% | 63.1 7,320 100% | **82.6** 100% |
| DEER                             | 96.3 977 68.5% | 94.6 3,316 73.6% | 95.0 5,782 85.1% | 70.0 10,097 93.3% | 64.1 6,163 84.2% | **84.0** **80.9%** |

Results on 3 SoTA reasoning models. "Acc" denotes accuracy, "Tok" denotes token count, and "CR" denotes compression rate. 

---


## 🚀 Quick Start
### 1. Installation
```bash
git clone https://github.com/yourusername/DEER.git
cd DEER
pip install -r requirements.txt
```

### 2. DEER on vLLM (Recommended)
Considering efficiency, we recommend reproducing the results using the code based on the **vLLM** framework.

#### For Most Reasoning Models
```
CUDA_VISIBLE_DEVICES=1 python ../vllm-deer.py \
    --model_name_or_path "./DeepSeek-R1-Distill-Qwen-14B" \
    --dataset_dir "./data/" \
    --output_path "./outputs" \
    --dataset "math" \
    --threshold 0.95 \
    --max_generated_tokens 16000 \
    --think_ratio 0.9 \
    --batch_size 2000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \ 
```
or run:
```bash
bash ./bashes/bash-vllm-deer.sh.
```


#### For Qwen3 Models

```
CUDA_VISIBLE_DEVICES=1 python ../vllm-deer-qwen3.py \
    --model_name_or_path "./Qwen3-4B" \
    --dataset_dir "./data/" \
    --output_path "./outputs" \
    --dataset "math" \
    --threshold 0.95 \
    --max_generated_tokens 16000 \
    --think_ratio 0.9 \
    --batch_size 2000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
```
or run:
```bash
bash ./bashes/bash-vllm-deer-qwen3.sh.
```
In our experiments, we found that Qwen3-series models tend to be over-confident in confidence prediction, so we made some modifications to its implementation. 

### 3. DEER on Transformers

For inference using HuggingFace Transformers (without vLLM), run:
```bash
bash ./bashes/bash-vanilla-deer.sh
```


## 📊 Evaluation

DEER currently supports evaluation on 7 reasoning benchmarks. The rule-based evaluation for these benchmarks is based on the code implementation from the project [LIMO](https://github.com/GAIR-NLP/LIMO/tree/main).


```
python ../check.py \
    --model_name_or_path "./DeepSeek-R1-Distill-Qwen-14B" \
    --data_name "math" \
    --generation_path "your_output.jsonl" \
```
or run
```bash
bash ./bashes/bash-check-correct.sh
```



## 📜 Citation
If you use DEER in your research, please cite our paper:
```bibtex
@misc{yang2025dynamicearlyexitreasoning,
      title={Dynamic Early Exit in Reasoning Models}, 
      author={Chenxu Yang and Qingyi Si and Yongjie Duan and Zheliang Zhu and Chenyu Zhu and Qiaowei Li and Zheng Lin and Li Cao and Weiping Wang},
      year={2025},
      eprint={2504.15895},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.15895}, 
}
```
## 💬 Community

Join our WeChat group for discussions:
<p align="center"> <img src="./images/WechatIMG.jpg" style="width: 85%;" id="title-icon">       </p>

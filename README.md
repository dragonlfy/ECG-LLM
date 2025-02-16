# ECG-LLM: Leveraging Large Language Models for Low-Quality ECG Signal Restoration

This repository contains the code for the paper "ECG-LLM: Leveraging Large Language Models for Low-Quality ECG Signal Restoration" published by IEEE in 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM).

[https://ieeexplore.ieee.org/document/10822461](https://ieeexplore.ieee.org/document/10822461).


1. Install Pytorch and necessary dependencies.

```
pip install -r requirements.txt
```

1. Put the datasets [[Google Drive]](https://drive.google.com/drive/folders/1EOf2FHA6Y18S-YpxuvCTOnbQ8b0IF1Q7?usp=sharing)
under the folder ```./dataset/```.

2. Download the large language models from [Hugging Face](https://huggingface.co/) and specify the model path using the `llm_ckp_dir` parameter in scripts.
   * [LLaMA2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)
   * [LLaMA3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

3. Train and evaluate the model. We provide all the above tasks under the folder ```./scripts/```.

```

# ecg forecasting
bash ./scripts/time_series_forecasting/long_term/AutoTimes_ETTh1.sh

```

> It is recommended that your graphics card computing power is greater than or equal to an RTX 3090-24G.


## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)

## Contact

If you have any questions or want to use the code, feel free to contact:
* Yong Liu (lf.liu@siat.ac.cn)

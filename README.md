<h1 align="center"><strong>UniVoice: Unifying Autoregressive ASR and Flow-Matching based TTS with Large Language Models</strong></h1>

<p align="center" style="font-size: 1 em; margin-top: 1em">
<a href="">Wenhao Guan<sup>1,2</sup></a>,
<a href="">Zhikang Niu<sup>2,3</sup></a>,
<a href="">Ziyue Jiang<sup>4<sup></a>,
<a href="">Kaidi Wang<sup>1<sup></a>,
<a href="">Peijie Chen<sup>1<sup></a>,
<a href="">Qingyang Hong<sup>1<sup></a>,
<a href="">Lin Li<sup>1<sup></a>,
<a href="">Xie Chen<sup>2,3<sup></a>,
</p>

<p align="center">
  <sup>1</sup>Xiamen University, China <br>
  <sup>2</sup>Shanghai Innovation Institute, China <br>
  <sup>3</sup>Shanghai Jiao Tong University, China <br>
  <sup>4</sup>Zhejiang University, China <br>
</p>


<div align="center">

  <a href="https://huggingface.co/guanwenhao/UniVoice" target="_blank">
    <img alt="Homepage" src="https://img.shields.io/badge/📃  Project%20Page-UniVoice-ffc107?color=ffc107&logoColor=white" />
  </a>
  </a>
  <a href="https://huggingface.co/guanwenhao/UniVoice" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-UniVoice-ffc107?color=ffc107&logoColor=white" />
  </a>

</div>
<div align="center">
  <a href="LICENSE">
    <img alt="Code License" src="https://img.shields.io/badge/Code_License-MIT-f5de53?&color=f5de53">
  </a>
</div>


<p align="center">
  <a href="#2-model-download"><b>Model Download</b></a> |
  <a href="#3-quick-start"><b>Quick Start</b></a> |
  <a href="#4-license"><b>License</b></a> |
  <a href="#5-citation"><b> Citation</b></a> <br>
  📄 Paper Link (<a href="https://arxiv.org/abs/"><b>UniVoice</b></a>)
</p>


## News

**🚀 2025.03.30**: The inference codes and checkpoints are released!



## 1. Introduction

Large language models (LLMs) have demonstrated promising performance in both automatic speech recognition (ASR) and text-to-speech (TTS) systems, gradually becoming the mainstream approach. However, most current approaches address these tasks separately rather than through a unified
framework. This work aims to integrate these two tasks into one unified model. Although discrete speech tokenization enables joint modeling, its inherent information loss limits performance in both recognition and generation. In this work, we present UniVoice, a unified LLM framework through continuous representations that seamlessly integrates speech recognition and synthesis within a single model. Our approach combines the strengths of autoregressive modeling for speech recognition with flow matching for high-quality generation. To mitigate the inherent divergence between autoregressive and flow-matching models, we further design a dual attention mechanism, which switches between a causal mask for recognition and a bidirectional attention mask for synthesis. Furthermore, the proposed text-prefix-conditioned speech infilling method enables high-fidelity zero-shot voice cloning. Experimental results demonstrate that our method can achieve or exceed current single-task modeling methods in both ASR and zero-shot TTS tasks. This work explores new possibilities for end-to-end speech understanding and generation.

In this work, we use [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) as the LLM backbone.



## 2. Model Download

### Huggingface

| Model                  | Download                                                                    |
|----------------------|-----------------------------------------------------------------------------|
| UniVoice-TTS | [🤗 Hugging Face](https://huggingface.co/guanwenhao/UniVoice/tree/main/univoice_tts) |
| UniVoice-All | [🤗 Hugging Face](https://huggingface.co/guanwenhao/UniVoice/tree/main/univoice_all) |

NOTE: We now only trained a model on a 960hs LibriSpeech datatset, We will release a model trained with more data in the future.

## 3. Quick Start
### Installation

On the basis of `Python >= 3.10` environment, install the necessary dependencies by running the following command:

```shell
git clone https://github.com/gwh22/UniVoice
cd UniVoice
# We recommend using conda to create a new environment.
conda create -n UniVoice python=3.10
conda activate UniVoice
# install cuda >= 11.8
conda install cudatoolkit=11.8 -c nvidia

pip install -r requirements.txt
```

### Inference
```shell
cd UniVoice
# for ASR task
sh scripts/infer_asr.sh
# for TTS task
sh scripts/infer_tts.sh
```


## 4. License

Our code is released under MIT License. If our work and codebase is useful for you, please cite as:
```
@article{guan2025univoice,
  title={UniVoice: Unifying Autoregressive ASR and Flow-Matching based TTS with Large Language Models},
  author={Guan, Wenhao and Niu, Zhikang and Jiang, Ziyue and Wang, Kaidi and Chen, Peijie and Hong, Qingyang and Li, Lin and Chen, Xie},
  journal={arXiv preprint arXiv:2510.04593},
  year={2025}
}
```

## 5. Acknowledgments

This codebase borrows from [DiT](https://github.com/facebookresearch/DiT), [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M), [F5-TTS](https://github.com/SWivid/F5-TTS), [Monoformer](https://github.com/MonoFormer/MonoFormer), [LLaVA](https://github.com/haotian-liu/LLaVA), and [Transformers](https://github.com/huggingface/transformers). Thanks for their great works.




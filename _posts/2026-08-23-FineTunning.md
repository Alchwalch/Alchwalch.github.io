---
title: "Fine-tunning (Classifying)"
date: 2026-08-23 18:30:00 +0900
math: true
categories: [DeepLearning]
tags: [DeepLearning]
---

## Introduction

## Reference

밑바닥부터 만들면서 배우는 LLM

[nanoGPT](https://github.com/karpathy/nanogpt)

핸즈온 LLM

## Dataset

rotten tomato dataset을 이용했다. 허깅페이스에서 가져옴.

```python
from datasets import load_dataset

ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
```

## Architecture

맨 뒤에 히든레이어 바꿈

## FineTunning

### 1차

이리저리 sdsd

### 2차

padding 부분말고 eos부분

### 3차

아 모델 로드하는거 깜빡하고 있었음 ㅋㅋ

그러기 위해선 padding부분을 다 지우고 eos로 교체할 필요가 있었음. (임의로 추가한 padding id가 모델 로드를 방해함)

### 4차

AdamW로 교체 + 그라디언트 클리핑

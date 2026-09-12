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

rotten tomato dataset을 이용했다. 허깅페이스에서 가져왔다.

```python
from datasets import load_dataset

ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
```

데이터셋의 구조를 불러오면 ds는 이렇게 생긴것을 볼 수 있다.

```text
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 8530
    })
    validation: Dataset({
        features: ['text', 'label'],
        num_rows: 1066
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 1066
    })
})
```

위 text부분을 tokenizing 시키고 그에 해당하는 label을 이어붙이는 Dataset클래스를 만든다.

```python
class MovieDataset(Dataset):
  def __init__(self, dataset, tokenizer):
    self.data=dataset
    self.encoded_texts=[
        tokenizer.encode(text) for text in dataset['text']
    ]

  def __getitem__(self, idx):
    encoded=self.encoded_texts[idx]
    label=self.data[idx]['label']
    return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)


  def __len__(self):
    return len(self.data)
```

마지막으로 텍스트 데이터를 뒤에 padding으로 다 채우는 collate_fn함수를 구현했다.

```python
def collate_fn(
    batch,
    eos_id=50256,
    pad_id=50257,
    allowed_max_len=None,
    device='cpu'
):
  batch_max_len=max(len(item[0])+1 for item in batch)
  inputs_lst, targets_lst=[] , []

  for item_text,tgt in batch:
    item_text = torch.cat([
        item_text,
        torch.tensor([eos_id], dtype=torch.long)
    ])

    inp=torch.cat([
        item_text,
        torch.tensor([pad_id]*(batch_max_len-len(item_text)), dtype=torch.long)
    ])

    if allowed_max_len is not None:
      inp=inp[:allowed_max_len]

    inputs_lst.append(inp)
    targets_lst.append(tgt)

  inputs=torch.stack(inputs_lst).to(device)
  targets=torch.tensor(targets_lst).to(device)

  return inputs, targets
```

## Architecture



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

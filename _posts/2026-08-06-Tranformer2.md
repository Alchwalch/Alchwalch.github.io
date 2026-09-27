---
title: "Transformer (Implementation)"
date: 2026-08-06 18:30:00 +0900
math: true
categories: [DeepLearning]
tags: [DeepLearning]
---

## 파운데이션

### Dataset

WMT14 en-de로 진행했다.

```python
from datasets import load_dataset

print("loading...")
dataset = load_dataset("regisss/wmt14-en-de-pre-processed")
print(dataset)
```

4548885 개의 train 시퀀스가 있지만 내가 군대에 있는것과 같은 이유로 train dataset을 16만개로 줄임
나머지는 val 2169 train 2999 개는 그대로 동일

토크나이저는 gpt2 tokenized을 이용 (밑바닥 부터 만들면서 배우는 LLM 차용)

```python
import tiktoken
tokenizer=tiktoken.get_encoding("gpt2")

BOS_ID = tokenizer.n_vocab
EOS_ID = tokenizer.n_vocab+1
PAD_ID = tokenizer.n_vocab+2
```

Dataset 구성

dataset의 아이템 마다 (en,de) pair로 구성시켜서 Dataset을 만들자. 코드는 아래와 같이 짰음.

```pytorch
class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.max_length = max_length
        self.pairs = []
        for item in dataset:
            item = item["translation"]
            src = tokenizer.encode(item["en"])[: max_length - 2]
            tgt = tokenizer.encode(item["de"])[: max_length - 2]
            self.pairs.append((
                torch.tensor([BOS_ID] + src + [EOS_ID]),
                torch.tensor([BOS_ID] + tgt + [EOS_ID]),
            ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
```

dataset    - train,val,test같은 dataset 불러오는 파라미터.
tokenizer  - 토크나이저 받는 파라미터
max_length - 최대 context 길이

원래 Dataset에서 Padding처리 할려 했었는데 collate_fn쓰면 간편하게 할 수 있다 함. 그래서 LLM서비스 도움으로 collate_fn작성하고 각각의 dataset을 받아와서 Dataloader을 만들어 버리는 함수(create_dataloader)를 만듦.

```pytorch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
  src_batch, tgt_batch = zip(*batch)

  src_batch = pad_sequence(
      src_batch,
      batch_first=True,
      padding_value=PAD_ID
  )

  tgt_batch = pad_sequence(
      tgt_batch,
      batch_first=True,
      padding_value=PAD_ID
  )

  return src_batch, tgt_batch

def create_dataloader(dataset, tokenizer, batch_size, max_length,shuffle=True,drop_last=True,collate_fn=None,num_workers=2):
  dataset=TranslationDataset(dataset,tokenizer,max_length)
  dataloader=DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      drop_last=drop_last,
      collate_fn=collate_fn,
      num_workers=num_workers
  )

  return dataloader
```

train,val,test 각각 만듦.

```python
train_loader=create_dataloader(
    train_dataset,
    tokenizer,
    batch_size,
    max_length,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn,
    num_workers=2
)

val_loader=create_dataloader(
    val_dataset,
    tokenizer,
    batch_size,
    max_length,
    shuffle=False,
    drop_last=True,
    collate_fn=collate_fn,
    num_workers=2
)

test_loader=create_dataloader(
    test_dataset,
    tokenizer,
    batch_size,
    max_length,
    shuffle=False,
    drop_last=True,
    collate_fn=collate_fn,
    num_workers=2
```

### Setting

```python
TRANSFORMER_BASE_CONFIG={
    "vocab_size":VOCAB_SIZE,
    "max_len":1024,
    "d_model":512,
    "n_heads":8,
    "n_layers":6,
    "drop_attn":0.0,
    "drop_res":0.1,
    "drop_emb":0.1,
    "pad_idx":PAD_ID,
    "qkv_bias": False,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

논문 구현을 우선으로함. drop_attn은 Self_attention에서 attention에서 dropout을 시키는 rate임.

optimizer는 Adam을 썼고 나머지는 논문에 나와있는 스케줄러 차용. (자세한건 코드)

```python
optimizer=optim.Adam(model.parameters(),lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda step: noam_lr_lambda(step, TRANSFORMER_BASE_CONFIG["d_model"], warmup_steps)
)
```

calc_loss_batch는 이터레이션 한 번 돌때마다 배치단위로 cross_entropy를 추적하는 함수임.

$$P(w_1, \dots , w_{m})=\Pi_{t=1}^m P(w_t \mid w_1, \dots, w_{t-1})$$

$$\sum_{t=1}^m \log P(w_t \mid w_1, \dots , w_{t-1})$$

이므로 정답에 대해 cross_entropy로 나타낼 수 있음. 밑바닥 부터 만들면서 배우는 LLM참고함.

```python
def calc_loss_batch(input_batch,target_batch,model,device):
  input_batch=input_batch.to(device)
  target_batch=target_batch.to(device)

  tgt_input=target_batch[:,:-1]
  tgt_output=target_batch[:,1:]
  out=model(input_batch,tgt_input)

  loss=F.cross_entropy(out.reshape(-1,out.shape[-1]),tgt_output.reshape(-1),ignore_index=PAD_ID)

  return loss
```

이거는 loader 전체에 대해서 나타내는 loss 함수. val과 test에서 썼음. 이것도 밑바닥 부터 만들면서 배우는 LLM 참고함.

```python
def calc_loss_loader(data_loader,model,device,num_batches=None):
  total_loss=0
  if len(data_loader) == 0:
    return float("nan")
  elif num_batches is None:
    num_batches=len(data_loader)
  else:
    num_batches=min(num_batches,len(data_loader))

  for i,(src,trg) in enumerate(data_loader):
    if i >= num_batches:
      break

    loss=calc_loss_batch(src,trg,model,device)
    total_loss+=loss.item()

  return total_loss/num_batches

def evaluate_model(model, data_loader, device,eval_iter):
    model.eval()
    with torch.no_grad(), torch.amp.autocast(device, enabled=use_amp):
        loss=calc_loss_loader(data_loader,model,device,eval_iter)
    model.train()
    return loss
```

### 결과

validation, train에 대한 평가를 매 epoch에서 기록함. 결과가 다음과 같아짐.

![transformer6](assets/img/transformer6.png)

문제점이 여러가지가 있었음.

- overfitting
- 학습속도가 너무 느림
- 저장 간격이 너무 긺

## 1차 개선

- overfitting

dropout_rate를 높이고 label_smoothing을 하기로 함. label_smoothing은 정답 레이블과 오답 레이블 각각 1과 0으로 부여하는 대신 0.9 0.1과 같이 비율을 부여하여 overfitting을 낮춘다. label_smoothing을 설정하지 않았을때 (label_smoothing=0.0) cross_entropy는 알다시피 다음과 같다.

$$\sum_{i=1}^{D} y_i \log x_i + (1-y_i) \cdot \log(1-x_i)$$

여기서 $y_i$는 원래 1아니면 0이 되어야 하지만 label_smoothing에 부여하는 값을 $\epsilon$이라 했을 때, $y_i=1-\epsilon으로 표현할 수 있다.

$$\sum_{i=1}^{D} (1-\epsilon) y_i \log x_i + \epsilon (1-y_i) \cdot \log(1-x_i)$$


```python
TRANSFORMER_BASE_CONFIG={
    "vocab_size":VOCAB_SIZE,
    "max_len":1024,
    "d_model":512,
    "n_heads":8,
    "n_layers":6,
    "drop_attn":0.1,
    "drop_res":0.2,
    "drop_emb":0.2,
    "pad_idx":PAD_ID,
    "qkv_bias": False,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

```python
def calc_loss_batch(input_batch,target_batch,model,device,label_smoothing=0.1):
  input_batch=input_batch.to(device)
  target_batch=target_batch.to(device)

  tgt_input=target_batch[:,:-1]
  tgt_output=target_batch[:,1:]
  out=model(input_batch,tgt_input)

  loss=F.cross_entropy(out.reshape(-1,out.shape[-1]),tgt_output.reshape(-1),ignore_index=PAD_ID, label_smoothing=label_smoothing)

  return loss
```

(원래 label_smoothing은 train에서만 진행해야 하는데 초반에 그거 생각 못하고 val,test를 포함해서 3개를 smoothing을 박아버림...)

- 학습속도 개선

GradScaler를 적용함.

```python
device = TRANSFORMER_BASE_CONFIG["device"]
use_amp = (device == "cuda")

scaler = torch.amp.GradScaler(device, enabled=use_amp)
```

이렇게 하고 모델 계산 할때마다 다음과 같이 적용 시키면 됨.

```python

# ...

with torch.amp.autocast(device, enabled=use_amp):
  loss = calc_loss_batch(input_batch, target_batch, model, device, label_smoothing=0.1)

# ...

```

- 저장간격

에포크마다가 아닌 500 이터레이션 마다 저장하는 걸로 수정.

결과는 다음과 같아짐.

![transformer7](assets/img/transformer7.png)

## 2차 개선

Tokenizer 개선함. GPT2에 사용한 BPE토크나이저 보단 아예 WMT14 en-de에 맞는 BPE를 만드는 것이 좋다고 생각함.

```python
train_dataset=dataset["train"].shuffle(seed=67).select(range(640_000))
val_dataset=dataset["validation"]
test_dataset=dataset["test"]

with open("en_de_combined.txt", "w", encoding="utf-8") as f:
    for item in train_dataset:   # train만
        pair = item["translation"]
        f.write(pair["en"].strip() + "\n")
        f.write(pair["de"].strip() + "\n")
```

train_dataset을 64만개로 늘렸다. train dataset을 뽑아서 en-de pair를 만든다.

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="en_de_combined.txt",
    model_prefix="wmt14_bpe",
    vocab_size=32000,
    model_type="bpe",
    character_coverage=1.0,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
)

sp = spm.SentencePieceProcessor()
sp.load("wmt14_bpe.model")

PAD_ID = sp.pad_id()
UNK_ID = sp.unk_id()
BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()
VOCAB_SIZE = sp.get_piece_size()
```

그런다음 SentencePieeceTrainer로 BPE알고리즘으로 토크나이징 시킨다. 
input이 위에서 만든 en-de 쌍이다.

```python
class SPTokenizerWrapper:
    def __init__(self, sp):
        self.sp = sp
    def encode(self, text):
        return self.sp.encode(text, out_type=int)
    def decode(self, ids):
        return self.sp.decode(ids)

tokenizer = SPTokenizerWrapper(sp)
```

그리고 위와 같이 tokenizer 클래스를 구현하면 끝난다.

![transformer8](assets/img/transformer8.png)

아래와 같은 python코드로 번역기를 만들어 봤다.
generate함수는 이전 블로그에 설명했다.

```python
print("번역기 (영어 -> 독일어)")

prompt=input("영어 문장을 입력하세요: ")

token_ids=generate(
    model,
    text_to_token_ids(prompt,tokenizer),
    max_new_token=128,
    max_length=128,
    temperature=1.0,
    top_k=5,
    device=TRANSFORMER_BASE_CONFIG["device"]
)

print("출력 텍스트: ", token_ids_to_text(token_ids,tokenizer))
```

결과는 아래와 같았다.

```text
번역기 (영어 -> 독일어) 
영어 문장을 입력하세요: The committee will vote on the proposal next week. 
출력 텍스트:  Der Ausschuß wird über die nächste Lesung abstimmen.

번역기 (영어 -> 독일어) 
영어 문장을 입력하세요: Rising energy prices have affected millions of households. 
출력 텍스트:  Die Preise für energiegeladene Energien sind von Bedeutung.

번역기 (영어 -> 독일어)
영어 문장을 입력하세요: The two countries agreed to strengthen trade relations.
출력 텍스트:  Die Beziehungen von Agenten vereinbarte sich darauf, die Handelsbeziehungen zu stärken.
```

결과를 분석해보자. 
Der Ausschuß wird über die nächste Lesung abstimmen. 이 문장을 영어로 번역하면 "The committee will vote on the next reading." 이다. "Der Ausschuß wird über die nächste" 이 부분이 The committee will vote on the next... 부분인데 proposal 하고 week가 빠진것 빼고는 잘 번역이 되었다.

아래도 비슷하다 Die Preise für energiegeladene Energien sind von Bedeutung.
는 The prices for energy-rich energy sources are important.라는 뜻이다. Rising energy prices 정도만 번역이 되었다.

뭔가 그럴듯하게 보이지만 사실 위에 문장은 데이터셋에 맞춘 문장이고 우리가 평상시 사용하는 문장을 대입해보면 성능이 좋지는 않다는것을 알 수가 있다.

```text
번역기 (영어 -> 독일어)
영어 문장을 입력하세요: I love you
출력 텍스트: Ich bin sehr gutes Wert, und wir habe Ihr gutes Wert gehabt, wenn es nicht gab.
```

Ich 빼고는 뭔 완전 딴 소리를 늘어지게 하고 있다.

이문제는 진단을 2가지로 봤다.
- dropout rate를 너무 높임
- 데이터셋을 너무 적게함

사실 두 번째 이유가 가장 크고 더 늘려보려고 했지만 내가 공부하는 환경(군대)에서는 더 건드리기 힘든 환경이여서 여기까지 했다.

## 코드

[Transformer](https://github.com/Alchwalch/Deep-Learning-Study/blob/main/NLP/Transformer.ipynb)

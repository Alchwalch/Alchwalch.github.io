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

밑바닥 부터 시작하는 딥러닝 6

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

[GPT2](https://github.com/Alchwalch/Alchwalch.github.io/blob/main/_posts/2026-08-08-GPT2.md)구조를 그대로 가져와 분류 파인튜닝에 맞게 마지막 레이어만 바꾸었다.

```python
num_classes=2
model.head = torch.nn.Linear(model.cfg.d_model, num_classes, bias=model.cfg.bias)
model.to(device)
```

마지막 ouput을 2개의 unit으로 나타내는 신경망으로 대체했다. 

## FineTunning

모델 전체 매겨변수가 아닌 일부 매개변수만 조정하여 파인튜닝을 하는 기법인 PEFT를 이용하였다. 모델 전체 파라미터를 동결 시킨다음에 마지막 레이어하고 마지막 어텐션 블록만 동결을 해제하였다.

```python
for param in model.parameters():
  param.requires_grad = False

for param in model.blocks[-1].parameters():
  param.requires_grad = True

for param in model.norm.parameters():
  param.requires_grad = True

for param in model.head.parameters():
  param.requires_grad = True
```

### 1차

훈련 loop를 돌릴 때 아래 함수를 이용하여 모델을 훈련시켰다.
```python
  def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch=input_batch.to(device)
    target_batch=target_batch.to(device)
  
    logits=model(input_batch)[:,-1,:]
    loss=F.cross_entropy(logits,target_batch)
    return loss
```

그리고 훈련을 시킨 결과 아래 그림과 같이 나왔다.

![FT1](assets/img/gpt2_ft_1.png)

매우 훈련이 잘 안 되었다.

### 2차

텍스트의 각각의 길이와 무관하게 무지성으로 가장 마지막 시퀀스([:,-1,:])에서 logit을 구한게 문제였다. 패딩토큰을 모두 고려한 곳에서 뽑아내지 않고 문장이 진짜 끝나는 부분인 <EOS>에서 token을 뽑게 했다.

먼저 sequence각각 eos가 있는 인덱스가 있는 부분을 True로 하고 나머지를 False로 한 다음 True의 인덱스에 접근할 수 있게끔 했다. 모든 sequence에는 무조건 Eos를 붙이게 했으므로 예외 케이스는 따로 적지는 않았다.

```python
def calc_loss_batch(input_batch, target_batch, model, device, eos_id=50256):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)


    is_eos = (input_batch == eos_id)
    seq_lengths = is_eos.float().argmax(dim=1)

    all_logits = model(input_batch)
    logits = all_logits[torch.arange(input_batch.size(0)), seq_lengths]

    loss = F.cross_entropy(logits, target_batch)
    return loss
```

결과는 다음과 같이 나왔다

![FT2](assets/img/gpt2_ft_2.png)

겉으로 봐선 학습이 잘 된것처럼 보이지만 그렇지는 않았다. 이걸 하기전에 데이터셋이 정답을 얼마나 잘 맞추는지 정확도를 특정하는 함수를 하나 만들었었다.

```python
def calc_accuracy_loader(data_loader, model, device, num_batches=None,eos_id=50256):
  model.eval()
  correct_predictions, num_examples=0,0

  if num_batches is None:
    num_batches=len(data_loader)

  num_batches=min(num_batches,len(data_loader))

  with torch.no_grad():
    for i, (inputs, targets) in enumerate(data_loader):
      if i >= num_batches:
        break

      else:
        inputs=inputs.to(device)
        targets=targets.to(device)

        is_eos = (inputs == eos_id)
        seq_lengths = is_eos.float().argmax(dim=1)

        all_logits=model(inputs)

        logits=all_logits[torch.arange(inputs.size(0)), seq_lengths]
        predicted_labels = torch.argmax(logits, dim=1)

        correct_predictions += (predicted_labels == targets).sum().item()
        num_examples += targets.size(0)

  accuracy=correct_predictions/num_examples
  return accuracy
```

이 함수를 통해 test accuracy를 돌려보니 50% 근처였다. 아직 학습이 안 된 것이다.

### 3차

알고보니 아주 큰 실수를 저질렀었다. 구조나 훈련루프는 다 만들어 놓고 GPT2 모델을 로드를 안 한 것이다. (ㅋㅋ...)

```python
model=GPT2(GPTConfig())
model.eval()
load_weights_into_gpt(model,params)
model.to(device)
```

이대로 그대로 실행하기는 좀 힘들었다. pad_id를 임의로 아이디를 더 부여를 했었는데 그게 모델을 로드하는 과정에서 걸린것이다. 그러기 위해선 padding부분을 다 지우고 eos로 교체할 필요가 있었다.

그 후의 실행결과는 매우 특이하게 되어 있었다.

![FT3](assets/img/gpt2_ft_3.png)

근데 accuracy가 여전히 50%를 맴돌고 있었다.

### 4차

여기서 크게 2가지를 개선했다. 먼저 Adam을 AdamW로 옵티마이저를 교체 하였다. 

AdamW에 대해 알기 전에 L2규제에 대해 알아보자.

**L2 규제(L2 regularization)** 는 손실 함수에 규제 항을 추가하여 파라미터를 지나치게 커지지 않도록 하여 overfitting을 막는다. 아래는 L2 정규화를 적용한 손실함수 식이다.

$$
L_{\mathrm{total}}(\theta)
=
L(\theta)
+
\frac{\lambda}{2}\|\theta\|_2^2
$$

위를 미분하면 아래와 같이 표현 된다.

$$
\nabla_\theta L_{\mathrm{total}}
=
\nabla_\theta L
+
\lambda\theta
$$

L2정규화로 파라미터를 업데이트 하는 방식은 다음식과 같다. 직관적으로 파라미터가 너무 크면 발산하는것을 막는것을 볼 수 있다.

$$\theta_{t+1}=\theta_{t}-\alpha(g_t+\lambda\theta_{t})$$

이렇게 기존적인 방식은 $g_t$ 에다가 $\lambda\theta_{t}$ 를 더하는 방식이고 *weight_decay*로 $\lambda$ 를 조절할 수 있다.

Adam 에서는 이 L2 정규화를 아래와 같이 각각 진행했다. Adam에 대해선 [이전 블로그](https://alchwalch.github.io/posts/adam/)에 잘 써져 있다.

$$
m_t \leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot (g_t+\lambda\theta_t)
$$

$$
v_t \leftarrow \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot (g_t+\lambda\theta_t)^2
$$

그러나 이렇게 하면 최종 식은 아래와 같기 때문에 본래 잘 작동하지 못하고 상쇄가 된다.

$$\theta_{t+1} \leftarrow \theta_{t} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

그래서 AdamW에서는 위 $m_t$와 $v_t$를 그대로 두고 최종식에서만 L2 정규화를 하자고 제안한다. 햇갈릴 수도 있으므로 좀더 명확한 이해를 위해를 돕기위해서 전체 수식을 적어 놓았다.

$$
m_t \leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t
$$

$$
v_t \leftarrow \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot g_t^2
$$

$$
\hat{m}_t \leftarrow m_t / (1-\beta_1^t)
$$

$$
\hat{v}_t \leftarrow v_t / (1-\beta_2^t)
$$

$$\theta_{t+1} \leftarrow \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha\lambda\theta_t$$

![FT4](assets/img/gpt2_ft_4.png)

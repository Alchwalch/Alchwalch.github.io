---
title: "Optimizer 톺아보기 1"
date: 2026-05-03 17:00:00 +0900
math: true
categories: [DeepLearning]
tags: [Learning and Optimizer]
---

## 머리말
딥러닝에 대해 공부를 하긴 했지만 흐름하고 공식정도만 보는정도로 그쳤다.
그래서 왜 이게 필요한건지 그리고 이게 어떤 효과를 줬는지에 대해서는 깊게 보지 않았다.
그리고 학부생이라 개발과 같은 막연한 딥러닝 프로젝트를 하는것 보단 이렇게 논문 읽으면서 하나하나 모델 만드는게 프로젝트로써 포트폴리오로써 더 이점일거 같았다.
그래도 **코드만 띡 하고 남기기는 내 생각을 펼치기 힘들다.**
이러한 연유로 **논문을 보면서 이해한 내용을 블로그로 간단하게 적으면서 정리**해 볼려고 한다.

## Intro
어떤 하나의 모델이 있을 때 그 모델의 결과와 우리가 의도한 목적이 부합할 수록 좋은 모델이라 말하고
그걸 수치로 나타낼 수 있는 함수는 목적함수라고 말할 수 있다. likelihood같이 높으면 높을 수록 좋다고 하는 경우도 있지만 
대부분의 경우에는 목적을 의미하는 target값과 그 결과 값을 일치 할때의 목적함수가 보통이기에 
낮을 수록 좋은 손실함수(혹은 criterion)를 쓴다.
<br/>
손실함수는 낮은 값일 수록 좋은 것이고 최저점은 극소값에 위치한다.
따라서 손실함수 결과가 극소값을 가질수록 유도하기 위해서 model의 parameter를 조작해야한다.
목적함수 결과를 가지고 parameter를 조정하는게 옵티마이저라 볼 수 있다.

옵티마이저 여러가지를 정리한 글인 [An overview of gradient descent optimization
algorithms](https://arxiv.org/pdf/1609.04747)을 참고했다.

## batch관련
모든 데이터를 한꺼번에 하는 Batch Gradiend descent<br/>
데이터 각각을 하는 Stochastic Gradient descent (SGD)<br/>
모든데이터가 아니고 데이터를 몇개씩 묶어서 하는 mini-batch Gradient descent가 있고<br/>
우리는 3번째가 제일 좋다고 배웠다.
<br/> <br/>
일단 내가 오랜만에 딥러닝을 공부하면서 햇갈렸던게 있는데
인풋 개수와 목적함수 결과 개수인데
*인풋이 여러개든 한개든 목적함수 결과 개수는 이터레이션 당 한 개*이다.<br/>
인풋에 대한 각각의 결과는 criterion에 합쳐진다.<br/>
옵티마이저 계산이 끝나면 파라미터가 조정되는 일이 벌어지기 때문에
1번째는 너무 없고 2번째는 너무 많아서 3번째로 해서 셔플시키면서 하는게 제일 좋다.

## 옵티마이저
### Vanila
$$
\theta_{t+1} = \theta_t - \eta \nabla_{\theta} L(\theta_t)
$$
```python
class SGD:
  def __init__(self, lr=0.01):
    self.lr = lr

  def update(self, params, grads):
    for key in params.keys():
      params[key]-=self.lr*grads[key]
```
이걸 선택하면 문제가 생기는데 대표적으로 4가지가 있다. 글에서 발췌한 말에 따르면
- 적절한 learning rate(lr)을 찾기 힘듦
- lr 조정할 수 없음
- 모든 파라미터에 똑같은 lr 적용
- saddle point라고 어떤 dimension에선 하강하지만 어떤 dimension에서 올라갈때 막힐 수도 있음.

4번째는 처음엔 이게 왜 하면서 이해 못했는데 Adam 공부하면서 알게 되었다. <br/>
*직관적으로 내려가야 할 곳에 안 내려가고 왔다갔다* 하면서 뻐팅기면서 방해가 된다.

### Momentum
lr조정하면서 적절한 lr을 찾으면 된다.
그래서 물리학과 합쳐서 속도와 중력가속도를 이용했다.
$$
v_t = \gamma v_{t-1} + \eta \nabla_{\theta} J(\theta)
$$
$$
\theta = \theta - v_t
$$
$\gamma$가 momentum 하이퍼파라미터로 보통 0.9로 표현한다. 중력가속도이다.<br/>
더 빠른 수렴(convergence)와 더 적은 진동(oscillation)을 보여준다.
```python
class Momentum:
  def __init__(self,lr=0.01, momentum=0.9):
    self.lr=lr
    self.momentum=momentum
    self.v=None

  def update(self, params, grads):
    if self.v is None:
      self.v={}
      for key,val in params.items():
        self.v[key]=np.zeros_like(val)

    for key in params.keys():
      self.v[key]=self.momentum*self.v[key]+self.lr*grads[key]
      params[key]-=self.v[key]
```

### Nesterov accelerated gradient (NAG)

근데 이게 관성의 법칙때문에 올라가고 있는데 속도가 나중에 낮춰져서 좀 느리게 갈 수도 있다는 문제가 있다.
그래서 대략적인 **next position**을 줘서 거기서 기울기를 계산 하게 하자는 아이디어다.
그러면 좀 더 빨리 수렴 할 수 있을 것이다. (increased responsiveness)
$$
v_t = \gamma v_{t-1} + \eta \nabla_{\theta} J(\theta - \gamma v_{t-1})
$$
$$
\theta = \theta - v_t
$$
처음보는 옵티마이저였어서 기존 템플릿으로 코드를 짤려 했다. 그러나 걸리는게 있었다.
파라미터가 현재 위치에 해당되는 값($\theta$)이 아니라<br/>
미래위치에 해당되는 값($\theta - \gamma v_{t-1}$)에서 일어난다는 것이다.

그래서 내가 생각한 아이디어는 함수 하나를 더 만들어서 미래위치에 해당되는 값(tmp)을 내보내는 함수를 만들고 (prev 함수) <br/>
그 값(tmp)에서 backward시켜서 grads를 구하고 거기서 계산을 하면 되는 것이다.
기울기를 구하는 일은 옵티마이저와 별개의 일이기 때문에 이것이 맞다고 생각하고 코드를 짰다.
```python
class NAG:
  def __init__(self,lr=0.01, momentum=0.9):
    self.lr=lr
    self.momentum=momentum
    self.v=None


  def prev(self,params):
    if self.v is None:
      self.v={}

      for key,val in params.items():
        self.v[key]=np.zeros_like(val)

    tmp={}

    for key in params.keys():
      tmp[key]=params[key]-self.momentum*self.v[key]

    return tmp

  def update(self, params, grads):

    for key in params.keys():
      self.v[key]=self.momentum*self.v[key]+self.lr*grads[key]
      params[key]-=self.v[key]
```
파이토치는 아래와 같이 표현할 수 있다.
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```
nesterov=True 는 prev함수 역할이랑 같은 역할을 한다.
python은 interpreter로 돌아가는 고능아 언어이기 때문에
보통 backward뒤에 optim을 쓰지만 문제는 없다. <br/>
뭐 안 보고 스스로 생각해서 썼는데 진짜 이렇게 하니깐 좀 신기했다.

### Adagrad

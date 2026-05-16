---
title: "Optimizer 톺아보기 1"
date: 2026-05-01 17:00:00 +0900
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
그러나 momentum계열의 옵티마이저는 못 고친 부분이 하나가 있다.
- 모든 파라미터에 똑같은 lr 적용

파라미터마다 다른 학습률을 적용할 필요가있으며 그렇게 등장한게 Adadelta이다.
$$
g_t = \nabla_{\theta_t} J(\theta_t)
$$
$$
G_t = g_t \odot g_t
$$
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t
$$
더 정확히 하면
$$
\theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,ii} + \epsilon}} \cdot g_{t,i}
$$
이렇게 하면 자동으로 각각의 파라미터에 대한 학습률을 조정할 수 있다.<br/>
(eliminates the need to manually tune the learning rate.)

```python
class Adagrad:
  def __init__(self, lr=0.01):
    self.lr = lr
    self.h=None

  def update(self, params, grads):
    if self.h is None:
      self.h={}
      for key,val in params.items():
        self.h[key]=np.zeros_like(val)

    for key in params.keys():
      self.h[key]+=grads[key]*grads[key]
      params[key]-=self.lr*grads[key]/(np.sqrt(self.h[key])+1e-7)
```

아무튼 $\frac{1}{\sqrt{G_t + \epsilon}}$ 이렇게 곱하면서 학습률을 조정하는게 Ada계열이라 볼 수 있다.<br/>
참고로 $/epsilon$ 은 루트값이 0보다 커야한다는 조건때문에 있는 아주 작은 수이다.

### Adadelta
Adagrad의 문제점은 학습이 어떻게 되든간에 lr을 계속 감소시킨다는 것이다. 그래서 결과 도달이 힘들 수 있다.
그래서 Adadelta는 계속 누적시키는 대신에 평균을 사용하자고 제안한다. (recursively
defined as a decaying average of all past squared gradients)

$$
E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma)g^2_t
$$
$$
\Delta\theta_t = -\frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t
$$
$$
\theta_{t+1} = \theta_t + \Delta\theta_t
$$

$\gamma$ 는 momentum term과 비슷한 의미를 가지며 0.9를 가진다.
엄밀한 기댓값보단 대략적인 기댓값으로 이해하면 쉽다.

아무튼 $E[g^2]_t$대신에 $RMS[g]_t$로 쓸 수 있다.
$$
\Delta\theta_t = -\frac{\eta}{\sqrt{RMS[g]_t + \epsilon}}g_t
$$

이때, 이거 만든 저자가 "the update should have the same hypothetical units as the parameter" 라고 한다. <br/>
$\Delta\theta_t$가 좌변에 있고 우변에 없다는 것이 불편하다는 말이다. <br/>
Hessian을 고려하면 납득 가능한 얘기다.
$$
E[\Delta\theta^2]_t = \gamma E[\Delta\theta^2]_{t-1} + (1-\gamma)\Delta\theta^2_t
$$
$$
RMS[\Delta\theta]_t = \sqrt{E[\Delta\theta^2]_t + \epsilon}
$$
$$
\Delta\theta_t = -\frac{RMS[\Delta\theta]_{t-1}}{RMS[g]_t} g_t
$$
$$
\theta_{t+1} = \theta_t + \Delta\theta_t
$$

```python
class Adadelta:
  def __init__(self, momentum=0.9):
    self.momentum=momentum
    self.msdx=None
    self.msg=None

  def update(self,params,grads):
    if self.msg is None:
      self.msdx={}
      self.msg={}
      for key,val in params.items():
        self.msdx[key]=np.zeros_like(val)
        self.msg[key]=np.zeros_like(val)


    for key in params.keys():
      self.msg[key]=self.momentum*self.msg[key]+(1-self.momentum)*grads[key]*grads[key]
      rms_dx = np.sqrt(self.msdx[key]+1e-7)
      rms_g = np.sqrt(self.msg[key]+1e-7)
      dx=-(rms_dx / rms_g) * grads[key]
      self.msdx[key]=self.momentum*self.msdx[key]+(1-self.momentum)*dx*dx
      params[key]+=dx
```
### RMSprop
근데 학습을 돌려보면 아는게
unit을 맞출려고 하는 부분이 개삽질이라는 것을 알 수 있다. 
초반에 업데이트가 너무 느리고 학습률 조정이 불가능해
실무에서는 거의 사용되지 않는다. 이러한 한계를 극복하고자 RMSprop이 등장한다.
$$
\Delta\theta_t = -\frac{\eta}{\sqrt{RMS[g]_t + \epsilon}}g_t
$$
그냥 여기서 끝내면 된다.
```python
class RMSprop:
  def __init__(self, lr=0.01, momentum=0.9):
    self.lr=lr
    self.momentum=momentum
    self.e=None

  def update(self,params,grads):
    if self.e is None:
      self.e={}
      for key,val in params.items():
        self.e[key]=np.zeros_like(val)

    for key in params.keys():
      self.e[key]=self.momentum*self.e[key]+(1-self.momentum)*grads[key]*grads[key]
      params[key]-=self.lr*grads[key]/(np.sqrt(self.e[key])+1e-7)
```
## 마무리

너무 길어서 분량을 나눴다.
[코드](https://github.com/Alchwalch/Deep-Learning-Study/blob/main/Learning%20and%20Optimization/Optimizer.ipynb)

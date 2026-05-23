---
title: "Batch Normalization"
date: 2026-05-10 14:00:00 +0900
math: true
categories: [DeepLearning]
tags: [Learning and Optimizer]
---

## Reference
[Batch Normalization: Accelerating Deep Network Training b
y
Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167)

[Understanding the backward pass through Batch Normalization Layer](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)

## 서론

Batch Normalization을 쓰면 초깃값이 뭣같아도 괜찮아진다. 아니면 성능이 좋아진다. 간단한 공식 등 이정도만 알고 있었다. 이번에 다시 처음부터 공부하는 김에 이걸 왜 쓰는건지 그리고 어떻게 작동하는지 알아보려고 한다.

## Introduction

MLP와 같은것을 쓰는 딥러닝에선 parameter들로 작은변화가 일어나고 깊이가 깊어질 수록 그 변화는 증폭된다. 특히 위 논문에서는 **분포(Distribution)** 가 달라지는 점을 지적한다. 각 층(layer)마다 새로운 분포에 대해 적응을 해야 하는데
이때, covariate shift 문제가 발생한다고 말한다. Training을 더 효율적으로 하기 위해선 test data와 training간 **분포를 고정**시켜야 될 필요가 있다. 즉. Distribution을 Fix시킬 필요가 있다. 

분포를 고정시키지 않으면 어떤일이 발생할까? Logistic Regression을 생각하면 알 수가 있다. $\frac{1}{1+e^{-x}}$ 시그모이드 함수는 $\lvert x \rvert$값이 커질 수록 시그모이드 함수에 대한 도함수는 0으로 수렴하는 경향이 있다. 이때 입력값인 x는
매 레이어 마다 W,b와 같은 파라미터에 영향을 받기 때문에 분포가 안정적으로 안 나와서 좌우로 편향되어 나타날 수도 있다. 이때 출력값이 **saturated regime**(포화상태)로 들어간다고 한다. saturation problem 은 convergence하는데 큰 방해가 될 수 있다. 이러한 *Internal Covariate Shift*를 제거하기 위해서 매 레이어를 갈때마다 분포를 고정시켜야 될 이유를 확실히 알 수 있다.

*Batch Normalization*은 평균과 분산을 각각 0(zero-means)과 1(unit-variance)로 만들어 버리기 때문에 gradient flow에 큰 이점이 된다. 그러므로, **초깃값과 parameter의 크기에 대한 의존도(dependence)를 줄이고** **높은 learning rate(학습률)** 를 부담없이 사용할 수도 있다.

## Whitening

배치 정규화 전 Whitening에 대해서 설명한다. whitening은 매 레이어마다 평균을 0 분산을 1로 만들어 주고 각각의 입력에 대한 값들을 decorrelate 시켜주는 것이다. 이렇게 하면 분포를 고정시킬 수 있다. 그러나 문제가 크게 2가지가 있다. gradient를 구하는데 문제가 된다. <br/>
예를 들어, $Y=Wx+b$ 가 있다고 할 때, whitening 할때 $Y-E[Y]$를 할 것이므로 b에 대한 gradient를 구할 수 없는 문제가 발생한다. bias가 없고 파라미터가 분포가 계속 고정된다고 가정해도 문제가 생긴다. Batch집합에 있는 데이터를 x라고 하고 그 집합을 *X*라고 했을때, 
그것에 대한 Covariance를 구해야 된다는 문제가 발생한다. Covariance를 미분한다는 불가능에 가깝기 때문에 whitening은 사용하기 힘든 방법이다.

## Normalization via Mini-Batch

논문을 읽으면서 알게 되었는데, Batch Normalization에서 Normalization보다 중요한 키워드는 **Batch**라는 것이다. 입력 데이터 하나에서 정규화를 수행하는 것이 아니라 Batch로 묶인 집합 X에 대해서 정규화를 하는것이다. 위에서 말했다시피 공분산(Covariance)구하는 것은 불가능하다. Batch Normalization은
d-dimension인 입력 $x = (x^{(1)} \cdots x^{(d)})$에 대해서 각각의 차원에 대해 정규화를 시키는 것이다. 식으로 표현하면 아래와 같다.

$$
\hat{x}^{(k)} = \frac{x^{(k)} - E[x^{(k)}]}{\sqrt{\text{Var}[x^{(k)}]}}
$$

여기서 포인트는 입력 데이터 하나하나에서 정규화를 진행한다기 보단 **배치 집합에서 각 차원에 대한 값(per-dimension variance)을 정규화** 시킨 것이다. 공분산 행렬 계산 없이도 mini-batch 단위로 안정적으로 정규화할 수 있고, convergence속도를 높일 수 있다.

여기서 끝내면 레이어가 표현하고 싶은 특징을 잘 표현하지 못할 수도 있다. 매 레이어가 평균이 0이고 분산이 1이면 좀 뭐시기 할 것일 뿐더러 whitening의 문제점인 bias가 무시될 수도 있다.<br/>
이때 두 파라미터를 붙여서 분포는 fix하지만 평균과 분산을 다양하게 표현할 수 있다.

$$
y^{(k)} = \gamma^{(k)}\hat{x}^{(k)} + \beta^{(k)}
$$

알고리즘은 아래와 같다.

![Algorithm1](assets/img/batch1.png)

### Backpropagation

![Backpropagation](assets/img/batch2.png)

아래 순서대로 공식을 유도할 수 있다. 자세한 유도 과정은 위에 나와있는 kratzert의 블로그를 참고하자.

$$
y^{(k)} = \gamma^{(k)}\hat{x}^{(k)} + \beta^{(k)}
$$

이므로 $\beta$를 구하고 그 다음 $\gamma$를 구해야 한다.

$$
\frac{\partial \ell}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i}
$$

$$
\frac{\partial \ell}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i} \cdot \hat{x}_i
$$

이것을 구하면 $\hat{x}$도 유도할 수 있다.

$$
\frac{\partial \ell}{\partial \hat{x}_i} = \frac{\partial \ell}{\partial y_i} \cdot \gamma
$$

이제 $(x-\mu)$와 $(\sigma_\mathcal{B}^2 + \epsilon)^{-1/2}$에 대해 곱하는 것으로 각각 나눠서 계산하면 먼저 분산값을 유도할 수 있다.

$$
\frac{\partial \ell}{\partial \sigma_\mathcal{B}^2} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial \hat{x}_i} \cdot (x_i - \mu_\mathcal{B}) \cdot \frac{-1}{2}(\sigma_\mathcal{B}^2 + \epsilon)^{-3/2}
$$

$(x-\mu)$에서 제곱하고 N으로 나눈것이 분산이기 때문에 반대로 역전파 시켜서 나눠서 역전파 시킨 것을 다시 합친다.

$$
\frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} + \frac{\partial \ell}{\partial \sigma_\mathcal{B}^2} \cdot \frac{2(x_i - \mu_\mathcal{B})}{m}
$$

위의 공식의 차원은 (N,D)이지만 $\mu$에 대한 차원은 (D,)이다. $\sum_{i=1}^{m} \frac{2(x_i - \mu_\mathcal{B})}{m}=0$이므로 논문 보다 간결하게 아래와 같이 식을 나타낼 수 있다.

$$
\frac{\partial \ell}{\partial \mu_\mathcal{B}} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}
$$

마지막으로 입력에 대한 역전파를 구할 수 있다.

$$
\frac{\partial \ell}{\partial x_i} = \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} + \frac{\partial \ell}{\partial \sigma_\mathcal{B}^2} \cdot \frac{2(x_i - \mu_\mathcal{B})}{m} + \frac{\partial \ell}{\partial \mu_\mathcal{B}} \cdot \frac{1}{m}
$$

## Inference with Batch-Normalization

학습시와 달리 Inference(test data)시에는 mini-Batch를 사용할 수 없으며 또한 그렇게 해서도 안 된다. Inference시에는 결과를 deterministic하게 하기 위하여 고정된 평균과 분산을 이용하여 구하게 된다.
이때, **고정된 값은 학습에 이용된 데이터들의 모집단** (population)을 이용하여 구한다. 

$$
E[x] \leftarrow E_\mathcal{B}[\mu_\mathcal{B}]
$$

$$
\text{Var}[x] \leftarrow \frac{m}{m-1} E_\mathcal{B}[\sigma_\mathcal{B}^2]
$$

(참고로 $\frac{m}{m-1}$은 실무에선 잘 쓰지 않고 생략하는 편이다.)

그 다음 똑같이 학습된 파라미터 $\gamma$와 $\beta$를 이용하여 결과를 도출한다.

![Algorithm2](assets/img/batch3.png)

## Convolutional Net에서 Batch-Normalization

단순 2차원이 아닌 3차원 이상일때 어떻게 처리하는지 궁금할 수 있다. CNN에서는 channel을 기준으로 정규화 한다. 배치수가 n이고 이미지 크기가 p x q 일때 정규화 한 번 할때 계산되는 픽셀 수는 n x p x q 이다.

pytorch 에선 Linear 계층을 정규화 할 때는 nn.BatchNorm1d를 사용하지만
CNN같이 4차원을 다룰 때는 nn.BatchNorm2d를 사용한다.<br/>

## 실습

[Batch Normalization Colab](https://github.com/Alchwalch/Deep-Learning-Study/blob/main/Learning%20and%20Optimization/Batch_Normalization.ipynb)

처음에 Sigmoid를 활성화 함수로 하고 실습했더니 Batch-Normalization을 안 썼을때가 성능이 더 좋은 것을 볼 수 있다. (위에 링크 참고) <br/>
saturation problem을 해결하기 위해 쓴거 아닌가 라고 생각했는데 오히려 비선형성을 죽여버린 것이다. <br/>
아마 층이 얕아서 saturation problem보단 비선형성이 두드러지는 곳에 쓰여서 결과가 안 좋게 나온 것 같다.

그러나 ReLU랑 같이 쓸 때는 매우 성능이 좋았다. 초깃값 민감도가 해결된 것 같다.

## Conclusion

매 레이어 마다 입력에 대한 분포가 달라질 수 있어 배치 정규화로 분포를 안정화한다.

그에 대한 이점은 크게 이정도 있다고 보면 된다.
-초깃값에 대한 민감도가 줄어듦
-높은 학습률로 최적화 가능

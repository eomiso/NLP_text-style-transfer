This is a pytorch implementation of NeuLIPS Paper **[Style Transfer from Non-Parallel Text by Cross-Alignment](https://arxiv.org/pdf/1705.09655v2.pdf)**

Detailed Information about the implementation and experiments are in this [blog post](https://blog.diyaml.com/teampost/Text-Style-Transfer/)


## Index
1. [Introduction](#intro)
2. [Paper Summary](#paper_summary)
3. [Implementation](#implementation)
4. [Experiment](#experiment)


## 1. <a name="intro">Introduction</a>
Vision 분야에서 Style Transfer는 직관적인 방법론으로 두루 쓰이고 있습니다. Pix2pix - CycleGAN - CartoonGAN 으로 이어지는 image-to-image translation의 발전은 __병렬 데이터 없이도__ 하나의 도메인에서 다른 도메인으로 이미지 변환이 가능하다는 것을 보여주었습니다. ([DIYA의 CartoonGAN 포스트 바로가기](https://blog.diyaml.com/teampost/Improving-CartoonGAN/#image)) 여기서 저희는 이를 NLP에 적용하면 어떻게 될 것인가 하는 질문을 던지게 되었습니다. 병렬 데이터 없이도 language style transfer가 가능하다면, parallel data가 부족한 언어 도메인에 대해서도 _generative한 method를 이용한 data augmentation_ 등 다양한 시도를 할 수 있을 것으로 예상하기 때문입니다.

## 2. <a name="paper_summary">Paper Summary</a>
### 2.1. Introduction
*Language Style Transfer* 논문은 Non-Parallel 한 Text를 사용하여 문장의 내용은 그대로 두고 style 만 바꾸는 방법을 찾는 문제를 해결하고자 시도하였습니다. 서로 다른 text corpora 사이에 공통된 latent content distribution($z$)이 있을 것이라 가정하고 이를 align하여 Style Transfer를 하는 방법을 제시하고자 합니다.
구체적으로 latent content(잠재된 내용)를 풍부하게 하기위해<sup>[1*](#richness_in_z)</sup> typical한 VAE를 사용하지 않고 aligned auto-encoder를 사용하였습니다. 

### 2.2. Fomulation
Style($y$)과 content($z$)는 각기 다른 distribution ($p(y)$, $p(z)$)에서 생성되었다고 가정합니다. 여기에 관측 또는 생성되는 텍스트 $x$는 style과 content가 각각 주어져 있다고 가정한 상태에서 생성된 것으로 봅니다($p(x_1|y_1)$ 과 $p(x_2|y_2)$). 이때 논문이 해결하고자 하는 문제는 바로 $p(x_1|x_2; y_1, y_2)$ 과 $p(x_2|x_1; y_1, y_2)$의 분포를 갖는 transfer function을 찾는 것입니다. 이때 논문에서는 latent content(잠재된 내용)를 풍부하게 하기위해<sup>[1*](#richness_in_z)</sup> typical한 VAE를 사용하지 않고 aligned auto-encoder를 사용하였습니다. <sup>[1*](#richness_in_z)</sup>에서 언급한 바대로 다음과 같은 proposition으로 분포들을 제한 합니다. <br/><br/>
$\text{ Proposition 1. In the generative framework, $x_1$ and $x_2$'s joint distribution}$ <br/> $\text{can be recovered from  their marginals only if for any different $y,y' \in Y$,}$<br/> $\text{distributions $p(x|y)$ and $p(x|y')$ are different}$<br/>

### 2.3. Method
Vision task 와는 달리 language는 연속적이지 않기 때문에(discrete), style transfer function을 직접 learn하거나 estimate할 수 없습니다. 따라서 latent space를 이용해야 하는데 여기서 적용 가능한 알고리즘이 바로 auto-encoder 입니다. 두 데이터 셋이 같은 z 분포로 부터 생성되었음을 가정하기 위해 VAE를 사용할 수도 있으나, VAE는 content의 prior density를 Normal으로 간단하게 가정함으로 Proposition1에 대한 위배가 될 수 있습니다. 그래서 본 논문에서는 __$p(z|y_1)$ 과 $p(z|y_2)$ 분포를 align 시키는 aligned auto-encoder를 사용합니다.__

#### 2.3.1. Aligned Auto Encoder
구체적으로 $p(z)$에 대한 posterior를 align 하는 대신에 __(1) $p_E(z|y_1)$ 와 $p_E(z|y_2)$를 서로 align 시키고__ 이를 위해서 __(2) adversarial Discriminator__ 를 사용합니다.

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020NLP/style-transfer/alignment_optimization.png" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt>수식 1. alignment를 위해 풀어야할 최적화 문제 </font></figcaption>
</center>

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020NLP/style-transfer/adversarial_loss.png" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt>수식 2. $p(z|y_1)$과 $p(z|y_2)$를 align 하기 위한 adversarial loss function  </font></figcaption>
</center>

여기서 실제 적용할 때에는 해당 constraint optimization 문제에 대해 Lagrangian relaxation을 적용해서 optimization을 적용하게 됩니다.
정리하면 Training 과정은 E, G, D 간의 min-max game이 됩니다.

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020NLP/style-transfer/minmax_game.png" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt>수식 3. $z$를 align 시키면서 style transfer를 하기위해 풀어야할 최적화 문제 </font></figcaption>
</center>

#### 2.3.2. Cross-Aligned Auto Encoder
Aligned Auto-Encoder에 이어 저희의 baseline architecture은 2가지 요소를 추가한 Crossed-Aligned Auto-Encoder를 사용합니다. 
1) 2개의 Discriminator ($x_1$과 transferred $x_2$를 구분하는 $D_1$과 $x_2$와 transferred $x_1$를 구분한는 $D_2$)를 사용합니다. Generative assumption에 의하면 $p(x_2|y_2) = \int_{x_1} p(x_2|x_1; y_1, y_2)p(x_1|y_1)dx_1$ 이므로, $x_2$(좌변에서 도출)는 transfer된 $x_1$(우변에서 도출)와 같은 분포를 가지고 있어야 합니다. 이는 $x_1$와 transfer된 $x_2$에도 마찬가지로 적용됩니다. 이를 이용하여 각각의 쌍을 구분하는 Discriminator를 2개 사용하였습니다. _이는 직관적으로 이해할 때, 각각의 style을 구분하면서 $z$를 align 시키는 aligned auto encoder와는 달리, 생성된 문장의 style transfer 여부를 판단하는 과정에서 $z$가 align 시키는 것으로 볼 수 있습니다._

<center>
<img src="https://diya-blogpost.s3.amazonaws.com/imgs_2020NLP/style-transfer/algorithm.png" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt>그림 1. cross-alignment를 적용한 알고리즘 </font></figcaption>
</center>


2) Usage of $\text{softmax}(v_t/\gamma)$ as an input to the generator RNN & Professor-Forcing<sup>[1](#prof_force)</sup>
G에 의해서 generate 된 discrete sample에 대해서 adversarial training을 적용하는 것은 gradient propagation을 방해합니다. 이에 대해서 저희 baseline은 2가지 테크닉을 적용하는데 하나는 generator RNN에 $\text{softmax}(v_t/\gamma)$를 적용하는 것이고, 다른 하나는 Professor-Forcing을 적용하는 것입니다.

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020NLP/style-transfer/professor_forcing.png" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt>그림 2. Professor Forcing </font></figcaption>
</center>

아키텍처의 각 파트별 구성은 다음과 같습니다.
Encoder와 Generator는 GRU cell을 적용한 single layer RNN 으로 구성합니다. Encoder에는 input: x, initial hidden state: y(이때 y는 x를 인풋으로 받는 FFN의 결과), output: last hidden state z가 변수로 들어가고, Generator는 encoder를 통해서 추출한 latent space $(y,z)$에 대한 x를 생성합니다. 다만 이 아키텍쳐에서는 x를 D에 넣지 않고, Generator 단의 hidden state을 $Z$을 output으로 잡고 $D_1$과 $D_2$에 넣어줍니다.

### 2.4. Setup
#### 2.4.1. Sentiment Modification
baseline에서는 sentiment를 하나의 style로 보아 negative to positive 그리고 positive to negative style transfer 작업을 수행했습니다. 이때 데이터로는 Yelp restaurant reviews를 사용하였고 3점 이상은 postive, 3점 미만은 negative로 구분하였습니다. 총 데이터는 negative sentence로는 250k 문장, positive sentence로는 350k 문장을 사용하였습니다.
해당 모델에 대한 evaluation으로 textCNN<sup>[2](#textcnn)</sup>을 사용한 classifier를 통한 quantitative evaluation, 두 명의 사람에 의한 랜덤한 테스트 문장 500개 에 대한 qulitative evaluatio이 있습니다. 이때 후자의 경우 1)문장의 유창성(Fluency)와 감정(Sentiment) 그리고 2) tranfer process에 대한 비교 분석이 평가항목으로 사용되었습니다.
본 논문에서는 실험의 baseline을 Hu et al.(2017) ControlGAN으로 두고 있습니다. 실험 결과상으로는 Hu et al.2017의 ControlGAN이 보다 높은 accuracy를 기록했으나, 논문에서는 예시에서처럼 본 논문의 아키텍쳐가 보다 consistent하고 overlapping한 문장을 생성한다고 주장하였습니다.

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020NLP/style-transfer/actual_example.png" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt> 그림 3. $z$를 alignment 시키면서 style transfer를 하기위해 풀어야할 최적화 문제 </font></figcaption>
</center>

#### 2.4.2. Word substitution decipherment
plain text에 있는 dicpher token을 1-to-1 substitution key로 바꾸는 작업이었습니다. 병렬 데이터가 없는 상황에서도 이것이 가능할까에 대한 실험(병렬 데이터가 있을 땐 사실 너무나 쉬운 태스크이다)으로 볼 수 있습니다.
이때 암호화 되어 있는 정도에 따라 당연히 문제의 난이도가 바뀌고 따라서 cipher된 token의 percentage에 대한 진행했다고 소개하고 있습니다. Cipher에 있어서는 parallel data가 제공된 경우에는 당연하게도 좋은 결과가 나왔고, paraellel 하지 않은 경우에 있어서는 분명히 text-style-transfer 아키텍쳐가 월등한 성능을 보였다.

## 3. <a name="implementation">Implementation</a>
### 3.1. Threshold Control Problem
논문에서 제시한대로 코드를 구현했을 때는, generator가 학습을 포기해버리는 듯한 현상이 있었습니다. 이때 보다 작은 $\lambda$ 값을 사용 했을 때, 그나마 독해가 가능한 문장이 생성이 되는 것으로 보아 학습초반 단계부터 adversary discriminator loss값이 지나치게 높아 Generator가 학습을 못하는 상황이 발생(min-max 게임에서 max가 폭주)하는 것으로 추정했습니다.<sup>[2*](#lost_generator)</sup> 연구진의 [텐서플로우 구현 코드](https://github.com/shentianxiao/language-style-transfer)에서는 discrimiator loss에 대한 threshold를 걸어 discriminator가 충분히 학습되었을 때에만 adversarial disc loss를 loss에 포함 시켰습니다.



### 3.2. Two-stage Control
baseline 에서는 없었지만, 아예 reconstruction을 target task로 먼저 학습시키고 이후에 adversarial discrimnation loss를 loss에 포함시켜 학습시키는 것을 시도해 보았습니다

### 3.2. Loss Function : Vanilla GAN / LSGAN / WGAN-GP
저희의 baseline에서는 Vanilla GAN만을 사용했습니다. 저희는 여기에 LSGAN<sup>[3](#lsgan)</sup>과 WGAN<sup>[4](#wgan)</sup>을 적용해보았습니다.

### 3.3. Bert Classifier
긍부정 여부를 style로 지정하여 실험을 할 때, evaluation을 위해 sentiment classifier가 필요했습니다. 이를 위해 pretrained 된 Bert (nsmc데이터에 대해서는 koBert) 모델을 이용한 classifier를 사용하였습니다.

### 3.4. FID Score
실험 초기에 로스가 떨어지는데도 제대로 된 문장이 생성이 안되는 경우가 있었습니다. 그래서 loss 값과 Bert Classifier를 이용한 accuracy 외에도 FID score를 구현하여 evaluation metric으로 사용하였습니다.

## 4. <a name="experiment">Experiment</a>
### 4.1 Setting
- Data : nsmc, yelp
- CPU : 8-core Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- GPU : 2 NVIDIA Titan Xp
- OS : Ubuntu 18.04, CUDA 11.2, Python 3.6.9
- Framework : PyTorch v1.6.0

### 4.1 Basic: Vanilla
논문 집필진이 사용한 파라미터는 다음과 같습니다.


|Hyper Param | Value |
| ---- | ---- |
| Gan type | Vanilla |
| Threshold | 24e-1 |
| Rho | 1 |
| Generator lr | 5e-4|
|Discriminator lr| 5e-5|

<figcaption><font size=2pt> * Rho는 논문에서의 Lambda 값입니다. </font></figcaption>

베이스라인과 동일한 파라미터로 실험했을 때의 결과는 다음과 같습니다.
Loss는 0.6334까지 떨어지고, 정확도는 0.883 까지 나오는 것을 확인할 수 있습니다. 그러나 논문에서 주장한 만큼 인간의 언어직관에 맞는 문장이 생성되지는 않았습니다.

| epoch | FED      | Loss   | Acc   |
|-------|----------|--------|-------|
| 1     | 172.934  | 0.9898 | 0.8   |
| 2     | 44.8421  | 0.6334 | 0.883 |
| 3     | 56.251   | 1.0074 | 0.597 |
| 4     | 176.2587 | 1.4328 | 0.745 |
| 5     | 101.9128 | 1.1587 | 0.788 |
| 6     | 103.028  | 1.47   | 0.703 |
| 7     | 171.9582 | 1.3578 | 0.707 |
| 8     | 61.4906  | 1.5265 | 0.732 |
| 9     | 61.9985  | 2.477  | 0.625 |
| 10    | 33.9438  | 1.0339 | 0.819 |
| 11    | 34.0183  | 1.488  | 0.692 |
| 12    | 32.4507  | 1.1989 | 0.796 |
| 13    | 31.7006  | 1.745  | 0.729 |
| 14    | 40.3296  | 1.4423 | 0.687 |
| 15    | 56.4975  | 1.8542 | 0.668 |
| 16    | 170.5676 | 1.5279 | 0.741 |
| 17    | 91.437   | 1.0977 | 0.787 |
| 18    | 33.9317  | 1.4018 | 0.694 |
| 19    | 62.7822  | 1.5722 | 0.766 |
| 20    | 47.7829  | 0.8072 | 0.768 |

<figcaption><font size=2pt> 표1. 논문에서 제시한 아키텍처를 그대로 따랐을 때의 결과. </font></figcaption>

논문에 언급되지 않은 Threshold 를 사용하지 않는 경우에 Acc는 0.65 부근에서 더 나아지지 않고, Loss 값 또한 3.14 정도에서 수렴해 버리는 것을 확인할 수 있었습니다. Threshold의 역할은 Generator를 학습할 때 discriminator가 충분히 학습되기 전까지는 이로부터 흘러나오는 로스(discriminator를 속이는 정도)를 사용하지 않도록 하는데 있습니다.
즉 discriminator의 로스를 컨트롤 하는 것이 학습에 가장 중요한 부분이라고 추정할 수 있습니다. 이로부터 1. rho 값을 조절하는 방법과 2.아예 two_stage로 discriminator를 조절하는 것을 시도해보게 되었습니다.

| 변환 이전 문장 | 변환 후 문장 |
| ----------- | --------- |
|**POS**|**NEG**|
|it was tremendous . | unfortunately this luck horrible just bad that service was bad for our service was great for my money here was great for our server.|
|just remember to tip well , its worth it ! | unfortunately this luck that bad service just bad experience that just go it gave it never gave them once more horrible.|
|**NEG**|**POS** |
|if you 're looking for a good karaoke bar , i recommend looking elsewhere .| if you do that make amazing work new!|
| i guess they thought i was stupid or something . | i liked it being that being being treated him for your money here again. |

<figcaption><font size=2pt> 표2. 논문에서 제시한 아키텍처를 그대로 따랐을 때의 실제 문장 예시 </font></figcaption>


예시 외에도 문장 transfer 결과 긍부정을 제외한 semantics가 유지되지 않았고, 특히 동일한 단어 밑 어구가 반복적으로 등장하는 현상이 보였습니다. 이에 대해 Mode collapse 를 의심해볼 여지가 있어 추가적인 epoch 을 돌려보았으나 결과가 나아지지 않았습니다.

| 변환 이전 문장 | 변환 후 문장 |
| ----------- | --------- |
|pancakes , french toast , eggs , bacon and sausage patties = mmmmmm good . | unfortunately this luck horrible that she made she made she made she made she made she made she made she made she made she said choice.|
|great place for shipping or a po box . | skip this one one one one one one one one my gave one bad experiences my favorite mexican food.|

<figcaption><font size=2pt> * Mode collapse가 의심되는 예. 실제 translation 결과를 보면 이런 종류의 결과가 압도적으로 많았습니다. </font></figcaption>

### 4.2 LsGan: with rho control
Vanilla gan 대신 LsGan을 사용할 때는 two-stage를 적용하기보다 rho 값을 직접적으로 조절해서 학습하는 것이 보다 나은 결과를 보여주었습니다. 

|Hyper Param | Value |
| ---- | ---- |
| Gan type | lsgan |
| Threshold | . |
| Rho | 1 |
| Generator lr | 5e-4|
|Discriminator lr| 5e-5|

<figcaption><font size=2pt> * Rho는 논문에서의 Lambda 값입니다. </font></figcaption>

lsgan을 이용할 때는 threshold를 적용하는 것이 의미가 없었습니다. discriminator 학습이 거의 1000 batch가 끝나기 전에(1 epoch 당 2762 batch) discrimator loss가 5e-2 부근에서 수렴해 버렸기 때문입니다.

실험결과는 다음과 같습니다.

| epoch | FED      | Loss   | Acc   |
|----|---------|--------|-------|
| 1  | 24.3813 | 3.2256 | 0.469 |
| 2  | 29.4781 | 2.7037 | 0.549 |
| 3  | 24.1408 | 2.5234 | 0.58  |
| 4  | 26.3172 | 2.4434 | 0.572 |
| 5  | 20.6971 | 2.2238 | 0.62  |
| 6  | 21.7314 | 2.7975 | 0.554 |
| 7  | 25.3921 | 3.1957 | 0.485 |
| 8  | 25.5921 | 3.0749 | 0.526 |
| 9  | 28.3941 | 2.3544 | 0.616 |
| 10 | 22.6054 | 2.5086 | 0.58  |
| 11 | 22.9302 | 2.7004 | 0.585 |
| 12 | 24.6255 | 1.6427 | 0.711 |
| 13 | 25.9261 | 2.6442 | 0.55  |
| 14 | 27.8733 | 2.2905 | 0.6   |
| 15 | 34.3953 | 1.9826 | 0.675 |
| 16 | 23.9833 | 2.6035 | 0.61  |
| 17 | 29.2181 | 2.6603 | 0.574 |
| 18 | 42.9414 | 3.3339 | 0.516 |
| 19 | 28.7167 | 2.331  | 0.511 |
| 20 | 23.5751 | 1.3548 | 0.759 |

<figcaption><font size=2pt> 표3. lsgan 을 사용했을 때의 학습 결과 </font></figcaption>

| 변환 이전 문장 | 변환 후 문장 |
| ----------- | --------- |
|**POS**|**NEG**|
|it was tremendous . | he've had better|
|just remember to tip well , its worth it ! | ok we do not worth just say, helpful? |
|**NEG**|**POS** |
|if you 're looking for a good karaoke bar , i recommend looking elsewhere . | we really enjoyed what it looks super cool, about it looks awesome.|
|i guess they thought i was stupid or something . | we enjoyed our drinks were several enchiladas, etc and check it out.|

<figcaption><font size=2pt> 표4. lsgan 을 사용했을 때의 문장 변환 예시</font></figcaption>

[Mode Collapse 해결 여부]
```
i 've always loved southwest airlines since i was a little girl . -> we really disappointed and they gave the seafood.
```
전반적으로 mode collapse가 의심되는 결과는 나타나지 않았고, 다만 문장의 correctness가 아쉬운 비직관적인 문장들이 많이 보였습니다.

### 4.3 Wgan: with two-staged learning

|Hyper Param | Value |
| --- | ---- |
| Gan type | wgan |
| Threshold | . |
| Rho | 1 |
| Generator lr | 5e-4|
|Discriminator lr| 5e-4|

<center>
<img src="https://diya-blogpost.s3.us-east-1.amazonaws.com/imgs_2020NLP/style-transfer/plot.png" style="width: 80%; height: 80%;"/>
<figcaption><font size=2pt> 그림 4. 정답은 two-stage learning 이었습니다. </font></figcaption>
</center>

여기서 학습은 완전히 2단계로 나누어서 진행하였습니다. Discriminator 로스를 제외한 generator 학습을 총 16epoch 진행한 후에 discriminator 에서 loss를 받는 단계를 12에폭으로 했을 때 가장 성능이 좋았던 것을 확인할 수 있었습니다.
위 그래프에서 Accuracy, Loss 값이 각각 0.866, 0.747 으로, vanilla gan을 사용했을 때와 거의 비슷하면서 FED 값이 15.8998 정도로 가장 좋았습니다. (~~드디어 찾았다.~~) 예상했던 대로, FED 값이 좋을 때 원했던 문장(스타일 위주로 변환된 문장)이 생성되었습니다.
또한 vanilla gan과 lsgan에서 모두 나타났던 mode collapse 문제를 해결한 것으로 나타났습니다.
다만 여기서 약간의 tweaking이 있었는데요, discriminator learning rate 을 5e-5 가 아닌 5e-4로 했을 때 학습이 보다 완만하게 진행되는 것을 확인할 수 있었습니다. 5e-5를 사용하는 경우에는 초반에 최고 성능이 나오고 이후에 mode collapse가 발생하였습니다.


| 변환 이전 문장 | 변환 후 문장 |
| ----------- | --------- |
|**POS**|**NEG**|
|it was tremendous . | it was gross. |
|just remember to tip well , its worth it ! | just so if you want so much better! |
|**NEG**|**POS** |
|if you 're looking for a good karaoke bar , i recommend looking elsewhere . | if they do a great job, i got great to our pedicure.|
|i guess they thought i was stupid or something . | i always loved they were all so i fantastic.|

<br>

[추가 예시]

| 변환 이전 문장 | 변환 후 문장 |
| ----------- | --------- |
|fun local bar . | avoid this location|
|free wifi , good selection of food , plenty of places to sit . | first time here for two hours, and this place is unprofessional and disappointed.|
|highly recommended for a romantic dinner . | then they had a huge disappointment. |
|great service . | bad service |
|never ever recommend ! | would definitely recommend!|
|service was good enough . | food was a terrible experience.|

<figcaption><font size=2pt> 표. sentence transfer 추가 예시. </font></figcaption>


### 4.4. 추가로 해보고 싶은 것들
Encoder 단의 RNN 네트워크의 $z$에 대한 intial state은 convention에 따라 zeros 로 시작합니다. 이에 대해서 Contextual RNN<sup>[5]</sup> 이라는 2019년 논문에서 개선점을 소개한 바가 있는데 이것을 적용했을 때 결과가 궁금하였습니다.
language-style-transfer는 2017년에 처음 선보인 후에 꾸준히 관련 연구가 진행되고 있는 분야입니다. DIYA에서 다음 NLP 주제를 진행할 때는 관련 최신 연구 동향을 살펴보고 적용 가능한 아키텍쳐(~~V100 쓸 때가 좋았다~~)를 찾아보는 것도 좋겠습니다.
또한 연산자원의 부족으로 인해서 `Z`를 represent 하는 hidden state을 더 늘려보지 못했습니다. 보다 복잡한 contents 정보를 align 시키기 위해서 보다 큰 hidden state을 사용해 보는 것도 유의미할 것 같다는 점을 끝으로 이번 포스트를 마무리 하겠습니다.

---

<sup><a name="richness_in_z"></a>1*</sup> 논문에 의하면 latent space z가 충분히 복잡해야 $x_1$, $x_2$의 joint distribution을 recover할 수 있다고 합니다. 이를 위해 VAE를 사용하는 대신에 aligned auto-encoder, 나아가 Professor-Forcing을 적용한 crossed aligned auto-encoder를 적용하고  있습니다.


<sup><a name="lost_generator"></a>2*</sup> 실제 논문의 구현코드를 보면 Encoder와 Generator의 learning rate 보다 discriminator의 learning rate이 훨씬 작은 것을 확인할 수 있었습니다. 아마도 이 논문을 작성하는 과정에서도 generator가 학습을 포기하는 상황이 발생했을 것으로 추정(?)할 수 있었습니다.

## References
#### <sup><a name="lan_style_trans"></a>1</sup><sub><a href="https://papers.nips.cc/paper/2017/file/2d2c8394e31101a261abf1784302bf75-Paper.pdf" target="_blank">Style Transfer from Non-Parallel Text by Cross-Alignment, Tianxiao Shen et al, NIPS 2017</a></sub>
#### <sup><a name="textcnn"></a>2</sup><sub><a href="https://arxiv.org/pdf/1408.5882.pdf" target="_blank">Convolutional Neural Networks for Sentence Classification, Kim Yoon, arXiv:1408.5882, 2014
#### <sup><a name="lsgan"></a>3</sup><sub><a href="https://arxiv.org/pdf/1611.04076.pdf" target="_blank"> Least Squares Generative Adversarial Networks, Mao et al, arXiv:1611.04076, 2016</a><sub>
#### <sup><a name="wgan"></a>4</sup><sub><a href="https://arxiv.org/pdf/1704.00028.pdf" target="_blank"> Improved Training of Wasserstein GANs, Gulrajani et al , arXiv:1704.00028, 2017</a><sub>
#### <sup><a name="contextual_rnn"></a>5</sup><sub><a href="https://arxiv.org/pdf/1902.03455.pdf#:~:text=The%20performance%20on%20an%20associative,information%20from%20the%20input%20sequence.&text=The%20initialization%20method%20of%20the,most%20commonly%20equal%20to%20zero" target="_blank">Contextual Recurrent Neural Networks, Wenke et al, arXiv:1902.03455, 2019</a></sub>



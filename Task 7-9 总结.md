# Task 7-9 总结

# 批量归一化（BatchNormalization）
#### 对输入的标准化（浅层模型）
处理后的任意一个特征在数据集中所有样本上的均值为0、标准差为1。  
标准化处理输入数据使各个特征的分布相近
#### 批量归一化（深度模型）
利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。

### 1.对全连接层做批量归一化
位置：全连接层中的仿射变换和激活函数之间。  
**全连接：**  
$$
\boldsymbol{x} = \boldsymbol{W\boldsymbol{u} + \boldsymbol{b}} \\
 output =\phi(\boldsymbol{x})
$$


**批量归一化：**
$$
output=\phi(\text{BN}(\boldsymbol{x}))$$

$$
\boldsymbol{y}^{(i)} = \text{BN}(\boldsymbol{x}^{(i)})
$$

$$
\boldsymbol{\mu}_\mathcal{B} \leftarrow \frac{1}{m}\sum_{i = 1}^{m} \boldsymbol{x}^{(i)},
$$

$$
\boldsymbol{\sigma}_\mathcal{B}^2 \leftarrow \frac{1}{m} \sum_{i=1}^{m}(\boldsymbol{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B})^2,
$$

$$
\hat{\boldsymbol{x}}^{(i)} \leftarrow \frac{\boldsymbol{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B}}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}},
$$
这⾥ϵ > 0是个很小的常数，保证分母大于0

$$
{\boldsymbol{y}}^{(i)} \leftarrow \boldsymbol{\gamma} \odot
\hat{\boldsymbol{x}}^{(i)} + \boldsymbol{\beta}.
$$


引入可学习参数：拉伸参数γ和偏移参数β。若$\boldsymbol{\gamma} = \sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}$和$\boldsymbol{\beta} = \boldsymbol{\mu}_\mathcal{B}$，批量归一化无效。

### 2.对卷积层做批量归⼀化
位置：卷积计算之后、应⽤激活函数之前。  
如果卷积计算输出多个通道，我们需要对这些通道的输出分别做批量归一化，且每个通道都拥有独立的拉伸和偏移参数。
计算：对单通道，batchsize=m,卷积计算输出=pxq
对该通道中m×p×q个元素同时做批量归一化,使用相同的均值和方差。

### 3.预测时的批量归⼀化
训练：以batch为单位,对每个batch计算均值和方差。  
预测：用移动平均估算整个训练数据集的样本均值和方差。

# 残差网络（ResNet）
深度学习的问题：深度CNN网络达到一定深度后再一味地增加层数并不能带来进一步地分类性能提高，反而会招致网络收敛变得更慢，准确率也变得更差。
### 残差块（Residual Block）
恒等映射：  
左边：f(x)=x                                                  
右边：f(x)-x=0 （易于捕捉恒等映射的细微波动）

![Image Name](https://cdn.kesci.com/upload/image/q5l8lhnot4.png?imageView2/0/w/600/h/600)

在残差块中，输⼊可通过跨层的数据线路更快 地向前传播。

### ResNet模型
卷积(64,7x7,3)  
批量一体化  
最大池化(3x3,2)  

残差块x4 (通过步幅为2的残差块在每个模块之间减小高和宽)

全局平均池化

全连接

# 稠密连接网络（DenseNet）

![Image Name](https://cdn.kesci.com/upload/image/q5l8mi78yz.png?imageView2/0/w/600/h/600)

###主要构建模块：  
稠密块（dense block）： 定义了输入和输出是如何连结的。  
过渡层（transition layer）：用来控制通道数，使之不过大。

# 词嵌入基础

我们在[“循环神经网络的从零开始实现”](https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html)一节中使用 one-hot 向量表示单词，虽然它们构造起来很容易，但通常并不是一个好选择。一个主要的原因是，one-hot 词向量无法准确表达不同词之间的相似度，如我们常常使用的余弦相似度。

Word2Vec 词嵌入工具的提出正是为了解决上面这个问题，它将每个词表示成一个定长的向量，并通过在语料库上的预训练使得这些向量能较好地表达不同词之间的相似和类比关系，以引入一定的语义信息。基于两种概率模型的假设，我们可以定义两种 Word2Vec 模型：
1. [Skip-Gram 跳字模型](https://zh.d2l.ai/chapter_natural-language-processing/word2vec.html#%E8%B7%B3%E5%AD%97%E6%A8%A1%E5%9E%8B)：假设背景词由中心词生成，即建模 $P(w_o\mid w_c)$，其中 $w_c$ 为中心词，$w_o$ 为任一背景词；

![Image Name](https://cdn.kesci.com/upload/image/q5mjsq84o9.png?imageView2/0/w/960/h/960)

2. [CBOW (continuous bag-of-words) 连续词袋模型](https://zh.d2l.ai/chapter_natural-language-processing/word2vec.html#%E8%BF%9E%E7%BB%AD%E8%AF%8D%E8%A2%8B%E6%A8%A1%E5%9E%8B)：假设中心词由背景词生成，即建模 $P(w_c\mid \mathcal{W}_o)$，其中 $\mathcal{W}_o$ 为背景词的集合。

![Image Name](https://cdn.kesci.com/upload/image/q5mjt4r02n.png?imageView2/0/w/960/h/960)

在这里我们主要介绍 Skip-Gram 模型的实现，CBOW 实现与其类似，读者可之后自己尝试实现。后续的内容将大致从以下四个部分展开：

1. PTB 数据集
2. Skip-Gram 跳字模型
3. 负采样近似
4. 训练模型

## PTB 数据集

简单来说，Word2Vec 能从语料中学到如何将离散的词映射为连续空间中的向量，并保留其语义上的相似关系。那么为了训练 Word2Vec 模型，我们就需要一个自然语言语料库，模型将从中学习各个单词间的关系，这里我们使用经典的 PTB 语料库进行训练。[PTB (Penn Tree Bank)](https://catalog.ldc.upenn.edu/LDC99T42) 是一个常用的小型语料库，它采样自《华尔街日报》的文章，包括训练集、验证集和测试集。我们将在PTB训练集上训练词嵌入模型。



### 二次采样

文本数据中一般会出现一些高频词，如英文中的“the”“a”和“in”。通常来说，在一个背景窗口中，一个词（如“chip”）和较低频词（如“microprocessor”）同时出现比和较高频词（如“the”）同时出现对训练词嵌入模型更有益。因此，训练词嵌入模型时可以对词进行二次采样。 具体来说，数据集中每个被索引词 $w_i$ 将有一定概率被丢弃，该丢弃概率为


$$
P(w_i)=\max(1-\sqrt{\frac{t}{f(w_i)}},0)
$$


其中  $f(w_i)$  是数据集中词 $w_i$ 的个数与总词数之比，常数 $t$ 是一个超参数（实验中设为 $10^{−4}$）。可见，只有当 $f(w_i)>t$ 时，我们才有可能在二次采样中丢弃词 $w_i$，并且越高频的词被丢弃的概率越大。具体的代码如下：

## Skip-Gram 跳字模型

在跳字模型中，每个词被表示成两个 $d$ 维向量，用来计算条件概率。假设这个词在词典中索引为 $i$ ，当它为中心词时向量表示为 $\boldsymbol{v}_i\in\mathbb{R}^d$，而为背景词时向量表示为 $\boldsymbol{u}_i\in\mathbb{R}^d$ 。设中心词 $w_c$ 在词典中索引为 $c$，背景词 $w_o$ 在词典中索引为 $o$，我们假设给定中心词生成背景词的条件概率满足下式：


$$
P(w_o\mid w_c)=\frac{\exp(\boldsymbol{u}_o^\top \boldsymbol{v}_c)}{\sum_{i\in\mathcal{V}}\exp(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}
$$

## 负采样近似

由于 softmax 运算考虑了背景词可能是词典 $\mathcal{V}$ 中的任一词，对于含几十万或上百万词的较大词典，就可能导致计算的开销过大。我们将以 skip-gram 模型为例，介绍负采样 (negative sampling) 的实现来尝试解决这个问题。

负采样方法用以下公式来近似条件概率 $P(w_o\mid w_c)=\frac{\exp(\boldsymbol{u}_o^\top \boldsymbol{v}_c)}{\sum_{i\in\mathcal{V}}\exp(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}$：


$$
P(w_o\mid w_c)=P(D=1\mid w_c,w_o)\prod_{k=1,w_k\sim P(w)}^K P(D=0\mid w_c,w_k)
$$


其中 $P(D=1\mid w_c,w_o)=\sigma(\boldsymbol{u}_o^\top\boldsymbol{v}_c)$，$\sigma(\cdot)$ 为 sigmoid 函数。对于一对中心词和背景词，我们从词典中随机采样 $K$ 个噪声词（实验中设 $K=5$）。根据 Word2Vec 论文的建议，噪声词采样概率 $P(w)$ 设为 $w$ 词频与总词频之比的 $0.75$ 次方。

## 训练模型

### 损失函数

应用负采样方法后，我们可利用最大似然估计的对数等价形式将损失函数定义为如下


$$
\sum_{t=1}^T\sum_{-m\le j\le m,j\ne 0} [-\log P(D=1\mid w^{(t)},w^{(t+j)})-\sum_{k=1,w_k\sim P(w)^K}\log P(D=0\mid w^{(t)},w_k)]
$$


根据这个损失函数的定义，我们可以直接使用二元交叉熵损失函数进行计算：

## 参考
* [Dive into Deep Learning](https://d2l.ai/chapter_natural-language-processing/word2vec.html). Ch14.1-14.4.
* [动手学深度学习](http://zh.gluon.ai/chapter_natural-language-processing/word2vec.html). Ch10.1-10.3.
* [Dive-into-DL-PyTorch on GitHub](https://github.com/ShusenTang/Dive-into-DL-PyTorch/blob/master/code/chapter10_natural-language-processing/10.3_word2vec-pytorch.ipynb)

# 词嵌入进阶

在[“Word2Vec的实现”]()一节中，我们在小规模数据集上训练了一个 Word2Vec 词嵌入模型，并通过词向量的余弦相似度搜索近义词。虽然 Word2Vec 已经能够成功地将离散的单词转换为连续的词向量，并能一定程度上地保存词与词之间的近似关系，但 Word2Vec 模型仍不是完美的，它还可以被进一步地改进：

1. 子词嵌入（subword embedding）：[FastText](https://zh.d2l.ai/chapter_natural-language-processing/fasttext.html) 以固定大小的 n-gram 形式将单词更细致地表示为了子词的集合，而 [BPE (byte pair encoding)](https://d2l.ai/chapter_natural-language-processing/subword-embedding.html#byte-pair-encoding) 算法则能根据语料库的统计信息，自动且动态地生成高频子词的集合；
2. [GloVe 全局向量的词嵌入](https://zh.d2l.ai/chapter_natural-language-processing/glove.html): 通过等价转换 Word2Vec 模型的条件概率公式，我们可以得到一个全局的损失函数表达，并在此基础上进一步优化模型。

实际中，我们常常在大规模的语料上训练这些词嵌入模型，并将预训练得到的词向量应用到下游的自然语言处理任务中。本节就将以 GloVe 模型为例，演示如何用预训练好的词向量来求近义词和类比词。

## GloVe 全局向量的词嵌入

### GloVe 模型

先简单回顾以下 Word2Vec 的损失函数（以 Skip-Gram 模型为例，不考虑负采样近似）：


$$
-\sum_{t=1}^T\sum_{-m\le j\le m,j\ne 0} \log P(w^{(t+j)}\mid w^{(t)})
$$


其中


$$
P(w_j\mid w_i) = \frac{\exp(\boldsymbol{u}_j^\top\boldsymbol{v}_i)}{\sum_{k\in\mathcal{V}}\exp(\boldsymbol{u}_k^\top\boldsymbol{v}_i)}
$$


是 $w_i$ 为中心词，$w_j$ 为背景词时 Skip-Gram 模型所假设的条件概率计算公式，我们将其简写为 $q_{ij}$。

注意到此时我们的损失函数中包含两个求和符号，它们分别枚举了语料库中的每个中心词和其对应的每个背景词。实际上我们还可以采用另一种计数方式，那就是直接枚举每个词分别作为中心词和背景词的情况：


$$
-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij}\log q_{ij}
$$


其中 $x_{ij}$ 表示整个数据集中 $w_j$ 作为 $w_i$ 的背景词的次数总和。

我们还可以将该式进一步地改写为交叉熵 (cross-entropy) 的形式如下：


$$
-\sum_{i\in\mathcal{V}}x_i\sum_{j\in\mathcal{V}}p_{ij} \log q_{ij}
$$


其中 $x_i$ 是 $w_i$ 的背景词窗大小总和，$p_{ij}=x_{ij}/x_i$ 是 $w_j$ 在 $w_i$ 的背景词窗中所占的比例。

从这里可以看出，我们的词嵌入方法实际上就是想让模型学出 $w_j$ 有多大概率是 $w_i$ 的背景词，而真实的标签则是语料库上的统计数据。同时，语料库中的每个词根据 $x_i$ 的不同，在损失函数中所占的比重也不同。

注意到目前为止，我们只是改写了 Skip-Gram 模型损失函数的表面形式，还没有对模型做任何实质上的改动。而在 Word2Vec 之后提出的 GloVe 模型，则是在之前的基础上做出了以下几点改动：

1. 使用非概率分布的变量 $p'_{ij}=x_{ij}$ 和 $q′_{ij}=\exp(\boldsymbol{u}^\top_j\boldsymbol{v}_i)$，并对它们取对数；
2. 为每个词 $w_i$ 增加两个标量模型参数：中心词偏差项 $b_i$ 和背景词偏差项 $c_i$，松弛了概率定义中的规范性；
3. 将每个损失项的权重 $x_i$ 替换成函数 $h(x_{ij})$，权重函数 $h(x)$ 是值域在 $[0,1]$ 上的单调递增函数，松弛了中心词重要性与 $x_i$ 线性相关的隐含假设；
4. 用平方损失函数替代了交叉熵损失函数。

综上，我们获得了 GloVe 模型的损失函数表达式：


$$
\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} h(x_{ij}) (\boldsymbol{u}^\top_j\boldsymbol{v}_i+b_i+c_j-\log x_{ij})^2
$$


由于这些非零 $x_{ij}$ 是预先基于整个数据集计算得到的，包含了数据集的全局统计信息，因此 GloVe 模型的命名取“全局向量”（Global Vectors）之意。

### 载入预训练的 GloVe 向量

[GloVe 官方](https://nlp.stanford.edu/projects/glove/) 提供了多种规格的预训练词向量，语料库分别采用了维基百科、CommonCrawl和推特等，语料库中词语总数也涵盖了从60亿到8,400亿的不同规模，同时还提供了多种词向量维度供下游模型使用。

[`torchtext.vocab`](https://torchtext.readthedocs.io/en/latest/vocab.html) 中已经支持了 GloVe, FastText, CharNGram 等常用的预训练词向量，我们可以通过声明 [`torchtext.vocab.GloVe`](https://torchtext.readthedocs.io/en/latest/vocab.html#glove) 类的实例来加载预训练好的 GloVe 词向量。

## 求近义词和类比词

### 求近义词

由于词向量空间中的余弦相似性可以衡量词语含义的相似性（为什么？），我们可以通过寻找空间中的 k 近邻，来查询单词的近义词。

### 求类比词

除了求近义词以外，我们还可以使用预训练词向量求词与词之间的类比关系，例如“man”之于“woman”相当于“son”之于“daughter”。求类比词问题可以定义为：对于类比关系中的4个词“$a$ 之于 $b$ 相当于 $c$ 之于 $d$”，给定前3个词 $a,b,c$ 求 $d$。求类比词的思路是，搜索与 $\text{vec}(c)+\text{vec}(b)−\text{vec}(a)$ 的结果向量最相似的词向量，其中 $\text{vec}(w)$ 为 $w$ 的词向量。

# 文本情感分类

文本分类是自然语言处理的一个常见任务，它把一段不定长的文本序列变换为文本的类别。本节关注它的一个子问题：使用文本情感分类来分析文本作者的情绪。这个问题也叫情感分析，并有着广泛的应用。

同搜索近义词和类比词一样，文本分类也属于词嵌入的下游应用。在本节中，我们将应用预训练的词向量和含多个隐藏层的双向循环神经网络与卷积神经网络，来判断一段不定长的文本序列中包含的是正面还是负面的情绪。后续内容将从以下几个方面展开：

1. 文本情感分类数据集
2. 使用循环神经网络进行情感分类
3. 使用卷积神经网络进行情感分类

### 预处理数据

读取数据后，我们先根据文本的格式进行单词的切分，再利用 [`torchtext.vocab.Vocab`](https://torchtext.readthedocs.io/en/latest/vocab.html#vocab) 创建词典。

词典和词语的索引创建好后，就可以将数据集的文本从字符串的形式转换为单词下标序列的形式，以待之后的使用。

### 创建数据迭代器

利用 [`torch.utils.data.TensorDataset`](https://pytorch.org/docs/stable/data.html?highlight=tensor%20dataset#torch.utils.data.TensorDataset)，可以创建 PyTorch 格式的数据集，从而创建数据迭代器。

## 使用循环神经网络

### 双向循环神经网络

在[“双向循环神经网络”](https://zh.d2l.ai/chapter_recurrent-neural-networks/bi-rnn.html)一节中，我们介绍了其模型与前向计算的公式，这里简单回顾一下：

![Image Name](https://cdn.kesci.com/upload/image/q5mnobct47.png?imageView2/0/w/960/h/960)


![Image Name](https://cdn.kesci.com/upload/image/q5mo6okdnp.png?imageView2/0/w/960/h/960)


给定输入序列 $\{\boldsymbol{X}_1,\boldsymbol{X}_2,\dots,\boldsymbol{X}_T\}$，其中 $\boldsymbol{X}_t\in\mathbb{R}^{n\times d}$ 为时间步（批量大小为 $n$，输入维度为 $d$）。在双向循环神经网络的架构中，设时间步 $t$ 上的正向隐藏状态为 $\overrightarrow{\boldsymbol{H}}_{t} \in \mathbb{R}^{n \times h}$ （正向隐藏状态维度为 $h$），反向隐藏状态为 $\overleftarrow{\boldsymbol{H}}_{t} \in \mathbb{R}^{n \times h}$ （反向隐藏状态维度为 $h$）。我们可以分别计算正向隐藏状态和反向隐藏状态：


$$
\begin{aligned}
&\overrightarrow{\boldsymbol{H}}_{t}=\phi\left(\boldsymbol{X}_{t} \boldsymbol{W}_{x h}^{(f)}+\overrightarrow{\boldsymbol{H}}_{t-1} \boldsymbol{W}_{h h}^{(f)}+\boldsymbol{b}_{h}^{(f)}\right)\\
&\overleftarrow{\boldsymbol{H}}_{t}=\phi\left(\boldsymbol{X}_{t} \boldsymbol{W}_{x h}^{(b)}+\overleftarrow{\boldsymbol{H}}_{t+1} \boldsymbol{W}_{h h}^{(b)}+\boldsymbol{b}_{h}^{(b)}\right)
\end{aligned}
$$


其中权重 $\boldsymbol{W}_{x h}^{(f)} \in \mathbb{R}^{d \times h}, \boldsymbol{W}_{h h}^{(f)} \in \mathbb{R}^{h \times h}, \boldsymbol{W}_{x h}^{(b)} \in \mathbb{R}^{d \times h}, \boldsymbol{W}_{h h}^{(b)} \in \mathbb{R}^{h \times h}$ 和偏差 $\boldsymbol{b}_{h}^{(f)} \in \mathbb{R}^{1 \times h}, \boldsymbol{b}_{h}^{(b)} \in \mathbb{R}^{1 \times h}$ 均为模型参数，$\phi$ 为隐藏层激活函数。

然后我们连结两个方向的隐藏状态 $\overrightarrow{\boldsymbol{H}}_{t}$ 和 $\overleftarrow{\boldsymbol{H}}_{t}$ 来得到隐藏状态 $\boldsymbol{H}_{t} \in \mathbb{R}^{n \times 2 h}$，并将其输入到输出层。输出层计算输出 $\boldsymbol{O}_{t} \in \mathbb{R}^{n \times q}$（输出维度为 $q$）：


$$
\boldsymbol{O}_{t}=\boldsymbol{H}_{t} \boldsymbol{W}_{h q}+\boldsymbol{b}_{q}
$$


其中权重 $\boldsymbol{W}_{h q} \in \mathbb{R}^{2 h \times q}$ 和偏差 $\boldsymbol{b}_{q} \in \mathbb{R}^{1 \times q}$ 为输出层的模型参数。不同方向上的隐藏单元维度也可以不同。

利用 [`torch.nn.RNN`](https://pytorch.org/docs/stable/nn.html?highlight=rnn#torch.nn.RNN) 或 [`torch.nn.LSTM`](https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM) 模组，我们可以很方便地实现双向循环神经网络，下面是以 LSTM 为例的代码。

### 加载预训练的词向量

由于预训练词向量的词典及词语索引与我们使用的数据集并不相同，所以需要根据目前的词典及索引的顺序来加载预训练词向量。

### 训练模型

训练时可以调用之前编写的 `train` 及 `evaluate_accuracy` 函数。

由于嵌入层的参数是不需要在训练过程中被更新的，所以我们利用 `filter` 函数和 `lambda` 表达式来过滤掉模型中不需要更新参数的部分。

## 使用卷积神经网络

### 一维卷积层

在介绍模型前我们先来解释一维卷积层的工作原理。与二维卷积层一样，一维卷积层使用一维的互相关运算。在一维互相关运算中，卷积窗口从输入数组的最左方开始，按从左往右的顺序，依次在输入数组上滑动。当卷积窗口滑动到某一位置时，窗口中的输入子数组与核数组按元素相乘并求和，得到输出数组中相应位置的元素。如图所示，输入是一个宽为 7 的一维数组，核数组的宽为 2。可以看到输出的宽度为 7−2+1=6，且第一个元素是由输入的最左边的宽为 2 的子数组与核数组按元素相乘后再相加得到的：0×1+1×2=2。

![Image Name](https://cdn.kesci.com/upload/image/q5mo8qs7dc.png?imageView2/0/w/960/h/960)

多输入通道的一维互相关运算也与多输入通道的二维互相关运算类似：在每个通道上，将核与相应的输入做一维互相关运算，并将通道之间的结果相加得到输出结果。下图展示了含 3 个输入通道的一维互相关运算，其中阴影部分为第一个输出元素及其计算所使用的输入和核数组元素：0×1+1×2+1×3+2×4+2×(−1)+3×(−3)=2。

![Image Name](https://cdn.kesci.com/upload/image/q5moaawczv.png?imageView2/0/w/960/h/960)

由二维互相关运算的定义可知，多输入通道的一维互相关运算可以看作单输入通道的二维互相关运算。如图所示，我们也可以将图中多输入通道的一维互相关运算以等价的单输入通道的二维互相关运算呈现。这里核的高等于输入的高。图中的阴影部分为第一个输出元素及其计算所使用的输入和核数组元素：2×(−1)+3×(−3)+1×3+2×4+0×1+1×2=2。

![Image Name](https://cdn.kesci.com/upload/image/q5moav6ue3.png?imageView2/0/w/960/h/960)

*注：反之仅当二维卷积核的高度等于输入的高度时才成立。*

之前的例子中输出都只有一个通道。我们在[“多输入通道和多输出通道”](https://zh.d2l.ai/chapter_convolutional-neural-networks/channels.html)一节中介绍了如何在二维卷积层中指定多个输出通道。类似地，我们也可以在一维卷积层指定多个输出通道，从而拓展卷积层中的模型参数。

### 时序最大池化层

类似地，我们有一维池化层。TextCNN 中使用的时序最大池化（max-over-time pooling）层实际上对应一维全局最大池化层：假设输入包含多个通道，各通道由不同时间步上的数值组成，各通道的输出即该通道所有时间步中最大的数值。因此，时序最大池化层的输入在各个通道上的时间步数可以不同。

![Image Name](https://cdn.kesci.com/upload/image/q5mobv3kol.png?imageView2/0/w/960/h/960)

*注：自然语言中还有一些其他的池化操作，可参考这篇[博文](https://blog.csdn.net/malefactor/article/details/51078135)。*

为提升计算性能，我们常常将不同长度的时序样本组成一个小批量，并通过在较短序列后附加特殊字符（如0）令批量中各时序样本长度相同。这些人为添加的特殊字符当然是无意义的。由于时序最大池化的主要目的是抓取时序中最重要的特征，它通常能使模型不受人为添加字符的影响。

### TextCNN 模型

TextCNN 模型主要使用了一维卷积层和时序最大池化层。假设输入的文本序列由 $n$ 个词组成，每个词用 $d$ 维的词向量表示。那么输入样本的宽为 $n$，输入通道数为 $d$。TextCNN 的计算主要分为以下几步。

1. 定义多个一维卷积核，并使用这些卷积核对输入分别做卷积计算。宽度不同的卷积核可能会捕捉到不同个数的相邻词的相关性。
2. 对输出的所有通道分别做时序最大池化，再将这些通道的池化输出值连结为向量。
3. 通过全连接层将连结后的向量变换为有关各类别的输出。这一步可以使用丢弃层应对过拟合。

下图用一个例子解释了 TextCNN 的设计。这里的输入是一个有 11 个词的句子，每个词用 6 维词向量表示。因此输入序列的宽为 11，输入通道数为 6。给定 2 个一维卷积核，核宽分别为 2 和 4，输出通道数分别设为 4 和 5。因此，一维卷积计算后，4 个输出通道的宽为 11−2+1=10，而其他 5 个通道的宽为 11−4+1=8。尽管每个通道的宽不同，我们依然可以对各个通道做时序最大池化，并将 9 个通道的池化输出连结成一个 9 维向量。最终，使用全连接将 9 维向量变换为 2 维输出，即正面情感和负面情感的预测。

![Image Name](https://cdn.kesci.com/upload/image/q5modgownf.png?imageView2/0/w/960/h/960)

下面我们来实现 TextCNN 模型。与上一节相比，除了用一维卷积层替换循环神经网络外，这里我们还使用了两个嵌入层，一个的权重固定，另一个则参与训练。
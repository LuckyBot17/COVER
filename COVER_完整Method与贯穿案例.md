# COVER：基于负控制与过度识别的通用关系幻觉核验与修复

> **Working title**: *Beyond Consistency: Overidentified Relation Verification for Hallucination-Resistant LVLMs*  
> **Working name**: COVER (*Causal Overidentification for Verifying Entity Relations*)  
> **Document status**: 研究方案与 Method 完整草案，尚未经完整实验验证  
> **Relationship to ECD**: ECD 是 COVER 在几何空间关系上的一个解析特例  
> **Important**: 本文贯穿案例中的数值均为演示数值，不是已得到的实验结果。

---

## 1. 方法摘要

COVER 是一个作用于冻结 LVLM 的生成后关系核验与修复框架。给定图像和 LVLM 已生成的回答，COVER 先将其拆解为 subject–relation–object 三元组，再针对每个关系声明构造多个具有明确语义约束的观察视图。

COVER 不仅询问“模型在不同视图下是否给出一致答案”，而是进一步询问：

> 是否存在一个共同的视觉关系事实，能同时解释模型在所有合法干预下的响应？

为此，COVER 为每个正常视图构造匹配的视觉负控制，例如遮挡 subject、object 或两者。原图与负控制图之间的关系支持差被视为对“实体条件化视觉证据”的近似测量。多个测量随后被对齐到统一关系语义下，用于估计：

1. **Common visual support**: 多个视图共同提供了多少关系视觉证据；
2. **Overidentification conflict**: 这些视图是否真的在测量同一个潜在关系事实。

COVER 最终不做必然的二元选择，而是对生成关系执行 `ACCEPT`、`REPAIR` 或 `ABSTAIN`。

---

## 2. 问题定义

### 2.1 从开放生成到关系声明

设冻结的视觉语言大模型为

$$
p_\theta(y\mid I,q),
$$

其中 $I$ 是输入图像，$q$ 是用户问题或 caption prompt，$y$ 是 LVLM 生成的文本。COVER 不修改 $\theta$。

关系抽取器 $E$ 将生成文本映射为三元组集合：

$$
E(y)=\mathcal T_y=\{c_i\}_{i=1}^{N_y},
\qquad
c_i=(s_i,r_i,o_i),
$$

其中：

- $s_i$ 是 subject；
- $o_i$ 是 object；
- $r_i\in\mathcal R$ 是归一化后的二元关系；
- $\mathcal R$ 是支持的关系词表或知识图谱关系集。

例如，文本

> A man is riding a horse in a field.

被归一化为

$$
c=(\text{man},\text{riding},\text{horse}).
$$

在开放生成中，关系抽取器还需返回 $c_i$ 在 $y$ 中的字符或 token span，以便 COVER 在最终阶段替换或删除不可靠的关系短语。

### 2.2 关系幻觉

记图像中关系声明的真值为

$$
Y(I,c)\in\{0,1\}.
$$

当 LVLM 生成了 $c$ 但 $Y(I,c)=0$ 时，$c$ 构成关系幻觉。COVER 的目标不是仅仅检测 $Y(I,c)$，而是构造一个可选择决策器：

$$
D(I,c)\in
\{\texttt{ACCEPT},\texttt{REPAIR}(r'),\texttt{ABSTAIN}\}.
$$

### 2.3 单视图下的不可识别性

对一个关系声明 $c$，原始 LVLM 分数可作如下分析性分解：

$$
a_\theta(I,c)
=u_\theta(c,q)
+v_\theta(I,c,q)
+\epsilon_\theta(I,c,q),
\tag{1}
$$

其中：

- $u_\theta$ 表示与当前关键视觉证据近似无关的关系词、模板和语言共现先验；
- $v_\theta$ 表示由当前图像支持的实体关系证据；
- $\epsilon_\theta$ 表示无法完全建模的误差。

该分解不要求 LVLM 内部显式包含三个模块，而是 COVER 的分析假设。单次输出只能观测 $a_\theta$，因此无法判断一个高分关系是来自 $u_\theta$ 还是 $v_\theta$。这是 COVER 所要解决的基本不可识别问题。

---

## 3. 关系结构签名与统一算子

### 3.1 关系结构签名

每个关系 $r$ 不对应一套独立算法，只需声明一个结构签名：

$$
\mathcal S(r)
=\big(r^{-1},\chi_{\mathrm{sym}}(r),\mathcal G_r,\mathcal P_r\big),
\tag{2}
$$

其中：

- $r^{-1}$ 是论元交换后的逆关系；
- $\chi_{\mathrm{sym}}(r)\in\{0,1\}$ 表示该关系是否对称；
- $\mathcal G_r$ 是对该关系具有确定语义后果的图像变换集；
- $\mathcal P_r$ 是不改变关系真值的共享文本模板集。

典型签名如下。

| relation $r$ | inverse $r^{-1}$ | symmetric | optional image action |
|---|---|---:|---|
| left of | right of | 0 | hflip: left $\leftrightarrow$ right |
| above | below | 0 | vflip: above $\leftrightarrow$ below，需可靠性审计 |
| in front of | behind | 0 | 自然 2D 图中不强制几何变换 |
| riding | ridden by | 0 | none required |
| wearing | worn by | 0 | none required |
| inside | contains | 0 | none required |
| next to | next to | 1 | identity under argument swap |
| touching | touching | 1 | identity under argument swap |

这些签名可以从 R-Bench、Reefknot、MMRel 和 Tri-HE 已有关系词表中构建。签名表是 ontology metadata，而不是为每个关系训练的独立模型。

### 3.2 统一正控制算子

定义正控制算子 $e$ 为

$$
e=(T_e,\pi_e,\sigma_e,\varphi_e),
\tag{3}
$$

其中：

- $T_e$ 作用于图像；
- $\pi_e$ 作用于 subject/object 顺序；
- $\sigma_e$ 作用于关系语义；
- $\varphi_e$ 作用于文本表达模板。

经变换的声明为

$$
c_e
=\varphi_e\!\left(
\pi_e(s),\sigma_e(r),\pi_e(o)
\right),
\qquad
I_e=T_e(I).
\tag{4}
$$

合法正控制应满足真值保持：

$$
Y(I,c)=Y(I_e,c_e).
\tag{5}
$$

COVER 的通用正控制包括：

1. **Identity**: 原图与原始关系声明；
2. **Argument inversion**: $(s,r,o)\mapsto(o,r^{-1},s)$；
3. **Semantic paraphrase**: 不改变关系真值的句法或词汇改写；
4. **Pair crop**: 保留 subject/object 以及必要局部上下文的实体对裁剪；
5. **Typed equivariance**: 仅在 $\mathcal G_r$ 明确规定时使用的几何等变操作。

前四类构成跨关系族的核心算子；第五类只是有确定语义律时的附加测量。

### 3.3 候选关系集

对生成关系 $r_{\mathrm{gen}}$，先根据 subject/object 语义类型过滤出在当前论元顺序下可表达的关系集 $\mathcal R(s,o)$，再用原图分数提取 top-$K$ 候选，并确保原生成关系被包含：

$$
\mathcal C_K(s,o)
=\operatorname{TopK}_{r\in\mathcal R(s,o)}
a_\theta\big(I,(s,r,o)\big)
\cup\{r_{\mathrm{gen}}\}.
\tag{6}
$$

候选可以跨结构类型竞争。例如，错误的 `riding` 可以被修复为 `next to`，而不是只能在其他动作关系中选择。需要注意，$r^{-1}$ 只用于构造 $(o,r^{-1},s)$ 这个论元交换正控制，不能被无条件当作固定 $(s,o)$ 顺序下的修复候选。为了使不同类型的候选分数可比，所有候选都必须至少使用 identity、argument inversion、pair crop 和匹配负控制这组通用测量。

---

## 4. 声明分数与匹配视觉负控制

### 4.1 Yes/No log-odds

对任意图像 $J$ 和关系声明 $c$，构造问题

> Is it true that the $s$ is $r$ the $o$ in the image?

定义声明 log-odds：

$$
a_\theta(J,c)
=\log p_\theta(\text{Yes}\mid J,q(c))
-\log p_\theta(\text{No}\mid J,q(c)).
\tag{7}
$$

若 `Yes` 或 `No` 由多个 token 组成，实现应使用完整答案序列的长度归一化 log-probability，不应假设两个答案都是单 token。

对于需要关系候选词位打分的模型，也可定义

$$
a_\theta(J,c_r)
=\ell_J(r)-
\log\sum_{r'\in\mathcal C_K\setminus\{r\}}
\exp \ell_J(r'),
\tag{8}
$$

即当前关系相对其他候选的 log-odds。主实验必须在同一模型内固定一种打分接口，不能按数据集或关系类型切换。

### 4.2 匹配负控制

对每个正控制图像 $I_e$，构造三类实体负控制：

$$
I_{e,s}^{-}=M_s(I_e),
\qquad
I_{e,o}^{-}=M_o(I_e),
\qquad
I_{e,so}^{-}=M_{s,o}(I_e),
\tag{9}
$$

其中 $M_s$、$M_o$ 和 $M_{s,o}$ 分别遮挡 subject、object 和两者的视觉区域。遮挡区域来自：

- 受控实验：数据集提供的 ground-truth box/mask；
- 端到端实验：冻结的开放词汇定位器与分割器。

负控制不使用生成式 inpainting，以避免在遮挡过程中生成新的实体或关系。主实现应比较 mean-fill、blur 与中性灰块，并在校准集上选择总体分布偏移最小的实现。

### 4.3 实体条件化视觉证据

对正控制 $e$ 和负控制类型 $k\in\{s,o,so\}$，定义证据差：

$$
d_{e,k}(c)
=a_\theta(I_e,c_e)
-a_\theta(I_{e,k}^{-},c_e).
\tag{10}
$$

直觉上：

- $d_{e,k}(c)$ 大：移除关键实体后，模型对关系的支持明显下降；
- $d_{e,k}(c)\approx0$：有无关键实体时模型都给出类似回答，关系支持更可能来自先验；
- $d_{e,k}(c)<0$：移除实体后模型反而更相信该关系，表明当前负控制或模型响应异常。

#### 命题 1：负控制差分下的先验消除

若对任意 $e,k$ 有

$$
\begin{aligned}
a_\theta(I_e,c_e)
&=u_e(c)+v_{e,k}(c)+\epsilon_{e,k},\\
a_\theta(I_{e,k}^{-},c_e)
&=u_e(c)+\epsilon_{e,k}^{-},
\end{aligned}
\tag{11}
$$

即匹配负控制保留相同的文本/模板先验 $u_e(c)$，但移除了当前关系的实体条件化视觉证据，则

$$
d_{e,k}(c)
=v_{e,k}(c)+\epsilon_{e,k}-\epsilon_{e,k}^{-}.
\tag{12}
$$

因此，在该假设下，文本和模板先验在差分中被消除。

> **边界**：实体遮挡不是严格的 Pearl-style do-intervention。式 (11) 是必须通过遮挡形式对照、分布偏移指标和 oracle/predicted mask 对照实证检查的工作假设。

---

## 5. 标准化测量与过度识别估计

### 5.1 为什么不能直接平均

不同算子的证据差具有不同量纲和方差。例如，subject-only mask 对 `wearing` 的影响可能大于 object-only mask，而 vflip 对自然图像的噪声可能远大于 hflip。因此，COVER 不直接平均原始 $d_{e,k}$。

令 $j=(e,k)$ 表示一个“正控制×负控制”测量。在独立训练/校准集上，使用已标注的真假关系估计：

- $\hat b_j$：该测量在假关系或 null claim 上的稳健中心；
- $\hat g_j>0$：该测量的校准信号间隔，默认定义为真关系证据差中位数与 null 中心 $\hat b_j$ 之差；
- $\hat\Sigma$：真关系样本上标准化测量的 shrinkage covariance。

具体地，

$$
\hat b_j
=\operatorname{median}\{d_j(c_i):Y_i=0\},
\qquad
\hat g_j
=\operatorname{median}\{d_j(c_i):Y_i=1\}-\hat b_j.
$$

若 $\hat g_j\le g_{\min}$，说明该算子在校准数据上不能提供正向视觉区分信号；主实现将其标记为无效测量，而不通过取绝对值强行保留。

标准化测量为

$$
x_j(c)
=\frac{d_j(c)-\hat b_j}
{\hat g_j+\varepsilon},
\tag{13}
$$

其中 $\varepsilon>0$ 用于避免数值不稳定。校准参数只按算子类型共享，不使用具体 relation ID，以便在未见关系上评估迁移能力。

### 5.2 共同视觉支持

对声明 $c$，收集 $m_c$ 个有效测量：

$$
\mathbf x_c
=\big[x_1(c),\ldots,x_{m_c}(c)\big]^\top.
$$

COVER 假设，可靠关系声明的这些对齐测量围绕一个共同视觉支持 $\mu_c$：

$$
\mathbf x_c
=\mu_c\mathbf 1+\boldsymbol\eta_c.
\tag{14}
$$

令

$$
\hat W
=\big(\hat\Sigma+\gamma I\big)^{-1}
\tag{15}
$$

为 shrinkage inverse covariance，$\gamma>0$ 由校准集确定。线性 GMM/广义最小二乘形式为

$$
\hat\mu_c
=\arg\min_\mu
\big(\mathbf x_c-\mu\mathbf1\big)^\top
\hat W
\big(\mathbf x_c-\mu\mathbf1\big),
\tag{16}
$$

其闭式解为

$$
\boxed{
\hat\mu_c
=\frac{\mathbf1^\top\hat W\mathbf x_c}
{\mathbf1^\top\hat W\mathbf1}
}.
\tag{17}
$$

$\hat\mu_c$ 表示在校正了算子偏差、尺度和测量相关性后，多个视图共同提供的关系视觉支持。

### 5.3 过度识别冲突

当 $m_c>1$ 时，可以检查多个测量是否能被单一 $\mu_c$ 解释。定义

$$
J_c
=\big(\mathbf x_c-\hat\mu_c\mathbf1\big)^\top
\hat W
\big(\mathbf x_c-\hat\mu_c\mathbf1\big),
\tag{18}
$$

并用自由度归一化：

$$
\widetilde J_c
=\frac{J_c}{\max(1,m_c-1)}.
\tag{19}
$$

$\widetilde J_c$ 的意义是：

- 小：多个测量可由一个共同关系证据解释；
- 大：不同干预得到的证据相互冲突。

因此，必须联合解释 $\hat\mu_c$ 和 $\widetilde J_c$：

| $\hat\mu_c$ | $\widetilde J_c$ | 解释 |
|---:|---:|---|
| 高 | 低 | 多个视图共同且稳定地支持关系 |
| 低 | 低 | 多个视图一致地缺乏视觉证据；可能是稳定语言先验 |
| 高 | 高 | 部分视图强支持，但存在失效算子或不稳定视觉线索 |
| 低 | 高 | 证据弱且彼此矛盾，应拒答 |

### 5.4 稳健估计与失效算子影响

为降低单个分布外视图对 $\hat\mu_c$ 的破坏，主实现使用 Huber 稳健版本：

$$
\hat\mu_c^{\mathrm{rob}}
=\arg\min_\mu
\sum_{j=1}^{m_c}
w_j\rho_\kappa\big(x_j(c)-\mu\big),
\tag{20}
$$

其中

$$
\rho_\kappa(z)=
\begin{cases}
\tfrac12z^2,&|z|\le\kappa,\\
\kappa|z|-\tfrac12\kappa^2,&|z|>\kappa.
\end{cases}
\tag{21}
$$

$w_j$ 来自 $\hat W$ 的对角或等价白化后权重。主文使用稳健估计进行决策，式 (17) 的闭式解用于理论和可解释消融。

定义算子 $j$ 的 leave-one-measurement-out 影响：

$$
\mathcal I_j(c)
=J_c-J_{c,-j},
\tag{22}
$$

其中 $J_{c,-j}$ 表示删除测量 $j$ 后的冲突。$\mathcal I_j$ 越大，说明当前算子越可能是主要冲突源。

#### 命题 2：单个失效测量的冲突下界

在独立对角权重情况下，假设除测量 $j$ 外的所有测量都等于 $\mu_c$，而第 $j$ 个测量存在偏差 $\delta$。忽略随机噪声时，有

$$
J_c
=\delta^2w_j
\left(1-
\frac{w_j}{\sum_{l=1}^{m_c}w_l}
\right).
\tag{23}
$$

因此，只要失效测量的权重非零且 $m_c>1$，系统偏差就会在 $J_c$ 中产生与 $\delta^2$ 成比例的正冲突。

> **注意**：$J_c$ 在本方法中是过度识别启发的实例级冲突统计量。因为同一图像的多个视图高度相关，不直接使用经典 Hansen–J 检验的渐近卡方 $p$ 值。

---

## 6. 关系非一致分数与 Conformal 决策

### 6.1 候选关系非一致分数

对每个候选关系 $r\in\mathcal C_K(s,o)$，得到

$$
c_r=(s,r,o),
\qquad
\hat\mu_r,
\qquad
\widetilde J_r.
$$

定义非一致分数：

$$
A(I,s,o,r)
=-\hat\mu_r
+\lambda\widetilde J_r,
\tag{24}
$$

其中 $\lambda\ge0$ 在独立 development split 上选择，选择目标为最小化风险–覆盖曲线下面积，不使用最终测试集。

$A$ 越小，表示关系的共同视觉支持越强、视图冲突越小。

### 6.2 Split-conformal 关系集

设独立 conformal calibration split 包含 $n$ 个样本，每个样本的真关系为 $r_i^*$。计算

$$
\alpha_i=A(I_i,s_i,o_i,r_i^*).
$$

给定目标错误率 $\delta\in(0,1)$，令

$$
q_{1-\delta}
=\operatorname{Quantile}_{\lceil(n+1)(1-\delta)\rceil/n}
\{\alpha_i\}_{i=1}^{n}.
\tag{25}
$$

测试时输出关系候选集

$$
\boxed{
\Gamma_\delta(I,s,o)
=\left\{
r\in\mathcal C_K(s,o):
A(I,s,o,r)\le q_{1-\delta}
\right\}
}.
\tag{26}
$$

在样本交换性、真关系包含在候选集中且校准/测试分布匹配的条件下，有边际覆盖保证：

$$
\Pr\left[
r^*\in\Gamma_\delta(I,s,o)
\right]\ge1-\delta.
\tag{27}
$$

该保证是对“真关系被包含在输出集中”的边际覆盖保证，不能被表述为每个单独样本都有 $1-\delta$ 的正确率。

### 6.3 ACCEPT / REPAIR / ABSTAIN

设原生成关系为 $r_{\mathrm{gen}}$，决策规则为：

$$
D(I,c)=
\begin{cases}
\texttt{ACCEPT},
&\Gamma_\delta=\{r_{\mathrm{gen}}\},\\
\texttt{REPAIR}(r'),
&\Gamma_\delta=\{r'\},\ r'\neq r_{\mathrm{gen}},\\
\texttt{ABSTAIN},
&|\Gamma_\delta|=0\ \text{or}\ |\Gamma_\delta|>1.
\end{cases}
\tag{28}
$$

`ABSTAIN` 在开放生成中有两种输出形式：

1. 删除不可靠的关系短语，保留已验证的实体声明；
2. 在高风险场景显式输出“无法从当前图像可靠确定两者关系”。

---

## 7. COVER 完整推理算法

### 7.1 算法框

```text
Input:
    image I
    generated response y
    frozen LVLM p_theta
    relation ontology R and structural signatures S
    entity localizer L
    calibration statistics {b_j, g_j, W, lambda, q_(1-delta)}

Output:
    edited response y_hat
    per-claim decisions and diagnostics

1. T_y <- ExtractTriples(y)
2. Locate all subject/object regions required by T_y

3. For each generated claim c_gen = (s, r_gen, o):
4.     C <- BuildCandidateSet(I, s, o, r_gen, R, K)

5.     For each candidate r in C:
6.         c_r <- (s, r, o)
7.         E_r <- CompileValidOperators(c_r, S(r))

8.         For each positive-control operator e in E_r:
9.             (I_e, c_e) <- ApplyOperator(e, I, c_r)
10.            a_pos <- ClaimLogOdds(p_theta, I_e, c_e)

11.            For k in {subject, object, both}:
12.                I_neg <- ApplyMatchedNegativeControl(I_e, c_e, k)
13.                a_neg <- ClaimLogOdds(p_theta, I_neg, c_e)
14.                d_(e,k) <- a_pos - a_neg
15.                x_(e,k) <- Standardize(d_(e,k), b_(e,k), g_(e,k))

16.        mu_r <- RobustCommonSupport({x_(e,k)}, W)
17.        J_r  <- NormalizedOverIDConflict({x_(e,k)}, mu_r, W)
18.        A_r  <- -mu_r + lambda * J_r
19.        influence_r <- LeaveOneMeasurementOutInfluence(...)

20.    Gamma <- {r in C : A_r <= q_(1-delta)}

21.    If Gamma == {r_gen}:
22.        decision <- ACCEPT
23.    Else if Gamma == {r'} and r' != r_gen:
24.        decision <- REPAIR(r')
25.    Else:
26.        decision <- ABSTAIN

27.    Edit the span of c_gen in y according to decision
28. Return edited response and all diagnostics
```

### 7.2 自适应计算预算

为避免对所有声明无条件执行全部干预，COVER 按以下顺序逐级增加证据：

1. identity + subject/object/both masks；
2. argument inversion + matched masks；
3. pair crop + matched masks；
4. 如果关系签名支持，再调用 typed geometric view；
5. 每一级结束后重新构造 $\Gamma_\delta$，若已得到稳定单元集则提前停止。

该提前停止机制只改变计算量，不改变候选关系的分数定义。评测必须同时报告无限制完整 COVER 与预算受限 COVER。

---

## 8. 与 ECD 的数学关系

对 left/right 关系，令 $g$ 为水平翻转，且

$$
\sigma_g(\text{left})=\text{right},
\qquad
\sigma_g(\text{right})=\text{left}.
$$

当 COVER 只使用 identity 和 hflip 两个正控制，暂时去掉负控制差分和标准化，并使用等权聚合时，left 的共同分数正比于

$$
\ell_I(\text{left})
+\ell_{gI}(\text{right}),
\tag{29}
$$

right 的共同分数正比于

$$
\ell_I(\text{right})
+\ell_{gI}(\text{left}).
\tag{30}
$$

这正是 ECD 的 $\alpha=1$ 等变分数。因此：

> ECD 是 COVER 在二元几何变换轨道、两个等权视图、无显式负控制情况下的解析特例。

完整 COVER 在 ECD 之上增加了：

1. 通用的论元/逆关系和实体负控制，因而能处理非空间关系；
2. 多视图过度识别冲突，因而能识别失效变换；
3. 跨关系候选修复与 conformal 拒答；
4. 生成后三元组级接口。

---

## 9. 贯穿案例：将幻觉的 `riding` 修复为 `next to`

> **再次说明**：本节所有数值均为了展示算法如何运行而设置，不是现有实验数字。为了便于理解，本节使用等权原始证据差，忽略实际系统中的标准化与协方差加权。

### 9.1 场景与初始幻觉

图像中有一名男子在一匹马旁边行走。男子并没有坐在马背上。LVLM 生成：

> A man is riding a horse.

关系抽取器得到

$$
c_{\mathrm{gen}}
=(\text{man},\text{riding},\text{horse}).
$$

假设原图上 `riding`、`next to`、`holding` 进入 top-$K$ 候选集：

$$
\mathcal C_K
=\{\text{riding},\text{next to},\text{holding}\}.
$$

### 9.2 如果只做 self-consistency

对 `riding` 构造三个等价问题：

1. Is the man riding the horse?
2. Is the horse being ridden by the man?
3. Is the man mounted on the horse?

模型对三者都高置信回答 Yes。一般 self-consistency 会将这解释为可靠。

但是，三个答案可能全部来自 `man + horse -> riding` 的同一语言共现先验。因此，“多次一致”并不能证明“来自视觉”。

### 9.3 用负控制检查 `riding`

对每个正控制视图，COVER 再遮挡 man 和 horse，得到以下演示结果。

| `riding` measurement | positive-view log-odds | masked-control log-odds | evidence drop $d_j$ |
|---|---:|---:|---:|
| original statement | 4.8 | 4.1 | 0.7 |
| inverse statement | 4.4 | 3.9 | 0.5 |
| semantic paraphrase | 4.5 | 4.0 | 0.5 |
| pair crop | 3.6 | 3.4 | 0.2 |

原始 log-odds 全部很高，但遮挡实体后只小幅下降。在等权简化下：

$$
\hat\mu_{\mathrm{riding}}
=\frac{0.7+0.5+0.5+0.2}{4}
=0.475.
$$

该候选的冲突不一定很大，因为所有视图可能一致地缺乏视觉证据。这是“低 $\hat\mu$ + 低 $J$”的典型情况：

> 模型稳定地说出了 `riding`，但这份稳定性在移除关键视觉证据后几乎没有变化。

因此 COVER 不会 ACCEPT `riding`。

### 9.4 核验候选 `next to`

`next to` 是对称关系：

$$
(\text{man},\text{next to},\text{horse})
\Longleftrightarrow
(\text{horse},\text{next to},\text{man}).
$$

COVER 使用原声明、论元交换声明和实体对裁剪。

| `next to` measurement | positive-view log-odds | masked-control log-odds | evidence drop $d_j$ |
|---|---:|---:|---:|
| man next to horse | 3.9 | 0.4 | 3.5 |
| horse next to man | 3.7 | 0.3 | 3.4 |
| pair crop | 3.8 | 0.6 | 3.2 |

等权简化共同支持为

$$
\hat\mu_{\mathrm{next\ to}}
=\frac{3.5+3.4+3.2}{3}
\approx3.37.
$$

三个证据差也彼此接近，因此 $\widetilde J_{\mathrm{next\ to}}$ 较低。这表明：

1. 实体可见时，模型对 `next to` 有明显支持；
2. 实体被移除时，支持明显消失；
3. 原声明、对称声明和局部裁剪都在测量类似的关系证据。

### 9.5 冲突候选 `holding`

假设 `holding` 得到以下证据差：

$$
\mathbf d_{\mathrm{holding}}
=\begin{bmatrix}0.4&1.8&0.3\end{bmatrix}^\top.
$$

其平均视觉支持虽高于 `riding`，但逆关系视图与其他视图强烈冲突，因而 $\widetilde J_{\mathrm{holding}}$ 较高。该候选不能被解释为稳定的共同视觉事实。

### 9.6 形成 conformal 候选集

对三个关系计算

$$
A_r=-\hat\mu_r+\lambda\widetilde J_r.
$$

假设经校准阈值筛选后：

$$
\Gamma_\delta(I,\text{man},\text{horse})
=\{\text{next to}\}.
$$

因为该集合为单元集，且其中的关系不是原生成关系，所以决策为

$$
D(I,c_{\mathrm{gen}})
=\texttt{REPAIR}(\text{next to}).
$$

原文本

> A man is riding a horse.

被修改为

> A man is next to a horse.

### 9.7 证据不足时的输出

如果图像存在严重遮挡，导致 `next to` 和 `behind` 都通过校准，则

$$
\Gamma_\delta
=\{\text{next to},\text{behind}\}.
$$

COVER 不强制选择 top-1，而是输出 `ABSTAIN`：

> A man and a horse are visible, but their exact relationship cannot be reliably determined.

这个结果展示了 COVER 的核心区分：

- self-consistency 问“模型是否反复说同一件事”；
- COVER 问“是否真的存在一个视觉事实，能够解释模型为什么这样说”。

---

## 10. 训练、校准与数据拆分

### 10.1 不训练 LVLM

COVER 始终冻结 LVLM $p_\theta$。可学习或可估计的部分仅包括：

- 算子偏差 $\hat b_j$；
- 算子信号间隔 $\hat g_j$；
- shrinkage covariance $\hat\Sigma$；
- 冲突权重 $\lambda$；
- conformal quantile $q_{1-\delta}$；
- 可选的小型异方差探针。

可选探针不得输入具体 relation ID。允许的输入只包括算子类型、原始 log-odds、遮挡差、图像质量和定位置信度等关系无关特征。

### 10.2 四层拆分

为避免校准泄漏，数据应按图像分组为：

1. **Estimator train**: 估计 $\hat b_j$、$\hat g_j$ 与 $\hat\Sigma$；
2. **Development**: 选择 $K$、$\lambda$、$\gamma$、Huber $\kappa$ 和遮挡方式；
3. **Conformal calibration**: 仅估计 $q_{1-\delta}$；
4. **Test**: 最终评估，不再修改任何阈值。

同一原图产生的所有裁剪、遮挡和几何变换必须位于同一 split。

### 10.3 未见关系泛化

执行 leave-one-family-out 评估：

- 使用 spatial、comparative、containment 估计校准参数，测试 interaction；
- 轮流将每个关系族作为未见测试族；
- 禁止在权重、探针或阈值中使用 relation ID。

该实验是证明 COVER 不是关系特定规则集的核心证据。

---

## 11. 方法依赖的假设

### A1. 候选覆盖

真实关系 $r^*$ 需要包含在 $\mathcal C_K(s,o)$ 中。若不满足，COVER 可以删除错误关系，但无法修复为正确关系。

### A2. 正控制语义有效性

式 (5) 对所使用算子近似成立。不能保证真值保持的变换不能进入主分数。

### A3. 匹配负控制有效性

遮挡主要移除实体关系证据，同时近似保留问题模板和语言先验。

### A4. 有足够多的有效测量

每个候选关系至少需要两个非退化测量才能定义过度识别冲突。主实验默认要求 $m_c\ge3$。

### A5. Conformal 交换性

校准样本和测试样本需要在目标分布上可交换。跨数据集或显著分布偏移时，需重新校准或仅报告经验风险–覆盖曲线。

---

## 12. 主要失效模式

### 12.1 背景捷径在遮挡后仍然存在

例如，即使遮挡 man 和 horse，草地、马鞍或牵引绳仍可能使模型预测 `riding`。这时负控制差分无法完全隔离视觉捷径。

### 12.2 遮挡伪影主导模型响应

如果 mask 产生强分布外伪影，$d_{e,k}$ 可能只在测量模型对伪影的敏感度。需要通过多种遮挡形式、图像质量分层和 oracle/predicted mask 对照检查。

### 12.3 候选集不包含真关系

该情况下只能 `ABSTAIN`，不能安全 `REPAIR`。必须单独报告 candidate recall@K，避免将候选生成失败误归因于 COVER 核验器。

### 12.4 多个视图共享同一个错误视觉特征

若所有正控制和负控制都保留了同一个伪相关视觉特征，则 $\hat\mu$ 可能高且 $J$ 可能低。这是 COVER 无法通过内部冗余观测排除的强失效域。

### 12.5 关系本身不可从 2D 图像确定

front/behind、遮挡关系或社会交互关系可能需要深度、时间或外部知识。COVER 应将其输出为多元候选集或拒答，不应将高拒答率隐藏在 top-1 accuracy 中。

---

## 13. 需要实证检验的 Method 主张

| Method claim | 必需证据 | 当前状态 |
|---|---|---|
| 原始高置信关系中混合语言先验与视觉证据 | 原图/遮挡图/空白图的条件对照 | 部分有 ECD blank 证据，多关系待验证 |
| 负控制差分比原始 confidence 更能识别关系幻觉 | AUROC/AUPRC、置信度条件分层 | 待验证 |
| $J$ 能识别失效视图 | hflip/vflip、paraphrase/crop 的影响分析 | above/below 有初步失效现象，待形式化 |
| 统一估计器可跨关系族泛化 | leave-one-family-out | 待验证 |
| `REPAIR` 减少关系幻觉而不只是删除文本 | relation precision/recall/F1 + correction precision | 待验证 |
| conformal 集在目标分布上达到预期覆盖 | empirical coverage vs target coverage | 待验证 |

---

## 14. 方法的最小可证伪版本

在投入完整生成修复系统前，应先实现一个最小版本检查核心假设：

1. 数据：R-Bench、Reefknot 或 MMRel 中带真假标签的关系声明；
2. 模型：已有 LLaVA-1.5-7B；
3. 正控制：identity + argument inversion；
4. 负控制：subject mask + object mask + both mask；
5. 分数：Yes/No log-odds；
6. 指标：原始 confidence、平均 evidence drop、$\hat\mu$、$J$ 的 AUROC/AUPRC；
7. 关键检验：在原始 confidence 分位数内部，真假关系的 $\hat\mu$ 和 $J$ 是否仍然可分。

若负控制差分在至少两个非空间关系族上不优于原始 confidence，则应停止完整 COVER 开发，而不应通过增加更多组件来掩盖核心假设失败。

---

## 15. 可用于论文的核心表述

### 15.1 One-sentence contribution

> We formulate relation hallucination verification as an overidentified latent-evidence estimation problem, where semantically typed interventions provide redundant measurements of a shared visual fact and matched entity-level negative controls separate visual support from persistent language priors.

### 15.2 通俗版本

> 一致性只检查模型是否反复说同一件事；COVER 检查是否真的存在一个视觉事实，能够解释模型为什么这样说。

### 15.3 不应提前使用的强主张

在完成跨模型、跨数据集和 leave-one-family-out 验证前，不应声称：

- COVER 已经通用解决了多类关系幻觉；
- 实体遮挡是严格因果干预；
- $J_c$ 具有经典 Hansen–J 检验的卡方显著性；
- conformal 覆盖保证等于每个样本的正确率保证；
- COVER 是首个关系幻觉缓解方法。

---

## 16. 与近邻工作的方法边界

- [R-Bench](https://proceedings.mlr.press/v235/wu24l.html) 主要定义、评估和分析 relationship hallucination；COVER 的对象是生成后关系核验与修复。
- [ChainMPQ](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2dbab01f42b0a4f35733a5e413384ca9-Abstract-Conference.html) 通过 subject–object–relation 分解和交错文本–图像推理链缓解关系幻觉；COVER 检验多个干预响应是否由同一视觉事实解释。
- [Tri-HE](https://kaichen1998.github.io/projects/tri-he/) 在 object–relation–object 三元组层同时评估物体和关系幻觉，可作为 COVER 开放生成主评测。
- [Reefknot](https://aclanthology.org/2025.findings-acl.322.pdf) 使用置信度进行 detect-then-calibrate；COVER 不将原始 confidence 直接视为视觉支持。
- [UHP Detection](https://arxiv.org/abs/2608.03817) 用图像/文本扰动和逻辑极性形成结构化一致性特征；COVER 的区分点必须是语义类型化干预、匹配负控制、共同潜在证据估计、失效算子诊断与关系修复，而不是简单的“更多一致性特征”。

---

## 17. 符号表

| Symbol | Meaning |
|---|---|
| $I$ | 原始图像 |
| $q$ | 用户问题或生成 prompt |
| $y$ | LVLM 生成文本 |
| $p_\theta$ | 参数冻结的 LVLM |
| $c=(s,r,o)$ | subject–relation–object 关系声明 |
| $\mathcal R$ | 支持的关系 ontology |
| $\mathcal S(r)$ | 关系 $r$ 的结构签名 |
| $e$ | 正控制算子 |
| $T_e$ | 图像变换 |
| $\pi_e$ | subject/object 置换 |
| $\sigma_e$ | 关系语义映射 |
| $I_e,c_e$ | 变换后图像与关系声明 |
| $I_{e,k}^{-}$ | 与视图 $e$ 匹配的实体负控制 |
| $a_\theta(I,c)$ | 关系声明 Yes/No log-odds |
| $d_{e,k}$ | 正控制与负控制之间的证据差 |
| $\hat b_j,\hat g_j$ | 测量 $j$ 的 null 中心与校准信号间隔 |
| $x_j$ | 标准化后的第 $j$ 个测量 |
| $\hat\Sigma,\hat W$ | 测量协方差及其正则化逆矩阵 |
| $\hat\mu_c$ | 估计的共同视觉支持 |
| $J_c,\widetilde J_c$ | 过度识别冲突及其归一化版本 |
| $\mathcal I_j$ | 测量 $j$ 的 leave-one-out 冲突影响 |
| $A(I,s,o,r)$ | 候选关系的非一致分数 |
| $\Gamma_\delta$ | conformal 关系候选集 |
| $D(I,c)$ | `ACCEPT / REPAIR / ABSTAIN` 决策 |

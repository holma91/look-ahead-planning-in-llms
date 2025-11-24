```
Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 7713–
November 12-16, 2024 ©2024 Association for Computational Linguistics
```

# Unlocking the Future: Exploring Look-Ahead Planning Mechanistic

# Interpretability in Large Language Models

## Tianyi Men1,2, Pengfei Cao1,2, Zhuoran Jin1,2, Yubo Chen1,2,\*, Kang Liu1,2,3, Jun Zhao1,

(^1) The Key Laboratory of Cognition and Decision Intelligence for Complex Systems,

## Institute of Automation, Chinese Academy of Sciences, Beijing, China

(^2) School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China
(^3) Shanghai Artificial Intelligence Laboratory
{tianyi.men, pengfei.cao, zhuoran.jin, yubo.chen, kliu, jzhao}@nlpr.ia.ac.cn

## Abstract

```
Planning, as the core module of agents, is cru-
cial in various fields such as embodied agents,
web navigation, and tool using. With the de-
velopment of large language models (LLMs),
some researchers treat large language models
as intelligent agents to stimulate and evaluate
their planning capabilities. However, the plan-
ning mechanism is still unclear. In this work,
we focus on exploring the look-ahead planning
mechanism in large language models from the
perspectives of information flow and internal
representations. First, we study how planning
is done internally by analyzing the multi-layer
perception (MLP) and multi-head self-attention
(MHSA) components at the last token. We find
that the output of MHSA in the middle layers at
the last token can directly decode the decision
to some extent. Based on this discovery, we fur-
ther trace the source of MHSA by information
flow, and we reveal that MHSA mainly extracts
information from spans of the goal states and
recent steps. According to information flow, we
continue to study what information is encoded
within it. Specifically, we explore whether fu-
ture decisions have been encoded in advance in
the representation of flow. We demonstrate that
the middle and upper layers encode a few short-
term future decisions to some extent when plan-
ning is successful. Overall, our research ana-
lyzes the look-ahead planning mechanisms of
LLMs, facilitating future research on LLMs
performing planning tasks.
```

## 1 Introduction

Planning is the process of formulating a series of
actions to transform a given initial state into a de-
sired goal state (Valmeekam et al., 2024; Zhang
et al., 2024). As the core module of agents (Xi
et al., 2023; Wang et al., 2024a), planning has been
widely applied in many fields such as embodied
agents (Shridhar et al., 2020; Wang et al., 2022),
\*Corresponding authors.

```
Step 3 : pick up red
Step 4 : put on yellow
```

```
yellow
table
```

```
yellow
blue
```

```
Init State
```

```
I can only foresee the next Goal State
one step of action. Given the
goal state and init state, I'm
only thinking of the next one
step between picking yellow
and blue, only step 1 color is
in my mind. I pick up
yellow. After finishing step 1, I think about step 2 action.
```

```
I can foresee the next several
steps of actions. Given the
goal state and init state, I'm
thinking of the next several
steps, step 1,2,3,4 color is in
my mind. I pick up blue. After
finishing step 1, I think about
step 2,3,4 actions.
```

```
S11 S
```

```
S21 S22 S23 S
```

```
Look-Ahead
```

```
Step 1 : pick up yellow
Step 2 : put on blue
```

```
green
yellow
```

```
blue
red
Greedy
```

```
Figure 1: An example of greedy and look-ahead plan-
ning.
```

```
web navigation (Zhou et al., 2023; Deng et al.,
2024) and tool using (Xu et al., 2023; Qin et al.,
2023). With the development of large language
models, some researchers treat large language mod-
els as intelligent agents to solve complex tasks.
This is because large language models may pos-
sess some preliminary planning capabilities (Huang
et al., 2022). Recently, researchers have made ef-
forts to stimulate and evaluate the planning capa-
bilities of large language models. They propose
prompt engineering (Zheng et al., 2023) and in-
struction fine-tuning (Zeng et al., 2023) to boost
the planning abilities of large language models.
Additionally, some researchers construct bench-
marks such as AgentBench (Liu et al., 2023) and
AgentGym (Xi et al., 2024) to evaluate the planning
capabilities of large models. Although they have
made some progress, the underlying mechanisms
in planning capabilities of large language models
remain a largely unexplored frontier. Revealing
7713
```

the planning mechanisms of large language models
helps to better understand and improve their plan-
ning capabilities. Therefore, we focus on exploring
the underlying mechanisms behind the planning
abilities of large language models.
In this work, we focus on exploring look-ahead
planning mechanisms in large language models.
We study the classical planning task Blocksworld,
which is a fully-observed setting. All entity states
are known from the init state and goal state, so
exploration is not needed (Zhang et al., 2024). As
illustrated in Figure 1, given an initial state and
a goal state of Blocksworld, the model can only
pick up or put down one block. The model must
generate a sequence of actions to transform the
initial state into the goal state, as shown by the
green path. However, it is still unclear whether the
model, at stept, greedily considers only the action
att+ 1or look-ahead considers the actions att+ 2
and beyond. Inspired by psychology, humans en-
gage in look-ahead thinking when making plans
(Baumeister et al., 2016). Based on this, we fur-
ther propose the hypothesis of model look-ahead
planning, which is as follows:

- Look-Ahead Planning Decisions Existence
  Hypothesis: In the task of planning with large
  language models, given a rule, an initial state,
  a goal state, and task description prompts. At
  the current step, the model needs to predict
  the next action, the probe can detect decisions
  to some extent for future steps in the internal
  representations in the short term within a fully-
  observed setting when planning is successful.
  We design a two-stage paradigm to verify this
  hypothesis. It can be divided into the finding in-
  formation flow stage and the probing internal rep-
  resentations stage. The first stage is to analyze
  the information flow and component functions dur-
  ing planning (§5). The second stage is examining
  whether the model stores future information in in-
  ternal representations (§6). The specifics are as
  follows:
  (1) In the first stage, we study how planning is
  done internally by analyzing the MLP and MHSA
  components at the last token. Inspired by meth-
  ods of calculating extraction rates methods (Geva
  et al., 2023), we find the output of MHSA in the
  middle layers at the last token can directly decode
  the correct colors to some extent (§5.1). Based on
  this discovery, we further investigate the sources of
  information on MHSA. We trace the source of the

```
decisions. And find that planning mainly depends
on spans of the goal states and recent steps (§5.2).
(2) In the second stage, we study what informa-
tion is encoded in the information flow and whether
this information has been considered in advance for
future decisions. For future decisions existence, we
use the probing method to probe future decisions
and reveal that the middle and upper layers encode
a few short-term future decisions when planning
is successful (§6.1). For history step causality, we
prevent the information flow from history steps and
explore the impact of different history steps on the
final decision (§6.2).
In summary, our contributions are as follows:
```

- To the best of our knowledge, this work is the
  first to investigate the planning interpretability
  mechanisms in large language models. We
  demonstrate theLook-Ahead Planning Deci-
  sions Existence Hypothesis.
- We reveal that the internal representations of
  LLMs encode a few short-term future deci-
  sions to some extent when planning is success-
  ful. These look-ahead decisions are enriched
  in the early layers, with accuracy decreasing
  as planning steps increase.
- We prove that MHSA mainly extracts informa-
  tion from spans of the goal states and recent
  steps. The output of MHSA in the middle lay-
  ers at the last token can directly decode the
  correct decisions partially in planning tasks.

## 2 Experimental Setup

```
In this paper, we study the Blocksworld task in a
fully-observed setting where all entity states are
known from the init state and goal state, so explo-
ration is not needed (Zhang et al., 2024). Given a
ruleR, an initial stateSinit, a goal stateSgoal, task
description promptsC, the current stept, history
a 1... at, model needs to predict the next action
at+1in accordance with its generative distribution
p(at+1 | R, Sinit, Sgoal, C, a 1... at)(Hao et al.,
2023). In this paper, all inputs are in text form. All
inferences are performed using the teacher-forcing
method. Previous evaluation works (Valmeekam
et al., 2023) mainly involved generating a complete
plan and then placing it into the environment for as-
sessment. However, since our primary focus is on
open-source models, we have reduced the difficulty
by using a fill-in-the-blank format for evaluating
the models. An example is shown in Figure 2.
```

Data Previous Blocksworld evaluation bench-
marks (Valmeekam et al., 2023) put the plans gen-
erated by models into an environment to verify
the correctness. However, existing interpretability
methods, such as information flow (Wang et al.,
2023a), require gold labels. Therefore, we syn-
thesize a dataset containing optimal plans, with
specific data statistics shown in Table 1. We gener-
ate data with 4, 5, and 6 color varieties, 4 piles, and
a maximum of 6 steps, where pick-up and stack are
considered as two different steps. There are three
levels: LEVEL1 (L1) with two steps, LEVEL2 (L2)
with four steps, and LEVEL3 (L3) with six steps.
We choose the optimal path from the initial step to
the final step. For samples with multiple optimal
paths, we select one to include in the training set,
ensuring that samples in the test set have unique
optimal paths. We split the dataset into training and
test sets with a ratio of 1:3.
Metric In the Blocksworld task, we use two met-
rics: single-step success rate and complete plan
success rate. The single-step success rate evaluates
whether each individual action is correct, defined
as:
Sstep=

### 1

### N

### ∑N

```
i=
```

```
ri (1)
```

whereNis the total number of steps andriindi-
cates the success of thei-th step (1 for success, 0
otherwise). The complete plan success rate evalu-
ates whether the entire planning process is correct,
defined as:

```
Splan=
```

### 1

### M

### ∑M

```
j=
```

```
Rj (2)
```

whereMis the total number of tested plans and
Rj indicates the success of thej-th plan (1 for
complete success, 0 otherwise).
Model We evaluate two large language models:
Llama-2-7b-chat (Touvron et al., 2023) and Vicuna-
7B (Chiang et al., 2023). Since open-source models
have preliminary planning capabilities, we enhance
the ability of large language models to complete
planning tasks through training.
Experiment Setting We conduct full parameter
fine-tuning on Llama-2-7b-chat-hf and Vicuna-7B
for 3 epochs. The training process involves a global
batch size of 20, using the Adam optimizer with
a learning rate of 5e-5. Llama-2-7b-chat-hf and
Vicuna-7B achieve complete plan success rates of

```
Rule:
You can pick-up color1. stack color1 on-top-of color2.
All the blocks are on the table. There is no order in
the piles. Please output the optimal plan.
Init state:
<empty>
<black>
<white on gray on red on blue>
<green>
Goal state:
<gray on black>
<red on blue>
<green>
<white>
Plan:
step 1: pick-up ____ (answer: white)
step 2: stack white on-top-of ____ (answer: table)
step 3: pick-up ____ (answer: gray)
step 4: stack gray on-top-of ____ (answer: black)
```

```
Figure 2: An example of Blocksworld.
```

```
61% and 63%, respectively, at LEVEL 3 with 6
blocks. We sample 400 correct data points from
LEVEL 3 with 6 blocks for our analysis. We con-
duct experiment based on HuggingFace’s Trans-
formers^1 , PyTorch^2 , baukit^3 and pyvene^4 (Wu
et al., 2024b).
```

## 3 Background

```
A transformer-based language model begins by
converting an input text into a sequence ofN
tokens, denoted ass 1 ,... , sN. Each tokensiis
mapped to a vectorx^0 i ∈Rd.E∈R|V|×dis the
decoder matrix in the last layer, whereV is the
vocabulary,dis embedding dimension. Each layer
comprises a multi-head self-attention (MHSA) sub-
layer followed by a multi-layer perception (MLP)
sublayer (Vaswani et al., 2017). Formally, the rep-
resentationxℓiof tokeniat layerℓcan be obtained
as follows:
```

```
xℓi=xℓi−^1 +attnℓi+mℓi (3)
```

```
attnℓi andmℓi represent the outputs of the
MHSA and MLP sub-layers of theℓ-th layer, re-
spectively. By usingE, an output probability dis-
tribution can be obtained from the final layer repre-
sentation:
```

```
pLi =softmax(ExLi) (4)
```

(^1) https://github.com/huggingface/transformers/
(^2) https://github.com/pytorch/pytorch/
(^3) https://github.com/davidbau/baukit/
(^4) https://github.com/stanfordnlp/pyvene

```
0 5 10 15 20 25 30
layers
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
1.
```

```
extration rates
```

```
MLP
MHSA
Layer
```

```
Figure 3: Extraction rate of different components in
Llama-2-7b-chat-hf.
```

## 4 Overview of Analysis

We analyze the look-ahead planning mechanisms
of the models from two stages. (1) In the first
stage, we explore the internal mechanisms of this
process in planning tasks from the perspectives of
information flow and component functions. We
demonstrate that the middle layer MHSA can di-
rectly decode the answers to a certain extent, and
we prove that MHSA mainly extracts information
from spans of the goal states and recent steps (§5).
(2) In the second stage, to determine the presence
of future decisions, we employ the probing method
to examine future decisions, uncovering that the
intermediate and upper layers encode these deci-
sions. Regarding the causality of historical steps,
we inhibit the information flow from past steps and
analyze the effects of different historical steps on
the ultimate decision (§6).

## 5 Information Flow in Planning Tasks

To trace the source of the correct answer, we be-
gin with the last token. For example, in the first
step "pick up", the last token is "up". The model
should process the initial state, target state, and
history of steps to decide which color to pick up,
such as "blue". We analyze this process from two
perspectives. (1) First, we study MLP and MHSA
functions at the last token by extraction rates (Geva
et al., 2023). We find that the output of MHSA in
the middle layers can directly decode the correct
colors to a certain extent (§5.1). (2) Based on this,
we further trace the source of the correct colors by
information flow (Wang et al., 2023a). From the
perspective of early and late planning stages, we
prove that MHSA mainly extracts information from
spans of the goal states and recent steps (§5.2).

```
0 5 10 15 20 25 30
layers
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
1.
```

```
extration rates
```

```
MLP
MHSA
Layer
```

```
Figure 4: Extraction rate of different components in
Vicuna-7b.
```

```
5.1 Attention Extract the Answers
From the perspective of the model’s internal com-
ponents, we analyze the functions of different com-
ponents of the models. The first question is how
the model extracts answers from history. We start
from the position of the last token and study the
roles of the MLP and MHSA components in the
answer generation process. Specifically, we inves-
tigate whether different components at different
layers can directly decode the final answer.
Experiments We use the extraction rate (Geva
et al., 2023) to analyze the functions of different
components. Specifically, we calculate the extrac-
tion rate:
e∗:=argmax
```

### (

```
pLN
```

### )

### (5)

```
̂e:=argmax
```

### (

```
EhℓN
```

### )

### (6)

```
In this equation,hrepresents the internal repre-
sentation of the MLP, MHSA and layer output,N
is the position of the last token,ℓis the layer of
models,ℓ∈[1, L]. Whene∗=̂e, it is considered
as an extraction event. We calculate the extraction
rate of the last token for each layer for each step in
the Blocksworld. We then compute the mean and
variance of these rates.
Results and Analysis As shown in Figure 3 and
Figure 4, we observe that (1) MHSA has a higher
extraction rate compared to MLP, indicating that
attention is primarily responsible for answer ex-
traction. (2) Layer output gradually forms a stable
answer in the middle to upper layers (from the 15th
layer to the 20th layer). In these layers, the ex-
traction rate of MHSA is significantly higher than
MLP, suggesting that MHSA plays a major role
during the decision-making period. (3) The vari-
ance in extraction rates across different steps is
```

```
0 5 10 15 20 25 30
```

```
init tokeninit state
goal tokengoal state
action promptplan token
```

```
step 1
```

```
0 5 10 15 20 25 30
```

```
init tokeninit state
goal tokengoal state
plan tokenhistory 1
action prompt
```

```
step 2
```

```
0 5 10 15 20 25 30
```

```
init tokeninit state
goal tokengoal state
plan tokenhistory 1
action prompthistory 2
```

```
step 3
```

```
0 5 10 15 20 25 30
```

```
init tokeninit state
goal tokengoal state
plan tokenhistory 1
history 2history 3
action prompt
```

```
step 4
```

```
0 5 10 15 20 25 30
```

```
init tokeninit state
goal tokengoal state
plan tokenhistory 1
history 2history 3
action prompthistory 4
```

```
step 5
```

```
0 5 10 15 20 25 30
```

```
init tokeninit state
goal tokengoal state
plan tokenhistory 1
history 2history 3
history 4history 5
action prompt
```

```
step 6
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
1.
```

```
Figure 5: Information flow of last token in Llama-2-7b-chat-hf.
```

```
smaller for MHSA compared to MLP, indicating
that MHSA layers show higher consistency across
different steps.
```

```
5.2 Attention Extract from Goal and History
In the previous section, we discover that MHSA is
responsible for extracting answers from the context,
but which chunk to extract the answer from is still
unclear. In this section, we decompose the input
into several chunks to identify which chunk MHSA
primarily relies on. We use the information flow
method (Wang et al., 2023a), first calculating the
information flow at the token granularity, and then
taking the average of different tokens within the
same chunk to represent the information flow at
the chunk granularity. This will help us locate the
influence of different chunks on the last token.
```

Experiments We calculate the information flow
between layers. Specifically, for the input, we di-
vide it into different chunks, including init token
(which is "Init:"), init state (which is "<blue on
red>"), target token, target state, six history steps
(For step 1, which is "step 1: pick-up white"), ac-
tion prompt (pick-up or stack on-top-of) and last
token. We calculate the information flowItoken,ℓ
for each token at theℓ−thlayer. The specific
calculation method is as follows:

```
Itoken,ℓ(i, j) =
```

### ∣

### ∣∣

### ∣∣

### ∑

```
hd
```

```
Ahd,l(j, i)⊙
∂L(x)
∂Ahd,ℓ(j, i)
```

### ∣

### ∣∣

### ∣∣

### (7)

```
WhereAhd,lis the attention score of theℓ-th
layer,hdis thehd-th head, andL(x)is the loss
function. Here, we useI(i, j)to represent the
score flowing from token j to token i. Based on
the token information flow, we calculate the chunk
information flow, denoted asIchunk,ℓ:
```

```
Ichunk,ℓ=
```

```
∑k 2
i=k 1
```

```
∑t 2
j=t 1 Itoken,ℓ(i, j)
(k 2 −k 1 + 1)(t 2 −t 1 + 1)
```

### (8)

```
Specifically, we consider the information flow from
the span[k 1 , k2]of a chunkkto the span[t 1 , t2]
of another chunkt. We calculate the average of
information flow from chunkktot. Due to the
causal attention, we only compute the information
flow for the lower triangular matrix. We calculate
the chunk information flow for each prediction step.
Results and Analysis The results are shown in
Figure 5 and Figure 16. The vertical axis repre-
sents the information flow from the chunk to the
last token. The horizontal axis represents the infor-
mation flow at layerℓ. The values inside represent
the scores of information flow. We calculate the in-
formation flow for six decision steps. It shows that:
(1) In steps 1 to 6, the goal states are highlighted
at each step. This indicates that MHSA extracts
information from the goal state, demonstrating that
it mainly relies on goal states. (2) Taking the step
5 as an example, history 3 and history 4 are more
prominent compared to history 1 and history 2. It
reveals that MHSA also mainly relies on recent
history rather than earlier spans of steps.
```

## 6 Internal Representations Encode

## Planning Information

```
Based on the previous sections, we discover that
MHSA directly extracts answers from the context,
but it is still unclear what information is encoded in
internal representations. In this section, we demon-
strate the look-ahead capability of models from
both future decisions existence and history step
causality perspectives. (1) For future decisions ex-
istence, we use the probing method to probe each
layer of the main positions in the context. We find
that the accuracy of the current state information
gradually decreases as the steps progress. We also
find that the middle and upper layers encode fu-
ture decisions with accuracy decreasing as planning
steps increase, proving the look-ahead planning hy-
pothesis (§6.1). (2) For history step causality, we
employ a method that involves setting certain in-
formation keys of MHSA to zero. We find there is
still a probability of generating the correct answer
by relying solely on a single step, but it’s difficult
to support plan for the long-term (§6.2).
```

6.1 Internal Representations Encode Block
States and Future Decisions
In this section, we analyze what information is
encoded in the internal representations within the
information flow and how this information evolves
layer by layer. We examine whether the internal
representations encode two types of information
(Li et al., 2022; Pal et al., 2023): Current Block
StatesandFuture Decisions.Current Block States
refer to the state of the blocks at stept. For exam-
ple, in Figure 1, when following the green path, the
Current Block Stateinitially starts in theSinit. After
executing the first and second steps, the internal
representation of theCurrent Block Statetransi-
tions from theSinittoS 12 .Future Decisionsrefer
to the information about future decisions at stept.
For example, in Figure 1, when following the green
path and executing the first step (blue), the ques-
tion is whether the model’s internal representation
already stores information about future decisions
(red, yellow, blue).

Experiments We probe internal representations
of the initial state, goal state, and steps with
layerℓ ∈ [1, L]. We train linear probes and
nonlinear probes for each chunk and each layer.
A linear probe can be represented aspθ(xℓn) =
softmax(W xℓn). And a nonlinear probe can be de-
scribed aspθ(xℓn) =softmax(W 1 ReLU(W 2 xℓn)).

```
0 4 8 121620 2428
layers
```

```
0.
```

```
0.
```

```
0.
```

```
1.
```

```
weighted f
```

```
state linear probe
```

```
0 4 8 1216 202428
layers
```

```
0.
```

```
0.
```

```
0.
```

```
1.
```

```
weighted f
```

```
state nonlinear probe
```

```
init
goal
```

```
step 2 action
step 2 color
```

```
step 4 action
step 4 color
```

```
step 6 action
step 6 color
```

```
Figure 6: State probe in Llama-2-7b-chat-hf.
```

```
0 4 8 121620 2428
layers
```

```
0.
```

```
0.
```

```
0.
```

```
1.
```

```
weighted f
```

```
state linear probe
```

```
0 4 8 1216 202428
layers
```

```
0.
```

```
0.
```

```
0.
```

```
1.
```

```
weighted f
```

```
state nonlinear probe
```

```
init
goal
```

```
step 2 action
step 2 color
```

```
step 4 action
step 4 color
```

```
step 6 action
step 6 color
```

```
Figure 7: State probe in Vicuna-7b.
```

```
Using the linear probe as an example, we consider
six steps and six colors of blocks. ForCurrent
Block States, the input to the probe is a hidden
layer representationhof the model. The output is
a 12x8 matrix representing probabilities, where 12
denotes the colors of the blocks above and below
each color block, and 8 represents 6 colors plus
sky and table. ForFuture Decisions, the input to
the probe ish. The output is six predicted colors
from steps 1 to 6, we only consider future steps in
our evaluation. We split the training and test sets
in a 4:1 ratio for 400 samples. For the evaluation,
we calculate the weighted F1 accuracy forCurrent
Block Statesand accuracy forFuture Decisions.
Results and Analysis As shown in Figure 6 and
Figure 7, the horizontal axis represents the layers
probed, while the vertical axis represents the mean
accuracy of the probe test. Different colored lines
represent the probed spans of states and steps. (1)
We observe that as the number of layers increases,
the accuracy of the probe gradually improves. This
indicates that the early layers of the model are en-
riching the representation of the current state. (2)
The black line (step 6) in the figure has a lower
accuracy compared to the light blue line (step 2),
demonstrating that as the planning steps progress,
the models are difficult to maintain the represen-
```

tations of the current placement of the blocks. (3)
By comparing the linear probe in Figure 6 and the
nonlinear probe in Figure 7, we find that both have
the same trend, indicating that the model internally
stores the current state in a linear manner. A simi-
lar trend inFuture Decisionsis shown in Figure 8
and Figure 9 for actions. It reveals that look-ahead
decisions are enriched in the early layers.

```
0 4 8 12 1620 2428
layers
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
accuracy
```

```
future action linear probe
```

```
0 4 8 12 1620 2428
layers
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
accuracy
```

```
future action nonlinear probe
```

```
init
goal
```

```
step 1 action
step 1 color
```

```
step 2 action
step 2 color
```

```
step 3 action
step 3 color
```

```
Figure 8: Action probe in Llama-2-7b-chat-hf.
```

```
0 4 8 12 1620 2428
layers
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
accuracy
```

```
future action linear probe
```

```
0 4 8 12 1620 2428
layers
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
accuracy
```

```
future action nonlinear probe
```

```
init
goal
```

```
step 1 action
step 1 color
```

```
step 2 action
step 2 color
```

```
step 3 action
step 3 color
```

```
Figure 9: Action probe in Vicuna-7b.
```

Supplementary Analysis As shown in Figure 10
and Figure 12. They illustrate the accuracy of fu-
ture decisions based on the current step. Each col-
umn represents the current step, while the rows
represent the max accuracy of the probe in pre-
dicting future answers. We observe the following
: (1) For the sixth row and first column, the probe
can predict the future sixth step with an accuracy
of 0.51 at the first step. This indicates that the
model stores information about future decisions in
advance, supporting the hypothesis of forward plan-
ning. (2) For each row, the values increase from
left to right. For example, the accuracy in the fifth
column of the sixth row is higher than that in the
first column. This means the model is more certain
about the output of the sixth step at the fifth step
compared to the first step, demonstrating that the
model has difficulty in planning over long distances.

```
(3) The accuracy for the first column, representing
the prediction accuracy for the next five steps after
the initial step, shows a declining trend, indicating
that the model stores future decision information in
advance, supporting the hypothesis of look-ahead
planning decisions existence hypothesis.
```

```
6.2 Internal Representations Facilitate Future
Decision-making
In this section, we further verify the causal effect
of planning information at different steps. We test
the causality between planning information in the
previous historytaand decisions in steptb, where
ta < tb. Specifically, we compare whether the
information from stept 1 contributes to the planning
in stept 2. If the model is greedy in its planning,
there should be no decision information intathat
can help make better decisions intb. Therefore, we
set the key of MHSA in historical decisiontto 0
to study the causal effect of historical information
on future predictions.
```

```
Experiments For each stept, we have a history
Ht = [a 1 , a 2 , ..., at− 1 ], where each step spanai
contains color tokens.
(1) Mask all steps: First, identify all color to-
kens inHt, and set the keys to 0 for these colors
in each layer of MHSA, resulting in the masked
historical informationHt′. The main goal is to stop
past decision information from affecting the cur-
rent decision of the last token. Obtain the decision
probabilityy′tbased onHt′intstep.
(2) Make one step visible: Based onHt′, make
only the color at positionivisible, while masking
the other positions, resulting inHt,i′′. UseHt,i′′ for
prediction, Obtain the decision probabilityyt,i′′.
(3) Calculate one step effect: Compare the deci-
sion probabilities obtained from masking all steps
and from making one step visible to calculate the
effect of a single step. The larger this value, the
greater the impact of stepion stept:
```

```
Impacti,t=yt,i′′−y′t (9)
```

```
Results and Analysis As illustrated in the Figure
11 and Figure 15, the columns represent the steps
visible during prediction, the rows represent the
steps being predicted, and the values inside repre-
sent the contribution of steptto stepi. (1) For
example, in the second column of the sixth row, the
model can increase the probability of inferring the
correct decision in the sixth step by 0.24 just by
```

```
1 2 3 4 5 6
current step
```

```
1 2 3 4 5 6
```

```
probe future step
```

```
0.0 0.0 0.0 0.0 0.0 0.
0.85 0.0 0.0 0.0 0.0 0.
0.950.81 0.0 0.0 0.0 0.
0.7 0.780.81 0.0 0.0 0.
0.720.550.890.71 0.0 0.
0.510.520.690.64 0.8 0.
```

```
future action linear probe
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
1.
```

```
Figure 10: Future action linear probe in Llama-2-7b-
chat-hf.
```

```
using the information from the second step. This
indicates that the model is not greedy and is not
limited to only preparing for the next step, which
causally proves the conclusion of look-ahead plan-
ning. (2) Observing the values in each column,
for instance, the maximum value in the fifth row
is 0.46, located in the third column. This repre-
sents that the third step is the most important for
predicting the fifth step. It is found that the most
important steps for prediction tend to be later steps,
indicating that the look-ahead planning ability of
LLMs is still relatively preliminary.
```

## 7 Related work

LLM-Based Agents With the emergence of large
language models, researchers begin to use them
as intelligent agents (Xi et al., 2023; Wang et al.,
2024a). Significantly, ReAct (Yao et al., 2022)
innovatively combines CoT reasoning with agent
actions. Some tasks utilize the planning capabili-
ties of large language models through prompt en-
gineering methods (Huang et al., 2022; Hao et al.,
2023; Yao et al., 2024; Zhang et al., 2024). Other
researchers enhance the planning capabilities of
large language models through training methods
(Zeng et al., 2023; Chen et al., 2023; Wang et al.,
2023b; Yu et al., 2024). Some researchers con-
struct benchmarks to evaluate the planning ability
of large language models (Shridhar et al., 2020;
Wang et al., 2022; Zhou et al., 2023; Deng et al.,
2024; Xu et al., 2023; Qin et al., 2023).

```
Mechanistic Interpretability Recent works
study mechanistic interpretability in factual asso-
ciations, in-context Learning, and arithmetic rea-
```

```
1 2 3 4 5 6
visible step
```

```
1 2 3 4 5 6
```

```
predict step
```

```
0.
0.7 0.
0.260.190.
0.240.110.460.
0.220.240.140.220.
```

```
single step intervened analysis
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
Figure 11: Single step intervened analysis in Vicuna-7b.
```

```
soning tasks from the perspective of information
flow (Geva et al., 2023; Wang et al., 2023a; Stolfo
et al., 2023; Jin et al., 2024; Yuan et al., 2024).
Researchers also study Othello (Li et al., 2022;
Nanda et al., 2023), chess (Karvonen, 2024) and
Blocksword (Wang et al., 2024b) in transformer.
However, research on the mechanistic interpreta-
tion of LLMs performing planning tasks is still
unexplored. Our work conducts a study from the
perspective of information flow and representation.
Look-Ahead Pal et al. (2023); Wu et al. (2024a);
Jenner et al. (2024) demonstrate that it is possible
to decode future tokens from the hidden represen-
tations at previous token positions. In task plan-
ning, a model needs to have look-ahead capabilities.
However, it is not clear whether LLMs use similar
mechanisms when planning. Our work focuses on
the look-ahead mechanisms in planning in LLMs.
```

## 8 Conclusion

```
In this paper, we investigate the mechanisms of
look-ahead planning in LLMs through the perspec-
tives of information flow and internal representa-
tions. We demonstrateLook-Ahead Planning De-
cisions Existence Hypothesis. Our findings indi-
cate that internal representations of LLMs encode
a few short-term future decisions to some extent
when planning is successful. These look-ahead
decisions are enriched in the early layers, with
their accuracy diminishing as the number of plan-
ning steps increases. We demonstrate that MHSA
mainly extracts information from the spans of goal
states and recent steps. Additionally, the output of
MHSA in the middle layers at the final token can
partially decode the correct decisions.
```

## Limitation

Although our work provides an in-depth analysis
and explanation of look-ahead planning mecha-
nisms of large language models, there are several
limitations. First, our analytical methods require ac-
cess to the internal parameters and representations
of open-source models. Although black-box large
language models such as ChatGPT possess strong
planning capabilities, we cannot access their inter-
nal parameters, making it challenging to interpret
the most advanced language models. Second, our
research primarily focuses on the planning mecha-
nisms in Blocksworld. However, many other plan-
ning tasks, such as commonsense planning (e.g.,
"how to make a meal"), lack standard answers,
making it difficult to evaluate the correctness of
the planning and conduct quantitative analysis. We
leave these limitations for future work.

## A Additional Results

Future action nonlinear probe in Llama-2-7b-chat-
hf is shown in Figure 12. Future action linear probe
in Vicuna-7b is shown in Figure 13. Future action
nonlinear probe in Vicuna-7b is shown in Figure 14,
Single step intervened analysis in Llama-2-7b-chat-
hf is shown in Figure 15. Data statistics is shown in
Table 1. Information flow of last token in Vicuna-
7b is shown in Figure 16.

```
1 2 3 4 5 6
current step
```

```
1 2 3 4 5 6
```

```
probe future step
```

```
0.0 0.0 0.0 0.0 0.0 0.
0.85 0.0 0.0 0.0 0.0 0.
0.95 0.8 0.0 0.0 0.0 0.
0.720.750.81 0.0 0.0 0.
0.720.56 0.9 0.72 0.0 0.
0.5 0.570.680.65 0.8 0.
```

```
future action nonlinear probe
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
1.
```

Figure 12: Future action nonlinear probe in Llama-2-
7b-chat-hf.

```
1 2 3 4 5 6
current step
```

```
1 2 3 4 5 6
```

```
probe future step
```

```
0.0 0.0 0.0 0.0 0.0 0.
0.85 0.0 0.0 0.0 0.0 0.
0.940.77 0.0 0.0 0.0 0.
0.770.780.75 0.0 0.0 0.
0.8 0.8 0.890.77 0.0 0.
0.570.380.670.630.85 0.
```

```
future action linear probe
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
1.
```

```
Figure 13: Future action linear probe in Vicuna-7b
```

```
1 2 3 4 5 6
current step
```

```
1 2 3 4 5 6
```

```
probe future step
```

```
0.0 0.0 0.0 0.0 0.0 0.
0.85 0.0 0.0 0.0 0.0 0.
0.94 0.8 0.0 0.0 0.0 0.
0.770.760.76 0.0 0.0 0.
0.810.810.890.75 0.0 0.
0.580.390.670.620.84 0.
```

```
future action nonlinear probe
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
1.
```

```
Figure 14: Future action nonlinear probe in Vicuna-7b
```

```
1 2 3 4 5 6
visible step
```

```
1 2 3 4 5 6
```

```
predict step
```

```
0.
0.7 0.
0.280.160.
0.160.040.380.
0.140.04-0.00.040.
```

```
single step intervened analysis
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
Figure 15: Single step intervened analysis in Llama-2-
7b-chat-hf.
```

```
LEVEL L1 L2 L3 Total
Train Size
4 blocks 3 17 25 45
5 blocks 1 23 121 145
6 blocks 3 48 326 377
Total 7 88 472 567
Test Size
4 blocks 24 60 80 164
5 blocks 34 115 268 417
6 blocks 57 232 709 998
Total 115 407 1057 1579
```

```
Table 1: Blocksworld dataset statistics
```

```
0 5 10 15 20 25 30
```

```
init tokeninit state
goal tokengoal state
```

action promptplan token

```
step 1
```

```
0 5 10 15 20 25 30
```

```
init tokeninit state
goal tokengoal state
plan tokenhistory 1
action prompt
```

```
step 2
```

```
0 5 10 15 20 25 30
```

```
init tokeninit state
goal tokengoal state
plan tokenhistory 1
```

action prompthistory 2

```
step 3
```

```
0 5 10 15 20 25 30
```

```
init tokeninit state
goal tokengoal state
plan tokenhistory 1
history 2history 3
action prompt
```

```
step 4
```

```
0 5 10 15 20 25 30
```

```
init tokeninit state
goal tokengoal state
plan tokenhistory 1
history 2history 3
```

action prompthistory 4

```
step 5
```

```
0 5 10 15 20 25 30
```

```
init tokeninit state
goal tokengoal state
plan tokenhistory 1
history 2history 3
history 4history 5
action prompt
```

```
step 6
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
1.
```

```
Figure 16: Information flow of last token in Vicuna-7b.
```

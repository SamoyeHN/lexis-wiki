# Dual-Tier Neuro-Symbolic WSD 架构总结

## 核心原则

> 符号层负责判断：
>
> **这个义项能不能出现（Legality）**
>
> MiniLM 负责判断：
>
> **所有合法义项中哪个最自然（Naturalness）**

而不是让 MiniLM 直接在全部义项中自由竞争。

---

# 总体架构

```text
输入句子 + 目标词
        │
        ▼
┌────────────────────────────┐
│ Tier 1                     │
│ Symbolic Gate              │
│ 符号门禁层                 │
└────────────────────────────┘
        │
        ▼
  Legal Candidate Pool
      合法候选池
        │
        ▼
┌────────────────────────────┐
│ Tier 2                     │
│ MiniLM Ranking             │
│ 神经语义排序层             │
└────────────────────────────┘
        │
        ▼
      Locked Sense
      最终义项
```

---

# Tier 1：Symbolic Gate（符号门禁）

目标：

```text
快速剔除明显错误义项
```

特点：

```text
高精度
高可解释性
低计算成本
```

---

## 1. POS Gate（词性门禁）

### 规则

句法分析确定词性后：

```text
VERB → 只保留动词义项

NOUN → 只保留名词义项
```

### 示例

```text
record
```

句中：

```text
They record everything.
```

识别为：

```text
VERB
```

则：

```text
record (n.)
```

直接移除。

---

## 2. Passive Projection（被动语态投影）

### 目的

恢复真实论元结构。

### 输入

```text
The city was seized by the army.
```

依存分析：

```text
nsubjpass(city)
agent(army)
```

### 投影结果

```text
army seize city
```

即：

```text
Agent  = army
Patient = city
```

### 作用

让配价规则继续有效。

---

## 3. Sense-Level Valency（义项级配价）

### 核心思想

不是：

```text
Word → Frame
```

而是：

```text
Sense → Frame
```

---

### 示例

词典：

```text
run#1
move quickly

Frame:
SV
```

```text
run#2
manage

Frame:
SVO
```

---

句子：

```text
She runs a company.
```

抽取：

```text
SVO
```

匹配：

```text
run#1 ×

run#2 ✓
```

---

### 推荐结构

```json
{
  "sense_id": "run#2",
  "frames": [
    "SVO"
  ]
}
```

---

### 配价优先级极高

对于学习词典：

```text
★★★★★
```

很多情况下：

```text
配价已经足以锁定义项
```

甚至不需要 MiniLM。

---

## 4. Selection Restriction（选择限制）

### 目的

检查论元类型是否合理。

---

### 示例

```text
The army seized the city.
```

抽象：

```text
Agent  = ORGANIZATION
Patient = LOCATION
```

匹配：

```text
Physical Capture Sense
```

---

### 再看

```text
She seized the opportunity.
```

抽象：

```text
Patient = ABSTRACT
```

匹配：

```text
Take Advantage Sense
```

---

### 推荐做法

不要硬熔断：

```text
-∞
```

采用强惩罚：

```text
-30
-50
```

原因：

```text
隐喻
转喻
创造性表达
```

普遍存在。

---

例如：

```text
His words seized the audience.
```

这里：

```text
words ≠ animate
```

但表达完全合法。

---

## 5. Domain Gate（领域约束）

### 示例

词典：

```text
arrest
```

义项：

```text
法律义
医学义
```

---

如果没有：

```text
heart
cardiac
patient
hospital
```

则：

```text
医学义降权
```

而非直接删除。

---

# Tier 1 门禁策略

## Hard Gate（硬门禁）

适用于：

```text
POS
Valency
Passive Projection
```

因为属于：

```text
句法事实
```

错误率低。

---

## Soft Gate（软门禁）

适用于：

```text
Selection Restriction
Domain
Collocation
```

采用：

```text
Penalty
```

而不是：

```text
Kill
```

避免 Recall 崩溃。

---

# Tier 2：MiniLM Ranking

## MiniLM 的职责

不是：

```text
哪个义项合法
```

而是：

```text
合法义项中哪个最自然
```

---

## 工作流程

假设经过 Tier 1 后：

```text
run#1
manage a business

run#2
direct an activity

run#3
operate an organization
```

都合法。

---

句子：

```text
He runs the company.
```

MiniLM 分别计算：

```text
Context
↔
run#1 gloss

Context
↔
run#2 gloss

Context
↔
run#3 gloss
```

---

结果：

```text
run#1 0.92

run#2 0.84

run#3 0.79
```

最终：

```text
run#1
```

胜出。

---

# MiniLM 最擅长什么

## 近义词泛化

```text
company
business
enterprise
firm
```

---

## 场景联想

```text
manage
administer
direct
control
```

---

## 语义补全

即使词典写：

```text
business
```

而句子出现：

```text
startup
```

MiniLM 仍能理解：

```text
startup ≈ business
```

---

# 推荐：Predicate-Centered Encoding

不要直接编码整句。

---

例如：

```text
The army seized the bridge after a long battle.
```

直接编码：

```text
[整句]
```

噪声很多。

---

推荐抽取：

```text
army seize bridge
```

或者：

```text
Agent: army
Verb: seize
Patient: bridge
```

然后编码。

---

这样：

```text
关注论元结构
而不是段落噪声
```

---

# 推荐最终评分公式

```text
FinalScore
=
ValencyScore
+
SelectionScore
+
DomainScore
+
CollocationScore
+
α × MiniLMScore
```

---

示例：

```text
run#1

Valency      50
Selection    30
Domain       10
MiniLM        8

Total        98
```

---

```text
run#3

Valency      50
Selection     5
Domain        0
MiniLM        6

Total        61
```

---

最终：

```text
run#1
```

---

# 模块优先级

```text
1. Sense-Level Valency
★★★★★

2. Passive Projection
★★★★★

3. Selection Restriction
★★★★☆

4. Domain Label
★★★☆☆

5. MiniLM Ranking
★★★☆☆
```

---

# 最终定位

```text
配价（Valency）
    ↓
决定能否进入候选池

选择限制（Selection Restriction）
    ↓
决定是否合理

MiniLM
    ↓
在合理候选中做语义排序
```

---

# 一句话总结

> 先用配价、语态恢复和选择限制构建「合法候选池（Legal Candidate Pool）」，再让 MiniLM 在候选池内部做精细排序。符号层负责防止神经网络违反句法契约，MiniLM 负责提供语义泛化能力。这是细粒度 Learner's Dictionary WSD 最稳健、最可解释的 Neuro-Symbolic 路线。
# 📖 Lexis Wiki

<p align="center">
  <b>英语课文全自动精读与自批改互动刷题神器</b><br>
  <i>Transform textbook units, articles, audio, and video into Obsidian-style wiki study notes and self-grading interactive HTML quizzes.</i>
</p>

<p align="center">
  <a href="README.md">English</a> •
  <a href="README.zh-CN.md"><b>简体中文</b></a>
</p>

---

## 💡 什么是 Lexis Wiki？

无论你手里是一篇英语课文、学术讲义、新闻报道，还是一段没有字幕的英文视频/音频：

👉 **只要把原文丢进去，30 秒全自动生成：**
1. **百科级精读笔记**：包含核心生词、地道搭配、短语动词、语法难点与交互式思维导图，支持 Obsidian 风格双向链接 `[[wikilinks]]`。
2. **6 种开箱即用的互动测试卷**：词汇单选、纯开放式语篇完形、阅读理解、对比翻译、双人磨耳朵听力、视频字幕题。
3. **单文件直接答题**：导出的试卷是**独立的单个 `.html` 文件**。双击即可在浏览器中答题，自带倒计时、自动批改评分、错题解析和 AI 原生语音朗读，无需安装任何软件即可直接发给学生或自己刷题！

---

## 🥊 为什么它比“直接问 ChatGPT”更靠谱？

直接使用 ChatGPT 或普通 AI 出题，经常会遇到三个致命痛点：**生词乱挑、例句瞎编、选项冲突（存在多个正确答案或语法破绽）**。

Lexis Wiki 采用独创的**「神经-符号双引擎」**架构，将学术级权威语料库与现代大语言模型完美结合：

| 维度 | 普通 AI 自由生成 (Pure Prompting) | Lexis Wiki (神经-符号架构) |
| :--- | :--- | :--- |
| **生词与短语挖掘** | 随机挑词，容易漏掉核心搭配与短语动词 | 结合上下文句法树与权威大纲，自动识别整块搭配与习语 |
| **例句真实度** | 大模型自由臆造，经常出现不自然或中式英语 | **100% 锁定权威词典原版例句**，绝不胡编乱造 |
| **多义词匹配 (WSD)** | 容易张冠李戴，给出与课文上下文不符的词义 | 词性、句法槽位与语义特征硬门禁 + 智能仲裁，精准锁定当前词义 |
| **测试题干扰项** | 选项词形不对称、语法有破绽，或多个选项都说得通 | 词形屈折统一对齐、同义碰撞自动排查，确保**单空唯一确定解** |
| **难度分级控制** | 大模型凭“感觉”出题，难度忽高忽低 | 内置 CEFR 语言难度硬门禁，精确卡控 A1–C2 阶梯难度 |
| **交付形式** | 纯文本 / Markdown，需要人工复制排版 | **一键生成独立交互式 HTML 试卷**，自批改、带发音、适配手机与电脑 |

---

## ✨ 核心功能一览

### 1. 📚 课文全自动精读笔记
- **核心生词与地道短语**：自动提取单词与高频搭配（如 `phrasal verb`、`collocation`、`set phrase`、`idiom`），标注 CEFR 等级，并直接绑定权威词典原声例句。
- **句型结构与语法点**：基于 COBUILD 语法体系，梳理修辞强调、篇章衔接、信息包装等核心结构，指出常见搭配陷阱。
- **篇章大意与交互脑图**：自动梳理文章逻辑脉络，生成可视化层级思维导图（支持拖拽与缩放）。

### 2. 📝 6 大开箱即用的互动测试卷
- 🔤 **词汇与搭配单选 (Vocabulary Quiz)**：基于课文锁定的核心义项生成高质量语境填空，选项词形严格平行，零干扰项冲突。
- 🧩 **纯开放式语篇完形 (Pure Open Cloze)**：零 Token 离线运行，完全脱离生词表与四选一干扰项，均衡考查语篇衔接、固定搭配与词汇语法，支持 `accepted_answers` 多合法答案白名单判分。
- 📖 **篇章阅读理解 (Reading Quiz)**：涵盖主旨归纳、细节事实与深度推断题，紧密贴合四六级/考研/雅思/托福等标准化考试标准。
- ✍️ **对比翻译测评 (Translation Quiz)**：精选课文典型句，剖析优质译文与常见中式英语错误对比。
- 🎧 **双角色磨耳朵听力 (Listening Quiz)**：基于课文主题自动编排多轮日常交际对话，内置 **Kokoro** / **Edge-TTS** 双发音人高清语音朗读。
- 🎬 **本地音视频测验 (Video Quiz)**：拖入本地音视频，借助 **faster-whisper** 自动提取双语字幕并精准锚定时间戳，边看视频边刷题。

### 3. 📦 完全独立的单文件导出
- 导出的所有试卷均为**独立单个 HTML 文件**，内置样式、逻辑与离线题库。
- 学生在手机、平板或电脑上**直接用浏览器双击打开即可使用**；也可以通过内置命令一键打包成单文件 `.exe`。

---

## ⚡ 3 分钟快速上手

### 选项 A：Windows 用户（推荐，一键安装）

1. **下载或克隆** 本仓库到本地；
2. **双击运行 `install.cmd`**：自动创建独立虚拟环境并安装核心组件；
3. **双击运行 `start_dashboard.vbs`**（或在命令行输入 `lexis dashboard`）：启动可视化网页控制台；
4. **配置大模型连接**：
   - 点击网页右上角 **设置 (Settings)** 图标；
   - 支持本地免费 **Ollama**（默认 `http://localhost:11434`）或任意 **OpenAI 兼容接口**（如 DeepSeek、通义千问、Kimi 等），填入地址与模型名称即可；
5. **开始使用**：
   - 将文本/音视频文件放入单元，点击 **「Compile Extractions」** 生成笔记，再点击 **「Generate Quizzes」** 生成试卷！

---

### 选项 B：Linux / macOS / WSL 用户

```bash
# 1. 克隆代码库
git clone https://github.com/SamoyeHN/lexis-wiki.git
cd lexis-wiki

# 2. 一键运行自动化安装脚本
chmod +x install.sh
./install.sh

# 3. 启动网页控制台
lexis dashboard
```

脚本会自动检测 Python 环境，创建专用 `.venv`，下载词法索引并初始化工作目录。

---

### 选项 C：开发者手动安装

```bash
# 1. 安装核心包
git clone https://github.com/SamoyeHN/lexis-wiki.git
cd lexis-wiki
pip install -e .

# 2. （可选）安装视频字幕与语音转写支持
pip install -e ".[video]"

# 3. 初始化词库并启动服务
lexis init .
lexis dashboard
```

---

## 🖥️ 命令行快速操作 (CLI)

如果你更喜欢在终端中使用命令行，可以直接使用 `lexis` 命令完成所有操作：

```bash
# 启动可视化控制台
lexis dashboard

# 为指定单元编译精读笔记（词汇、短语、语法、思维导图）
lexis compile Book_2_Unit_1_Section_A

# 生成指定类型与题量的交互式试卷
lexis quiz Book_2_Unit_1_Section_A --template vocabulary --count 15
lexis quiz Book_2_Unit_1_Section_A --template reading --count 10

# 导入本地音视频文件并自动使用 Whisper 提取字幕
lexis video-import "lecture.mp4" --name Unit_Video_1

# 将 HTML 试卷打包为独立的单文件 EXE 程序
lexis export-exe wiki/Unit_Video_1/handouts/Unit_Video_1_video_quiz.html

# 检查知识库内部双链 [[wikilinks]] 完整性
lexis lint
```

---

## 📂 项目文件组织结构

每个学习单元（Unit）采用独立自包含目录设计，文件结构清晰一目了然：

```text
wiki/<单元名称>/
├── sources/                              # 原始输入素材
│   ├── <单元名称>.md                     # 课文正文 + 可选大纲词表
│   └── media/                            # 本地音视频及配套字幕文件
├── extractions/                          # 全自动提取的精读知识笔记
│   ├── <单元名称>_vocabulary.md          # 核心单词、短语、义项、音标与权威例句
│   ├── <单元名称>_grammar.md             # 句型公式、语法功能与易错点提示
│   ├── <单元名称>_summary.md             # 课文结构化摘要与核心观点
│   └── <单元名称>_mindmap.json           # 可视化思维导图数据
└── handouts/                             # 导出的独立交互式 HTML 试卷
    ├── <单元名称>_vocabulary_quiz.html   # 词汇与搭配单选题
    ├── <单元名称>_cloze_quiz.html        # 纯开放式语篇完形题
    ├── <单元名称>_reading_quiz.html      # 篇章深度阅读理解题
    ├── <单元名称>_translation_quiz.html  # 对比翻译纠错题
    ├── <单元名称>_listening_quiz.html    # 智能双角色听力对话题
    └── <单元名称>_video_quiz.html        # 音视频时间戳互动测试题
```

---

## ⚙️ 模型与运行配置建议

配置文件为根目录下的 `wiki_config.json`，在 Web 界面中修改后会自动热重载，无需重启服务：

| 参数项 | 类型 | 说明 | 推荐默认值 / 示例 |
| :--- | :--- | :--- | :--- |
| `api_type` | `str` | 模型接口类型（`ollama` 或 `openai`） | `"ollama"` |
| `api_url` | `str` | 模型服务接口地址 | `"http://localhost:11434"` |
| `api_key` | `str` | API 密钥（使用第三方 OpenAI 兼容接口时填写） | `""` 或 `"sk-..."` |
| `model` | `str` | 核心出题与词义仲裁主模型 | `"gemma4:12b"` / `"qwen2.5:7b"` |
| `enable_expert_audit` | `bool` | 是否开启 Level 2 专家模型第二道答题质检 | `false` |
| `judge_model` | `str` | 独立把关审查模型（开启质检时生效） | `"gemma4:27b"` / `"mistral-small3.2:24b"` |
| `tts_engine` | `str` | 听力对话语音合成引擎（`kokoro` 或 `edge`） | `"edge"` |

### 💡 电脑显卡与算力推荐
- **普通游戏本 / 8GB–12GB 显存（如 RTX 3060 / 4060 / 4070）**：
  - 推荐使用 7B–12B 规格模型（如 `qwen2.5:7b`、`llama3.1:8b`、`gemma4:12b`）；
  - 并发数设为 `1`，出题流畅且响应迅速。
- **高性能主机 / 24GB+ 显存（如 RTX 3090 / 4090）**：
  - 可同时开启 27B–32B 专家模型双重审查，进一步提升试卷题目的严谨度。
- **无独立显卡 / 轻薄本**：
  - 直接在设置里填写云端 API（如 DeepSeek、通义千问等开放平台 Key），纯走云端调用，不消耗本地显存与 CPU。

---

## 🔬 深入技术内幕（开发者与语言学研究者）

<details>
<summary><b>点击展开：了解背后的神经-符号架构与学术理论</b></summary>

Lexis Wiki 的底层并非简单的提示词工程，而是一套严谨的计算语言学与第二语言习得（SLA）架构：

1. **符号确定性挖掘与消歧 (Sense Locking & WSD)**：
   - 结合 `spaCy` 依存句法分析与离线词法数据库（`pedlex.db`，涵盖 189,000+ `ldoce_phrase_index` 词伙短语索引）；
   - 单义词（占比约 71%）通过词性与语法槽直接秒级锁定（0ms、零 Token 消耗）；
   - 多义词（占比约 29%）生成 $K \le 4$ 的精简候选池，由本地小模型进行 1-Token 单选仲裁，彻底杜绝词义漂移。
2. **五源权威例句池 (Five-Source Authentic Corpus Pool)**：
   - 严格在朗文五大语料仓（语法核心框、义项原生例句束、Thesaurus 同义词典、Activator 语言激活器、精选搭配库）中进行槽位适配筛选，确保例句完全源自真实语料。
3. **两级质检审计门禁 (Two-Level Quality Gate)**：
   - **Level 1 确定性代码硬门禁（<5ms）**：验证题目物理不变量，包括单空唯一性约束、词形平行对齐、CEFR 词汇难度上限、选项 Fisher-Yates 随机置换与双答案同义碰撞过滤；
   - **Level 2 专家模型语义盲测（可选）**：通过独立模型扮演盲测考生解答并评分，自动针对瑕疵选项执行外科手术式微调（`PASS` / `REPAIR` / `REWRITE`）。
4. **深入技术文档索引**：
   - [词汇选择与干扰项设计方法论](docs/vocabulary-selection-and-distractor-methodology.md)
   - [多词搭配单元切分与干扰项架构](docs/multi-word-units-slicing-and-distractor-architecture.md)
   - [PedLex 数据库设计与查询规范](docs/pedlex-database-architecture.md)
   - [听力交际对话场景与会话分析架构](docs/listening-functional-dialogue-architecture.md)
   - [阅读理解语篇关系与心理测量干扰项模型](docs/reading-discourse-architecture.md)
   - [纯开放式语篇完形架构规范](docs/cloze-discourse-architecture.md)
   - [质检评分标准与审核指南](docs/audit-scores-guide.md)
</details>

---

## 📄 开源许可证

本项目基于 **MIT License** 开源。详见 [LICENSE](LICENSE) 文件。

```text
Copyright (c) 2026 Lexis Wiki Contributors
```

# Common Educational Dialogue Characters (通用教学对话人物库)

本目录为 Lexis Wiki 教学视频与对话工作台专用的数字人人物资产库，基于 Wan2.1 480P fp8 + InfiniteTalk 驱动管线标准构建。

---

## 1. 角色阵容一览 (Cast Overview)

| 编号 | 标识符 | 角色名称与定位 | 年龄 | 外观与服饰风格 | 建议教学应用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **01** | `Character_01_Miss_Green` | **Miss Green** (知性女教师 / 职场女性) | 28–30 | 深蓝西装外套，内搭白色立领衬衫，齐肩微卷发，神态知性温和 | 教学主讲、语法拆解、商务问候、车站/机场问询 |
| **02** | `Character_02_John` | **John** (阳光男青年 / 大学生) | 20–22 | 深蓝圆领羊毛衫，内搭浅蓝牛津纺衬衫领，短发阳光诚恳 | 学生对话、失物招领（认领男士）、同学寒暄、借物 |
| **03** | `Character_03_Alice` | **Alice** (自然女青年 / 大学生) | 19–21 | 浅米色针织开衫，内搭浅色衬衫，低马尾，自然文静 | 同学结伴、日常交流、初次相识、图书馆借阅 |
| **04** | `Character_04_Mr_Blake` | **Mr. Blake** (成熟绅士 / 职员) | 40–45 | 经典细框眼镜，深灰粗花呢马甲与西服，白衬衫深色领带，稳重大方 | 衣帽间职员（大衣/雨伞）、机构主管、资深学者、店长 |

---

## 2. 几何与数字人标准校验 (Face Invariants)

所有角色均通过 `skills/digital-human/scripts/face_check.py` 自动化检测：
- 严格单人头像 (`faces == 1`)
- 双眼水平倾角 $|\theta| \le 2.0^\circ$（杜绝侧头导致的嘴部抖动）
- 偏航角 yaw 趋近正对 ($|\text{yaw}| \le 3.5^\circ$)
- 脸部宽度占比 `face_w` 处于 $0.14 \sim 0.22$ 区间（适合 16:9 构图）
- 嘴唇处于闭合静止状态（防止与初始发音嘴型冲突）
- 纯正教育级着装（无任何不适宜夸张或二次元风格）

### 检测数值记录：
```json
{"file": "Character_01_Miss_Green_896x512.png", "faces": 1, "eye_line_deg": -1.4, "roll": -1.3, "yaw": -0.3, "pitch": -6.0, "face_w": 0.146}
{"file": "Character_02_John_896x512.png", "faces": 1, "eye_line_deg": 0.0, "roll": 0.6, "yaw": 2.9, "pitch": -3.9, "face_w": 0.212}
{"file": "Character_03_Alice_896x512.png", "faces": 1, "eye_line_deg": -1.8, "roll": -1.3, "yaw": 0.9, "pitch": -6.5, "face_w": 0.173}
{"file": "Character_04_Mr_Blake_896x512.png", "faces": 1, "eye_line_deg": -2.0, "roll": -1.3, "yaw": -3.4, "pitch": -3.4, "face_w": 0.173}
```

---

## 3. 文件命名规范与用途

1. **高清母版图 (1376x768)**:
   - `Character_01_Miss_Green.png`
   - `Character_02_John.png`
   - `Character_03_Alice.png`
   - `Character_04_Mr_Blake.png`
2. **数字人标准视频底稿图 (896x512)**:
   - `Character_01_Miss_Green_896x512.png`
   - `Character_02_John_896x512.png`
   - `Character_03_Alice_896x512.png`
   - `Character_04_Mr_Blake_896x512.png`
3. **总览对比网格 (Contact Sheet)**:
   - `Characters_Dialogue_Cast_Grid.png`

---

## 4. 驱动视频生成命令 (Workflow Integration)

在为指定单元生成对话视频时，只需执行以下标准管线：

```bash
# 1. 由静态图生成无 AI 漂移的正弦微呼吸视频底稿
python /tmp/builder2-digital-human/skills/digital-human/scripts/still_source.py \
    --input librarian/data/characters/Character_02_John_896x512.png \
    --output /tmp/breathing_john.mp4 \
    --duration <AUDIO_DURATION_SECONDS>

# 2. 调用 20 服务器 Wan2.1 + InfiniteTalk (从第2步开始扩散嘴型)
python /tmp/builder2-digital-human/skills/digital-human/scripts/infinitetalk_run.py \
    --server 192.168.1.20:8188 \
    --video /tmp/breathing_john.mp4 \
    --audio <AUDIO_FILE.wav> \
    --output wiki/<Unit>/sources/media/<Output_Video.mp4>
```

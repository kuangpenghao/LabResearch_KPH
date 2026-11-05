from pathlib import Path
import json, re, shutil, hashlib

ROOT = Path(r'D:\PenghaoKuang\LabResearch\ResearchRecords\literature\PaperReading')
OUT = Path(r'C:\Users\26532\Documents\Codex\2026-10-29\read-paper-deeply-c-users-26532\outputs')
changes = []

def diagram(title, code, caption):
    return f'**{title}**\n\n```mermaid\n{code.strip()}\n```\n\n{caption}\n\n'

def add(name, heading_prefix, title, code, caption):
    changes.append((name, heading_prefix, diagram(title, code, caption)))

add('MLP_Memory', '### 2.3', '监督构造：先搜索，再把邻居变成词表分布', '''
flowchart TD
    A["训练语料的各个前缀"] --> B["语言模型编码上下文"]
    B --> C["数据库：上下文向量与真实后继 token"]
    B --> D["当前位置的查询向量"]
    C --> E["近邻检索：排除当前位置自身"]
    D --> E
    E --> F["按距离加权，同一后继 token 的权重相加"]
    F --> G["缓存检索教师的词表分布"]
''', '解读图：数据库只用于离线制造监督，后继 token 聚合后才形成 MLP 的学习目标。对应 §2.1–2.2。')

add('MLP_Memory', '### 2.5', '训练：固定输入与教师，只学习 MLP', '''
flowchart TD
    A["主模型上下文向量<br/>缓存读取，或冻结主模型现场计算"] --> B["可训练 MLP 与输出头"]
    B --> C["记忆词表分布"]
    D["固定的检索教师分布"] -.-> E["KL 损失"]
    C --> E
    F["真实下一 token"] -.-> G["交叉熵损失"]
    C --> G
    E --> H["总损失：0.4 × KL + 0.6 × CE"]
    G --> H
    H --> I["只更新 MLP 与独立输出头"]
''', '解读图：此处损失评价的是 MLP 自己的输出；最终概率插值发生在使用阶段。主模型和检索教师均不更新。对应 §2.4。')

add('MLP_Memory', '### 2.6', '推理：两份概率在输出端汇合', '''
flowchart TD
    A["当前已生成的前缀"] --> B["主模型运行到所选中间层"]
    B --> C["继续运行主模型后续层与输出头"]
    B --> D["MLP Memory 与独立输出头"]
    C --> E["主模型 softmax 概率"]
    D --> F["记忆 softmax 概率"]
    E -->|"权重 1 − λ"| G["概率加权求和"]
    F -->|"权重 λ"| G
    G --> H["选择下一 token，接回前缀"]
    H --> A
''', '解读图：λ 按正文式(7)表示记忆权重。记忆分支不写回同一步主模型隐藏状态，推理不访问检索库。对应 §2.5。')

add('LatentMem', '### 2.4', '角色定制：同一段经历，生成不同的连续输入', '''
flowchart TD
    A["当前任务"] --> B["从经验库检索相关合作轨迹"]
    B --> C["共享记忆生成器"]
    D["当前被调用智能体的角色描述"] --> C
    C --> E["该角色的 8 个连续记忆向量"]
    F["当前步骤提示的向量序列"] --> G["提示向量后拼接记忆向量"]
    E --> G
    G --> H["冻结的对应角色智能体"]
    H --> I["本步输出，交给原协作流程"]
''', '解读图：每次调用按当前角色生成记忆；图中“共享”指生成器权重共享，不是各角色共用同一份输出向量。当前任务先用于检索，生成器显式条件按原文式(5)为角色与检索轨迹。对应 §2.2–2.4。')

add('LatentMem', '### 2.6', '训练：奖励来自完整协作，梯度经冻结智能体回到生成器', '''
flowchart TD
    A["旧生成器与冻结智能体<br/>同一任务采样多条合作轨迹"] --> B["已采样的消息与旧 token 概率"]
    A --> C["执行任务，计算每条轨迹的最终奖励"]
    C --> D["组内比较：得到轨迹优势值"]
    E["当前生成器产生角色记忆"] --> F["冻结智能体重算<br/>已采样输出的条件概率"]
    B --> F
    B --> G["新旧概率比 × 优势值<br/>构造裁剪目标 LMPO"]
    F --> G
    D --> G
    G --> H["更新生成器 LoRA<br/>梯度经冻结智能体输入传回"]
''', '解读图：只更新生成器 LoRA；旧概率固定。离散消息采样、程序执行和奖励评判不直接反向传播。对应 §2.5，省略 token 求和与归一化细节。')

add('LatentMem', '### 2.7', '部署：参数固定，新的经验进入外部库', '''
flowchart TD
    A["新任务"] --> B["检索经验库"]
    C["原始合作轨迹库"] --> B
    B --> D["按角色生成连续记忆"]
    D --> E["智能体按原框架协作"]
    E --> F["任务结果与完整新轨迹"]
    F --> G["追加新轨迹"]
    G --> C
''', '解读图：经验库追加会改变下一次可检索的历史；此循环不包含在线 LoRA 更新。生成器和智能体在部署时均固定。对应 §2.6。')

add('Memory3', '### 2.4', '读取：外部记忆与本地上下文共用一次 softmax', '''
flowchart TD
    A["当前 token 的 query"] --> E["计算与所有可见 key 的分数"]
    B["取回的稀疏记忆 K、V"] --> D["分别拼接 K 与 V"]
    C["本地上下文 K、V"] --> D
    D --> E
    E --> F["联合 softmax<br/>在记忆与上下文之间分配权重"]
    F --> G["按权重汇总拼接后的 V"]
    D --> G
    G --> H["原注意力输出投影与后续模型计算"]
''', '解读图：这是选定记忆层内的读取路径，不是独立交叉注意力分支。query、key、value 仍使用模型原有投影。对应 §2.3、原文式(4)。')

add('Memory3', '### 2.5', '写入：一次参考编码得到可复用的稀疏 KV', '''
flowchart TD
    A["不超过 128 token 的参考文本"] --> B["固定模型版本，逐参考独立编码"]
    B --> C["仅取前 22 层、8 个 KV 头"]
    C --> D["每层每头按评分选 8 个位置"]
    D --> E["保留所选 K、V<br/>key 保留原有位置信息"]
    E --> F["可选：向量量化"]
    F --> G["外部记忆库：按参考 ID 保存"]
''', '解读图：不同层、不同头可选不同位置。这里是部署时的预编码写入，不执行参数更新；模型版本变化后需重建相应 KV。对应 §2.2、2.4。')

add('Memory3', '### 2.7', '训练顺序与部署：同一模型承担写入和读取', '''
flowchart TD
    A["无记忆预训练<br/>普通下一 token 预测"] --> B["带记忆继续预训练"]
    C["每个训练块配 1 条参考<br/>共享前 4 块参考"] --> B
    B --> D["用当前模型实时编码参考<br/>接入正文注意力，优化 LM 损失"]
    D --> E["后训练，确定部署模型版本"]
    E --> F["冻结该版本，预编码参考 KV"]
    F --> G["在线按 ID 加载 KV<br/>每生成满 64 token 刷新 5 条参考"]
''', '解读图：带记忆续训更新主模型，所以参考要实时编码；部署后参数固定，才可长期复用预编码 KV。图中 1 条参考及前 4 块共享是训练节省方案，不能当作推理检索策略。对应 §2.5–2.7。')

add('MSA', '### 2.4', '层内路由：先选文档，再读选中文档的内容', '''
flowchart TD
    A["各文档独立编码，按 64 token 均值池化"] --> B["路由 Key"]
    A --> C["内容 K、V"]
    D["问题或当前活动上下文"] --> E["专用路由 Query"]
    B --> F["块级相似度 → 文档最高分"]
    E --> F
    F --> G["选择 Top-16 篇文档"]
    G --> H["加载选中文档的全部压缩内容 KV"]
    C --> H
    H --> I["与本地 KV 拼接，普通注意力读取"]
    D --> I
''', '解读图：路由 Key 负责选择，内容 KV 负责提供信息；Top-16 的单位是文档，不是 16 个压缩块。此通路只在采用 MSA 的后半层启用。对应 §2.1–2.3。')

add('MSA', '### 2.7', '多跳循环：生成文档 ID 后，还要读回原文', '''
flowchart TD
    A["当前问题与已有证据原文"] --> B["通过稀疏记忆注意力继续生成"]
    B --> C{"生成什么？"}
    C -->|"证据文档 ID 与轮次分隔符"| D["按 ID 取回原始文档"]
    D --> E["原文追加到活动上下文"]
    E --> A
    C -->|"结束检索标记"| F["依据积累证据生成最终答案"]
''', '解读图：这是一层生成控制循环，和各层固定 Top-16 的路由不同；每轮输出多少文档 ID 由模型决定。压缩记忆帮助定位，读回原文提供细节。对应 §2.6。')

add('MSA', '### 2.8', '大库存储：GPU 扫描路由，CPU 提供被选中的内容', '''
flowchart TD
    A["压缩后的记忆库"] --> B["路由 Key：分片保存在多张 GPU"]
    A --> C["内容 K、V：保存在 CPU 内存"]
    D["当前路由 Query"] --> E["各 GPU 扫描自己的分片"]
    B --> E
    E --> F["合并候选，得到全局 Top-k 文档"]
    F --> G["仅传输选中文档的内容 KV 到 GPU"]
    C --> G
    G --> H["执行内容注意力与生成"]
''', '解读图：每张 GPU 复制完整模型，图中分片的是记忆；扩大记忆库仍增加路由扫描量。两张卡可运行不等于全部 KV 驻留显存。对应 §2.7。')

add('DFM', '### 2.5', '融合层：记忆分支与原自注意力并行', '''
flowchart TD
    A["当前层 token 状态 h"] --> B["主干原有归一化 z"]
    B --> C["原因果自注意力"]
    B --> D["记忆交叉注意力的 Query"]
    E["本块共享的预测记忆 M"] -->|"投影为 K、V"| F["各头在记忆槽间分配注意力"]
    D --> F
    B --> G["当前 token、每头一个 sigmoid 门"]
    F --> H["逐头乘门值 → 拼接 → 输出投影"]
    G --> H
    A --> I["残差相加：h + 自注意力 + 记忆更新"]
    C --> I
    H --> I
    I --> J["原 FFN 与后续 decoder 层"]
''', '解读图：投影后的记忆跨融合层共享，各层记忆注意力和门参数独立。记忆在中间隐藏状态生效，后续层继续处理它。对应 §2.3–2.4、原文式(2)–(3)。')

add('DFM', '### 2.7', '联合训练：目标只作监督，主模型始终读取预测记忆', '''
flowchart TD
    A["前一块文本"] --> B["离线检索 Top-16<br/>冻结编码器得到目标 V"]
    A --> C["主干输入 embedding 均值池化"]
    C --> D["因果预测器<br/>结合此前块表示，输出预测记忆"]
    B -.-> E["记忆 MSE"]
    D --> E
    D --> F["共享投影器与各层门控记忆注意力"]
    F --> G["冻结主干的后续计算与词表预测"]
    H["真实后续 token"] -.-> I["语言模型交叉熵"]
    G --> I
    E --> J["总损失：LM + MSE"]
    I --> J
    J --> L["更新预测器、输入投影与融合模块<br/>梯度穿过冻结主干，主干权重不更新"]
''', '解读图：目标 V 不作为主模型记忆输入。离散检索不在反向传播通路中，新增模块接受生成损失，预测器另接受 MSE。对应 §2.6。')

add('DFM', '### 2.8', '训练与实际 QA：查询来源、刷新方式不同', '''
flowchart TD
    subgraph TRAIN["训练对齐"]
        A["已完整可见的前一块"] --> B["生成下一块的共享记忆"]
        B --> C["从前一块末位置开始<br/>支持下一块各 token 的预测"]
    end
    subgraph QA["论文报告的 QA 实现"]
        D["当前可见的 prompt 块"] --> E["形成查询与预测记忆，回答期间固定"]
        E --> F["有效 prompt 块对应位置可直接注入"]
        F --> G["回答扩展出新块索引"]
        G --> H["新块查询与掩码补零<br/>没有新块的直接记忆更新"]
    end
''', '解读图：QA 回答不会每满一块就刷新记忆；新块仍可通过自注意力使用先前受记忆影响的上下文。此处画的是原文实际评测规则，不是按训练规则设想的持续刷新流程。对应 §2.7、附录A.1。')

manifest=[]
for name in sorted({x[0] for x in changes}):
    f=ROOT/name/(name+'_Reading.md')
    original=f.read_text(encoding='utf-8')
    work=ROOT/name/(name+'_Original')/'work'/'mermaid_review_20261030'
    work.mkdir(parents=True,exist_ok=True)
    backup=work/f'{name}_Reading_before_mermaid.md'
    if not backup.exists(): shutil.copy2(f,backup)
    if backup.exists() and (work/'content_check.json').exists():
        previous=json.loads((work/'content_check.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(f.read_bytes()).hexdigest()==previous['sha256'], 'Report changed after previous diagram edit'
        original=backup.read_text(encoding='utf-8')
    text=original
    for paper,anchor,block in changes:
        if paper!=name: continue
        matches=list(re.finditer(r'^'+re.escape(anchor)+r'.*$',text,re.M))
        assert len(matches)==1,(name,anchor)
        pos=matches[0].start()
        text=text[:pos]+block+text[pos:]
    assert re.findall(r'^## .+$',text,re.M)==re.findall(r'^## .+$',original,re.M)
    # Removing just the inserted blocks must recover the exact original text.
    recovered=text
    for paper,anchor,block in changes:
        if paper==name: recovered=recovered.replace(block,'',1)
    assert recovered==original
    f.write_text(text,encoding='utf-8')
    diagrams=re.findall(r'```mermaid\n(.*?)\n```',text,re.S)
    for i,code in enumerate(diagrams,1):
        source=work/f'diagram_{i}.mmd';source.write_text(code+'\n',encoding='utf-8')
        manifest.append({'paper':name,'number':i,'source':str(source),'code':code,'directory':str(work)})
    preview=OUT/name;preview.mkdir(parents=True,exist_ok=True)
    shutil.copy2(f,preview/f.name)
    shutil.copytree(ROOT/name/(name+'_Original')/'figures',preview/(name+'_Original')/'figures',dirs_exist_ok=True)
    (work/'content_check.json').write_text(json.dumps({'original_text_preserved':True,'headings_preserved':True,'diagrams_added':len(diagrams),'preview_matches':f.read_bytes()==(preview/f.name).read_bytes(),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    print(name,len(diagrams),'diagrams added')
(ROOT/'_Mermaid_Review'/'work'/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

# 企业级长期记忆与短期记忆实现方案

## 当前项目的问题

```python
# 当前实现：简单截断用户最新消息作为记忆查询
query_text = content.strip()[:MAX_QUERY_LENGTH]  # 最多 1000 字符
```

**问题**：
| 问题 | 影响 |
|---|---|
| 截断丢失语义 | 用户说了一段长话，关键信息可能在末尾，被截掉了 |
| 噪声太多 | 聊天内容不等于检索查询，直接用作查询语义模糊 |
| 被动应对 | 无法区分"这条消息需要记忆"和"只是普通聊天" |

---

## 企业级记忆架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户输入                                      │
│                          │                                          │
│                          ▼                                          │
│              ┌───────────────────────┐                              │
│              │   记忆决策层           │                              │
│              │   (Memory Router)     │                              │
│              └───────────┬───────────┘                              │
│                          │                                          │
│            ┌─────────────┼─────────────┐                            │
│            ▼             ▼             ▼                            │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐                        │
│    │ 工作记忆  │  │ 短期记忆  │  │ 长期记忆  │                        │
│    │ Working  │  │ Short-   │  │ Long-    │                        │
│    │ Memory   │  │ term     │  │ term     │                        │
│    └──────────┘  └──────────┘  └──────────┘                        │
│         │             │             │                               │
│         └─────────────┼─────────────┘                               │
│                       ▼                                             │
│              ┌───────────────────────┐                              │
│              │   记忆融合层           │                              │
│              │   (Context Builder)   │                              │
│              └───────────────────────┘                              │
│                       │                                             │
│                       ▼                                             │
│                    LLM 调用                                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. 工作记忆（Working Memory）

### 定义
当前对话的即时上下文，直接存在于 LLM 的 prompt 中。

### 特点
| 属性 | 说明 |
|---|---|
| 容量 | 受 context window 限制（如 128K tokens） |
| 生命周期 | 单次对话，不持久化 |
| 访问方式 | 无需检索，全量注入 |

### 企业实现

```python
# 滑动窗口：保留最近 N 轮对话
class SlidingWindowMemory:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
    
    def get_context(self, messages: list[Message]) -> list[Message]:
        return messages[-self.max_turns * 2:]  # 每轮 = user + assistant

# Token 预算：动态裁剪到 token 上限
class TokenBudgetMemory:
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
    
    def get_context(self, messages: list[Message]) -> list[Message]:
        total = 0
        result = []
        for msg in reversed(messages):
            tokens = count_tokens(msg.content)
            if total + tokens > self.max_tokens:
                break
            result.append(msg)
            total += tokens
        return list(reversed(result))
```

### 代表产品
- **ChatGPT**：当前对话窗口内的消息
- **Claude**：200K context window 内的完整对话

---

## 2. 短期记忆（Short-term Memory）

### 定义
近期对话的结构化摘要，跨会话但有时效性。

### 特点
| 属性 | 说明 |
|---|---|
| 容量 | 数十条摘要 |
| 生命周期 | 数天~数周，随时间衰减 |
| 访问方式 | 按时间/相关性检索 |

### 企业实现

#### 2.1 滚动摘要（Rolling Summarization）

```python
class RollingSummaryMemory:
    """对对话历史做滚动摘要，压缩长期上下文。"""
    
    async def summarize(
        self, 
        old_summary: str | None, 
        new_messages: list[Message]
    ) -> str:
        """将旧摘要 + 新消息合并为新摘要。"""
        prompt = f"""请将以下内容压缩为一段简洁的摘要：

{f'之前的摘要：{old_summary}' if old_summary else ''}

新的对话：
{format_messages(new_messages)}

要求：
- 保留关键事实、决策、偏好
- 去除寒暄、重复、无关内容
- 不超过 200 字
"""
        return await llm.ainvoke(prompt)
    
    async def get_context(self, user_id: str) -> str:
        """获取用户的当前短期记忆摘要。"""
        summary = await self.load_summary(user_id)
        return summary or "（无历史上下文）"
```

#### 2.2 时间衰减加权

```python
class DecayWeightedMemory:
    """按时间衰减给记忆加权，近期记忆权重更高。"""
    
    def __init__(self, decay_rate: float = 0.1):
        self.decay_rate = decay_rate  # 每天衰减 10%
    
    def score(self, memory: MemoryItem, now: datetime) -> float:
        days_old = (now - memory.created_at).days
        recency_weight = math.exp(-self.decay_rate * days_old)
        
        # 最终得分 = 语义相关性 × 时间权重
        return memory.relevance_score * recency_weight
    
    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        candidates = self.vector_search(query, top_k=50)
        now = datetime.utcnow()
        scored = [(m, self.score(m, now)) for m in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_k]]
```

### 代表产品
- **LangChain ConversationSummaryBufferMemory**：摘要 + 最近 N 条消息混合
- **Mem0**：自动提取的短期记忆，带时间衰减

---

## 3. 长期记忆（Long-term Memory）

### 定义
持久化的用户知识，跨会话、跨时间存在。

### 特点
| 属性 | 说明 |
|---|---|
| 容量 | 无上限（外部存储） |
| 生命周期 | 永久，直到用户删除或系统清理 |
| 访问方式 | 向量检索 / 关键词检索 / 图谱遍历 |

### 企业实现

#### 3.1 记忆提取（Extraction）

**问题**：不是每条对话都值得记住。

**解决方案**：用 LLM 判断是否需要提取记忆。

```python
class MemoryExtractor:
    """判断对话中是否包含值得长期记忆的信息。"""
    
    EXTRACTION_PROMPT = """分析以下对话，判断是否包含需要长期记忆的信息。

对话内容：
{conversation}

如果包含以下类型的信息，请提取：
- 用户偏好（如"我喜欢用 Python"、"不要使用 Java"）
- 客观事实（如"我是后端工程师"、"我在北京"）
- 重要约束（如"回复不超过 200 字"、"用中文交流"）

输出格式（JSON）：
{
    "should_extract": true/false,
    "memories": [
        {
            "kind": "preference|fact|constraint",
            "content": "提炼后的记忆内容",
            "importance": 1-5
        }
    ]
}
"""
    
    async def extract(self, conversation: list[Message]) -> list[MemoryCandidate]:
        response = await llm.ainvoke(
            self.EXTRACTION_PROMPT.format(conversation=conversation)
        )
        result = json.loads(response)
        
        if not result["should_extract"]:
            return []
        
        return [
            MemoryCandidate(
                kind=MemoryKind(m["kind"]),
                content=m["content"],
                importance=m["importance"],
            )
            for m in result["memories"]
        ]
```

#### 3.2 查询压缩（Query Compression）

**问题**：用户最新消息不直接等于好的检索查询。

**解决方案**：用 LLM 将对话上下文压缩为精准的检索查询。

```python
class QueryCompressor:
    """将对话上下文压缩为适合向量检索的查询。"""
    
    COMPRESSION_PROMPT = """基于对话上下文，生成一个精准的检索查询。

对话历史：
{conversation_history}

当前用户消息：
{current_message}

任务：
1. 理解用户当前消息的意图
2. 结合对话历史，消除指代（如"这个"、"他"）
3. 生成一个独立的、语义完整的检索查询

输出：只输出查询文本，不要解释。
"""
    
    async def compress(
        self, 
        current_message: str, 
        history: list[Message]
    ) -> str:
        """将当前消息 + 历史压缩为检索查询。"""
        response = await llm.ainvoke(
            self.COMPRESSION_PROMPT.format(
                conversation_history=format_messages(history[-5:]),
                current_message=current_message,
            )
        )
        return response.strip()

# 示例
# 对话历史：[..., User: "我最近在学 Rust", Assistant: "Rust 是个好选择..."]
# 当前消息：User: "它和 Python 比怎么样"
# 压缩后查询："Rust 和 Python 对比"（消除了"它"的指代）
```

#### 3.3 记忆去重与合并（Deduplication & Merging）

**问题**：多次对话可能提取出重复或矛盾的记忆。

```python
class MemoryDeduplicator:
    """检测并合并语义重复的记忆。"""
    
    async def deduplicate(
        self, 
        new_memory: MemoryCandidate,
        existing_memories: list[MemoryItem]
    ) -> MemoryCandidate | None:
        """检查新记忆是否与已有记忆重复。"""
        
        # 1. 向量相似度预筛
        similar = [
            m for m in existing_memories
            if cosine_similarity(new_memory.embedding, m.embedding) > 0.85
        ]
        
        if not similar:
            return new_memory  # 无重复，直接写入
        
        # 2. LLM 判断是否真正重复
        merge_prompt = f"""判断以下记忆是否表达相同含义：

新记忆：{new_memory.content}
已有记忆：
{chr(10).join(f'- {m.content}' for m in similar)}

如果语义相同，输出合并后的版本；如果不同，输出"不重复"。
"""
        result = await llm.ainvoke(merge_prompt)
        
        if result.strip() == "不重复":
            return new_memory
        
        # 3. 返回合并后的记忆（替代旧记忆）
        return MemoryCandidate(
            kind=new_memory.kind,
            content=result.strip(),
            importance=max(new_memory.importance, *(m.importance for m in similar)),
            replaces=[m.id for m in similar],  # 标记要替换的旧记忆
        )
```

#### 3.4 记忆重要性评分

```python
class MemoryImportanceScorer:
    """评估记忆的重要性，决定保留优先级。"""
    
    SCORING_PROMPT = """评估以下记忆的重要性（1-5 分）：

记忆内容：{content}
记忆类型：{kind}

评分标准：
5 分：核心身份信息（姓名、职业、核心偏好），几乎不会变化
4 分：重要偏好或约束，可能偶尔变化
3 分：一般事实，有一定时效性
2 分：临时偏好，可能很快改变
1 分：一次性信息，下次对话可能无关

输出：只输出数字 1-5。
"""
    
    async def score(self, memory: MemoryCandidate) -> int:
        response = await llm.ainvoke(
            self.SCORING_PROMPT.format(
                content=memory.content,
                kind=memory.kind.value,
            )
        )
        return int(response.strip())
```

### 代表产品

| 产品 | 长期记忆实现 |
|---|---|
| **ChatGPT Memory** | LLM 自动提取 + 用户可手动管理 |
| **Mem0** | 自动提取 + 去重合并 + 重要性评分 |
| **MemGPT/Letta** | 分层记忆 + 主动记忆管理 |
| **LangChain LTM** | 向量存储 + 实体知识图谱 |

---

## 4. 记忆融合（Context Building）

### 定义
将工作记忆、短期记忆、长期记忆组合成最终 prompt。

### 企业实现

```python
class ContextBuilder:
    """将多层记忆融合为 LLM 可消费的上下文。"""
    
    def __init__(
        self,
        system_prompt: str,
        core_memory_budget: int = 500,    # tokens
        long_term_budget: int = 1000,     # tokens
        short_term_budget: int = 2000,    # tokens
        working_budget: int = 4000,       # tokens
    ):
        self.system_prompt = system_prompt
        self.budgets = {
            "core": core_memory_budget,
            "long_term": long_term_budget,
            "short_term": short_term_budget,
            "working": working_budget,
        }
    
    async def build(
        self,
        user_id: str,
        current_message: str,
        working_memory: list[Message],
    ) -> list[Message]:
        """构建最终 prompt。"""
        
        # 1. 核心记忆（始终注入）
        core_memories = await self.get_core_memories(user_id)
        core_section = self.format_core_memories(core_memories)
        
        # 2. 长期记忆（按需检索）
        query = await self.compress_query(current_message, working_memory)
        long_term_memories = await self.search_long_term(user_id, query)
        long_term_section = self.format_long_term(long_term_memories)
        
        # 3. 短期记忆（近期摘要）
        short_term_summary = await self.get_short_term_summary(user_id)
        short_term_section = self.format_short_term(short_term_summary)
        
        # 4. 工作记忆（当前对话）
        working_section = self.truncate_to_budget(
            working_memory, 
            self.budgets["working"]
        )
        
        # 5. 组装最终 prompt
        system_content = f"""{self.system_prompt}

## 关于用户（核心记忆）
{core_section}

## 相关历史知识（长期记忆）
{long_term_section}

## 近期上下文（短期记忆）
{short_term_section}
"""
        
        return [
            SystemMessage(content=system_content),
            *working_section,
            HumanMessage(content=current_message),
        ]
```

### 融合策略对比

| 策略 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| **全量注入** | 不丢失信息 | 容易超出 context window | 记忆量小（< 1000 tokens） |
| **Top-K 检索** | 控制 token 用量 | 可能漏掉相关记忆 | 记忆量大，相关性明确 |
| **分层注入** | 核心信息不丢失 | 实现复杂 | 企业级产品 |
| **动态预算** | 灵活适应不同场景 | 需要精细调优 | 高级应用 |

---

## 5. 对比总结

| 维度 | 工作记忆 | 短期记忆 | 长期记忆 |
|---|---|---|---|
| **数据来源** | 当前对话 | 近期对话摘要 | 跨会话提取 |
| **存储位置** | 内存（prompt） | Redis / 数据库 | PostgreSQL + pgvector |
| **检索方式** | 无需检索 | 按时间 / 摘要 | 向量检索 / 图谱 |
| **更新时机** | 每轮对话 | 对话结束时摘要 | LLM 判断后提取 |
| **容量限制** | Context window | 数十条摘要 | 无上限 |
| **生命周期** | 单次会话 | 数天~数周 | 永久 |

---

## 6. DeepResearch 演进建议

### 当前阶段（已实现）
- ✅ 长期记忆：向量检索 + MemoryKind 分类
- ✅ 工作记忆：完整 messages 列表

### 下一步
1. **查询压缩**：用 LLM 将对话上下文压缩为检索查询，替代简单截断
2. **记忆提取**：对话结束后自动判断是否提取长期记忆
3. **短期记忆**：滚动摘要机制，保留近期上下文

### 远期
- **分层记忆**：区分核心记忆（始终注入）和长期记忆（按需检索）
- **记忆去重**：新记忆写入前检查语义重复
- **图谱增强**：利用 Neo4j 构建记忆关联

---

## 参考实现

| 开源项目 | 特点 |
|---|---|
| [Mem0](https://github.com/mem0ai/mem0) | 自动提取 + 去重 + 重要性评分 |
| [MemGPT/Letta](https://github.com/cpacker/MemGPT) | 分层记忆 + 主动管理 |
| [LangChain LTM](https://python.langchain.com/docs/modules/memory/) | 多种记忆类型集成 |
| [Zep](https://github.com/getzep/zep) | 长期记忆 + 实体提取 |

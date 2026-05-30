import re

file_path = 'temp/architecture.html'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Translation map (simple example)
translations = {
    "LLMBO Hybrid System Architecture": "LLMBO 混合系统架构",
    "Gaussian Process Surrogate Model & Pointwise LLM (Log-probs) Prior Collaborative Decision Engine": "高斯过程代理模型与逐点 LLM（Log-probs）先验协作决策引擎",
    "Workflow Map": "工作流图",
    "Layer Stack": "架构层级",
    "Operational View": "操作模式",
    "Benchmark View": "基准测试模式",
    "Vector Memory (RAG)": "向量记忆 (RAG)",
    "Doubao Embedding": "豆包嵌入模型",
    "FastAPI Backend": "FastAPI 后端",
    "GP UCB Pre-screening": "GP UCB 预筛选",
    "Pointwise LLM (Log-probs)": "逐点 LLM (Log-probs)",
    "Adaptive Weight Coordinator": "自适应权重协调器",
    "Hybrid Fusion": "混合融合",
    "System Orchestration": "系统调度",
    "System Flow": "系统流程"
}

for en, zh in translations.items():
    content = content.replace(en, zh)

with open('temp/architecture_zh.html', 'w', encoding='utf-8') as f:
    f.write(content)

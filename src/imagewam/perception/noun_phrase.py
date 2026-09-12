"""Qwen3 名词短语解析:从机器人指令提取物体名词短语。

纯文本任务,Qwen3-4B(base/instruct 都行)即可,不需要 VL。
语言 embedding 见 embed_noun_phrases(Task 5)。
"""
import torch

_PROMPT = (
    "List the physical object noun phrases in the following robot instruction. "
    "Output ONLY a comma-separated list of lowercase noun phrases, no extra text.\n"
    "Instruction: {instr}\n"
    "Objects:"
)


@torch.no_grad()
def extract_noun_phrases(instruction: str, qwen_model, qwen_tokenizer, max_new_tokens: int = 32) -> list[str]:
    """返回指令里的物体名词短语列表。

    Qwen3-4B 是 base 模型(非 Instruct),用 chat_template 提升指令遵循,并后处理
    (取第一行、去重、过滤 >4 词或 >30 字的幻觉句)抑制重复/幻觉。
    """
    prompt = _PROMPT.format(instr=instruction)
    messages = [{"role": "user", "content": prompt}]
    text = qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = qwen_tokenizer(text, return_tensors="pt").to(qwen_model.device)
    out = qwen_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = qwen_tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    first_line = gen.strip().split("\n")[0]
    seen, phrases = set(), []
    for p in first_line.split(","):
        p = p.strip().lower().strip(".。;；")
        if p and p not in seen and 1 <= len(p.split()) <= 4 and len(p) <= 30:
            seen.add(p)
            phrases.append(p)
    return phrases


@torch.no_grad()
def embed_noun_phrases(phrases: list[str], qwen_model, qwen_tokenizer) -> torch.Tensor:
    """名词短语 → Qwen3 embedding(与 text 同空间,供 action query 匹配)。

    mean-pool 末层 hidden。返回 [n_phrases, d_qwen=2560]。
    """
    embs = []
    for p in phrases:
        inp = qwen_tokenizer(p, return_tensors="pt").to(qwen_model.device)
        hidden = qwen_model(**inp, output_hidden_states=True).hidden_states[-1]  # [1, L, d]
        embs.append(hidden.mean(dim=1).squeeze(0))  # [d]
    return torch.stack(embs)  # [n_phrases, d_qwen]

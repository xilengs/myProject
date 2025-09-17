import sacrebleu

# 单个参考译文列表
references = [
    "今天天气很好。",
    "我喜欢在雨天散步。",
    "今天要早点下班。"
]

# 模型生成的候选翻译
hypotheses = [
    "今天天气不错。",
    "下雨的时候我喜欢散步。",
    "今天下班要早些。"
]

# sacrebleu 需要references 是list[list[str]]
# 即使只有一个参考，也要是内层 list
reference = [references] # 变成：[[ref1, ref2, ref3]]

# bleu = sacrebleu.corpus_bleu(hypotheses, reference, tokenize='zh')
bleu = sacrebleu.corpus_bleu(references, reference, tokenize='zh')

print(f"BLEU score: {bleu.score:.2f}")

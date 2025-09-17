import sys
print(sys.executable)

import sentencepiece as spm

sp_cn = spm.SentencePieceProcessor()
sp_cn.Load('zh_bpe.model')
sp_en = spm.SentencePieceProcessor()
sp_en.Load('en_bpe.model')

text = "今天天气非常好。"
text_en = "Nice to Meet you."

encode_result = sp_cn.Encode(text, out_type=int)
print("编码：", encode_result)

decode_result = sp_cn.Decode(encode_result)
print("解码：", decode_result)

encode_result_en = sp_en.Encode(text_en, out_type=int)
print("编码：", encode_result_en)

decode_result_en = sp_en.Decode(encode_result_en)
print("解码：", decode_result_en)
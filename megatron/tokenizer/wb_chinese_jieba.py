# encoding=utf-8
import jieba
import sys
import unicodedata

for aline in sys.stdin:
    aline = aline.strip()
    aline = unicodedata.normalize('NFKC', aline)
    seg_list = jieba.cut(aline)
    print(' '.join(seg_list))


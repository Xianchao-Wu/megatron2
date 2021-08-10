#########################################################################
# File Name: test_chinese_ltp.py
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Mon Aug  9 07:21:29 2021
#########################################################################

from ltp import LTP

ltp = LTP()  # 默认加载 Small 模型
seg, hidden = ltp.seg(["他叫汤姆去拿外衣。"])
print('seg={}, hidden={}'.format(seg, hidden))

pos = ltp.pos(hidden)
print('pos={}'.format(pos))

ner = ltp.ner(hidden)
print('ner={}'.format(ner))

srl = ltp.srl(hidden)
print('srl={}'.format(srl))

dep = ltp.dep(hidden)
print('dep={}'.format(dep))

sdp = ltp.sdp(hidden)
print('sdp={}'.format(sdp))

# encoding=utf-8
import jieba

#jieba.enable_paddle()# 启动paddle模式。 0.40版之后开始支持，早期版本不支持
strs=["我来到北京清华大学","乒乓球拍卖完了","中国科学技术大学"]
for astr in strs:
            seg_list = jieba.cut(astr,use_paddle=True) # 使用paddle模式
            print("Paddle Mode: " + '/'.join(list(seg_list)))

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))

seg_list = jieba.cut("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
print('jieba seg default:' + ' '.join(seg_list))
            
seg_list = jieba.cut("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", cut_all=True)
print('jieba seg cut.all=True:' + ' '.join(seg_list))

seg_list = jieba.cut("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", use_paddle=True)
print('jieba seg use paddle=True:' + ' '.join(seg_list))

from ltp import LTP

ltp = LTP()  # 默认加载 Small 模型

seg, _ = ltp.seg(["小明硕士毕业于中国科学院计算所，后在日本京都大学深造"])
print('ltp, seg={}'.format(seg))


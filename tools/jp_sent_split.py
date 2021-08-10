import nltk

jp_sent_splitter = nltk.RegexpTokenizer(u'[^！？。]*[！？。]+')

intxt = '''お忙しい大変申し訳ございません。。。どうもありがとうございます！ご飯を食べました。美味しいでしょうか！！！'''

outtxt = jp_sent_splitter.tokenize(intxt)
print(outtxt)
for asent in outtxt:
    print(asent)

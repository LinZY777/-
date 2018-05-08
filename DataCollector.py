from __future__ import print_function
from __future__ import unicode_literals

text = open('./news.txt',encoding='utf-8').read()


from snownlp import normal
from snownlp import seg
from snownlp.summary import textrank


if __name__ == '__main__':
    t = normal.zh2hans(text)
    sents = normal.get_sentences(t)
    doc = []
    for sent in sents:

        words = seg.seg(sent)
        words = normal.filter_stop(words)
        doc.append(words)
    rank = textrank.TextRank(doc)
    rank.solve()
    for index in rank.top_index(5):
        print(sents[index])
    keyword_rank = textrank.KeywordTextRank(doc)
    keyword_rank.solve()
    for w in keyword_rank.top_index(5):
        print(w)
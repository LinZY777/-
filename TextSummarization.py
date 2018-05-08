# 实验目的：掌握基于最大覆盖模型的文本摘要的基本思想及其贪心算法。
# 实验题目：基于最大覆盖模型的文本摘要
# 	针对有关你喜欢的人或物的评论主题，使用爬虫程序或人工下载一些网页；
# 	计算每个词的tf*idf值（排除停用词），每个句子视为其各个词的集合。每个句子的代价
#       视为其字的个数（包括停用词及标点符号）。
# 	利用贪心算法分别产生长度限制为400和800个字的摘要，并在实验报告中打印出来。
# 	具体算法见上课的PPT。
# ==============================================================================

import jieba
import jieba.analyse

def get_sentences(texts):
    """
    将文本拆分成单个句子，存放到列表中
    :param texts:需要拆分处理的文本
    :return:句子的列表
    """

    splitstr = '。！？'        # 以这些标点作为句子的结尾
    start = 0
    curr = 0        # 当前检测的字符的位置
    sentences = []

    while curr < len(texts):

        currWord = texts[curr]
        if curr == len(texts) - 1:
            nextWord = ""
        else:
            nextWord = texts[curr+1]
        if currWord in splitstr:        # 当前字符为句子结尾
            if nextWord == "”":         # 还需要检测下一个字符是不是引号，因为有时句子（对话）会以类似'。”'结束
                curr += 1
            sentences.append(texts[start:curr+1])       # 将句子添加到列表中
            start = curr + 1
        curr += 1

    return sentences

def cal_TFIDF(texts):
    """
    计算文本中每个词的TF*IDF值（排除停用词）
    :param texts: 需要处理的文本
    :return:一个字典，key是某个词，value是其对应的TF*IDF值
    """
    dist = {}
    tags = jieba.analyse.extract_tags(texts,len(texts),True)
    for tag in tags:
        dist[tag[0]] = float(tag[1])
    return dist


if __name__ == "__main__":
    # 读取文本文件
    f = open("./src.txt",encoding="utf-8")
    texts = f.read().replace("\n","")       # 去掉文章中的换行符
    f.close()

    # 获取文本中的句子
    sentences = get_sentences(texts)
    # 计算文本中每个词的tf*idf值（排除停用词）
    dist = cal_TFIDF(texts)
    # 对每个句子分词并添加原句的索引
    split_sentences = []
    index = 0
    for sentence in sentences:
        split_sentences.append(jieba.lcut(sentence))
        split_sentences[index].append(index)
        index += 1

    ### Algorithm BMC ###
    # 第一步，寻找权重最大的句子
    wSt = 0
    t = 0

    for split_sentence in split_sentences:
        weight = 0
        for word in split_sentence[:-1]:
            if word in dist:
                weight += dist[word]
        if weight > wSt:
            wSt = weight
            t = split_sentence[-1]

    # 第二步，获取集合G，计算w(G)
    word_sum = 0    # 当前摘要的字数，即c(G)
    wg = 0          # w(G),当前集合G的权值
    g = []          #
    L = 200         # 限定摘要的字数
    while len(split_sentences) != 0:        # 循环直至R集合为空
        max_weight = 0          # 用于记录最大项的w(S)
        max_weight_c = 0        # 用于记录最大项的w(S)/c
        max_weight_index = 0    # 记录最大项在split_sentences中的索引

        # 在R中寻找值w(S)/c最大的项
        index = 0
        for split_sentence in split_sentences:
            weight = 0
            for word in split_sentence[:-1]:
                if word in dist:
                    weight += dist[word]
            if weight/len(sentences[split_sentence[-1]]) > max_weight_c:
                max_weight = weight
                max_weight_c = weight/len(sentences[split_sentence[-1]])
                max_weight_index = index
            index += 1
        # 最大项Si
        Si = split_sentences[max_weight_index]
        # R=R\{Si}
        del split_sentences[max_weight_index]
        # if c(G)+ci <= L
        if word_sum + len(sentences[Si[-1]]) <= L:
            word_sum += len(sentences[Si[-1]])
            g.append(Si[-1])
            wg += max_weight
            for word in Si[:-1]:
                for split_sentence in split_sentences:
                    for x in split_sentence[:-1]:
                        if x == word:
                            split_sentence.remove(x)

    # 第三步，判断w(G)与w(St)的大小
    summary = ""
    if wg >= wSt:
        g.sort()
        for i in g:
            summary += sentences[i]
    else:
        summary = sentences[t]

    print("Summary:")
    print(summary)





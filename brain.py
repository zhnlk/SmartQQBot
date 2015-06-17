# encoding: utf-8
#-*- coding=utf-8 -*-
#coding=utf-8

from __future__ import print_function, unicode_literals, division
import sys
import jieba
import jieba.posseg as pseg
import nltk
import re

reload(sys)
sys.setdefaultencoding("utf-8")


class Brain():

    def __init__(self):
        self.sample = {}
        self.sampleLoaded = False
        self.totalScore = 0
        self.judgeValve = 0

    def loadSample(self):
        # 安全载入dict类型的关键词词频文件
        if not self.sampleLoaded:
            self.sampleLoaded = True
            with open("./data/freqData") as sampleFile:
                for line in sampleFile:
                    line = line.split(" ")
                    self.sample[line[0]] = int(line[1])
        return self.sample

    def anaBaseSample(self, self_print=False):
        # 通过basePostiveSample文件与baseNegativeSample文件生成关键词词频文件
        # 这两个样本文件需保证百分百正确率
        tmpList = {}

        with open("./data/basePostiveSample") as sampleFile:
            seg_list = jieba.cut(sampleFile.read())
            freq_list = nltk.FreqDist(seg_list)
            freq_list = sorted(freq_list.items(), key=lambda d: d[1])
            for a, b in freq_list:
                tmpList[a] = b

        with open("./data/baseNegativeSample") as exceptFile:
            eseg_list = jieba.cut(exceptFile.read())
            efreq_list = sorted(nltk.FreqDist(eseg_list).items(), key=lambda d: d[1])
            for a, b in efreq_list:
                if a in tmpList:
                    tmpList[a] -= b

        freq_list = sorted(tmpList.items(), key=lambda d: d[1])
        outputData = ""
        with open("./data/freqData", "w+") as outputFile:
            for word, count in freq_list:
                if len(word) >= 2 and not word.isdigit():
                    if self_print:
                        print(word, count)
                    if count > 10:
                        self.totalScore += abs(int(count))
                    outputData += (str(word) + " " + str(count)+"\n")
            outputFile.write(outputData)

        print("Analyze done.")
        print("TotalScore:" + str(self.totalScore))

    def isLostMsg(self, stn, returnDetail=False):
        if not self.sampleLoaded:
            self.loadSample()
        # 使用基于关键词评分模型，简单判断是否失物招领信息
        stn = stn.decode("utf-8")
        score = 0
        scoreList = []
        for word, freq in self.sample.items():
            if word in stn and freq >= 5:
                # print(word, freq)
                score += int(freq)
                scoreList.append([word, freq])
        # print(score)

        if not returnDetail:
            return int(score) + int(self.judgeValve)
        else:
            return [int(score) + int(self.judgeValve)] + scoreList

    def anaFile(self, filename, needOutput=False):
        # 对文件里每一行使用关键词评分模型进行评分
        tmpOutput = {}
        index = 0
        with open(str(filename)) as anaFile:
            anaFile = anaFile.readlines()
            fileLen = len(anaFile)
            for line in anaFile:
                index += 1
                if index % 100 == 0:
                    print("%.2f" % (index / fileLen * 100), "%")
                lineScore = self.isLostMsg(line)
                if lineScore:
                    tmpOutput[line] = lineScore

        tmpOutput = sorted(tmpOutput.items(), key=lambda d: d[1])
        if needOutput:
            for a, b in tmpOutput:
                if b > 0:
                    print(b, a, end="")

        return tmpOutput

    def anaStnPosByRE(self, stn):
        # 使用正则表达式，初步提取句子中物品丢失的位置
        successFlag = False
        stn = str(stn).decode("utf-8")
        pattern = re.compile(ur'((?<=(从|(?<!终)于|经|在|去|：|:))|(?<=(早上|中午|下午|晚上|下课|范围)))[a-zA-Z0-9\u4e00-\u9fa5\- ()、]+?((?<!的)(?=刚刚|吃|做|玩|用|消失|不见(?!了)|遗失|丢(?!在)|捡|拾|发现|上机|错拿|拿错|拿了|】|吧)|(这(?!个)|处|内(?!环)|位置|附近|之间|间|那|旁|路上|路|段|前面|教室|里面|(?<!空)中(?!午)|沿途))')
        rs = []
        tmp = 1
        while tmp:
            tmp = pattern.search(stn)
            # 测试使用findall 以及使用关键词评分系统 判断内容是否地点
            # 若整句被匹配命中存在bug
            if tmp and stn != tmp.group():
                # print(type(stn), stn)
                # print(type(tmp.group()), tmp.group())
                successFlag = True
                stn = tmp.group(0)
            else:
                break
        rs.append(stn)

        classroomPattern = re.compile(ur'[ab]\d{1,2}(?:\([1-6]\d{2}\)|[\D\W]{0,2}\d{3})', re.I)
        classroomRs = classroomPattern.findall(stn)
        rs += classroomRs

        if successFlag:
            return rs
        return []

    def anaStnContactByRE(self, stn):
        # 使用正则表达式，初步提取句子中的联系方式
        rs = []
        stn = str(stn).decode("utf-8")
        lPattern = re.compile(ur'[\D|](1[835]\d{9})(?!\d)')
        sPattern = re.compile(ur'[\D|](6\d{4,5})(?!\d)')
        roomPattern = re.compile(ur'c\d{1,2}[\D\W]{0,2}[1-7]\d{2}', re.I)
        rs = list(lPattern.findall(stn)) + list(sPattern.findall(stn)) + list(roomPattern.findall(stn))
        # 测试使用findall 以及使用关键词评分系统 判断内容是否地点
        return rs

    def anaStnQQByRE(self, stn):
        rs = []
        qqPattern = re.compile(ur'q.*?([1-9]\d{4,10})', re.I)
        rs = list(qqPattern.findall(stn))
        return rs

    def anaStnCardNumByRE(self, stn):
        rs = []
        qqPattern = re.compile(ur'卡号.*?([\d\*\.][\d \*\.]{17,23}[\d\*\.])', re.I)
        rs = list(qqPattern.findall(stn))
        return rs

    def extract_info(self, stn, self_print=False):

        stn = str(stn).decode("utf-8")
        # 综合性提取句子中的信息
        name = []
        item = []
        cardNum = []
        pos = []
        acad = []
        contact = []
        qq = []
        other = []

        # 替换长度大于等于2的空格为单空格
        subBlanks = re.compile(ur'\s{2,}')
        stn = subBlanks.sub(' ', stn)

        # 句子使用词性判断分词
        cStn = pseg.cut(stn)

        gradePattern = re.compile(ur'\d{2,4}级')
        try:
            other += gradePattern.search(stn)
        except:
            pass
        pos += self.anaStnPosByRE(stn)
        contact += self.anaStnContactByRE(stn)
        qq += self.anaStnQQByRE(stn)
        cardNum += self.anaStnCardNumByRE(stn)

        for word in cStn:
            if word.flag == "nr" and len(word.word) > 1:
                name.append(word.word)
            elif word.flag == "ns":
                pos.append(word.word)
            # 此处使用"nz（特殊名词）"来作为可能的失物
            elif word.flag == "nz":
                item.append(word.word)
            # 此处使用自创标识na，代表学院名称
            elif word.flag == "na":
                acad.append(word.word)
                contact.append(word.word)

        # 位置去重复
        if pos:
            tmpRemoveList = []
            for i in pos:
                for otherPos in [x for x in pos if x != i]:
                    # print("testing", i, "in", otherPos)
                    if i in otherPos:
                        tmpRemoveList.append(i)
                        break
            # 之所以采用这么笨的办法，是因为在循环中删除元素会导致循环不完整
            if tmpRemoveList:
                for i in tmpRemoveList:
                    pos.remove(i)
                    # print("romove", str(i))

        rs = {'name': set(name), 'item': set(item), 'itemDetail': set([]), 'cardNum': set(cardNum), 'pos': set(pos), 'acad': set(acad), 'major': set([]), 'contact': set(contact), 'qq': set(qq), 'other': set(other)}

        if len(rs['pos']) + len(rs['item']) + len(rs['contact']) + len(rs['name']) >= 1:

            if self_print:
                print(stn)
                print("姓名:%s" % "   ".join(rs['name']))
                print("丢失物品:%s" % "   ".join(rs['item']))
                print("位置:%s" % "   ".join(rs['pos']))
                print("联系方式:%s" % "   ".join(rs['contact']))
                print("QQ:%s" % "   ".join(rs['qq']))
                print("其他信息:%s\n\n" % "   ".join(rs['other']))
            return rs
        else:
            return False

    def start_up(self):
        try:
            jieba.initialize()
        except:
            print("jieba initializing error.")
            return False
        try:
            jieba.load_userdict("./data/userdict.txt")
        except:
            print("jieba load user dictionary error.")
            return False
        self.loadSample()
        return True

    def fullProcessStn(self, stn):
        stn = str(stn).decode("utf-8")
        if self.isLostMsg(stn):
            tmpinfo = self.extract_info(stn)
            if tmpinfo:
                return tmpinfo
        return False

if __name__ == '__main__':
    AIbrain = Brain()
    AIbrain.start_up()
    usrIp = raw_input("Please input the sentence you want to process:\n")
    while usrIp:
        tmpRs = AIbrain.fullProcessStn(usrIp)
        if tmpRs:
            print("\n")
            print("姓名:%s" % "   ".join(tmpRs['name']))
            print("丢失物品:%s" % "   ".join(tmpRs['item']))
            print("位置:%s" % "   ".join(tmpRs['pos']))
            print("联系方式:%s" % "   ".join(tmpRs['contact']))
            print("QQ号:%s" % "   ".join(tmpRs['qq']))
            print("其他信息:%s\n\n" % "   ".join(tmpRs['other']))
        else:
            print("\nIt is not a lost found message.\n")
        print("="*50)
        usrIp = raw_input("Please input the sentence you want to process:\n")

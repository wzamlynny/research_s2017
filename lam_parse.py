import sys

# http://stackoverflow.com/questions/8337069/find-the-index-of-the-nth-item-in-a-list
from itertools import compress, count, imap, islice
from functools import partial
from operator import eq

import numpy as np #TODO use NP FOR N-DIMENTIONAL ARRAYS


topics = []
vocab = []

# Empties topics file
def clearTopics():
    with open("output/topics.txt", "w") as file:
        file.write("")
    print "Cleared train"

def parseVocab(filename):
    with open(filename) as textfile:
        for line in textfile:
            vocab.append(line[:-1]) # Removes the \n at the end

# http://stackoverflow.com/questions/8337069/find-the-index-of-the-nth-item-in-a-list
def nth_item(n, item, iterable):
    indicies = compress(count(), imap(partial(eq, item), iterable))
    return next(islice(indicies, n, None), -1)

def getWords(prob, vocab, n):
    # Prob is probability where index is word id
    # I want ret to be [[a,b,c], [a,b,c]], where a = id, b = word, c = percent
    ret = []
    # This returns nice pairs but I can't parse pairs as easily accross column
    dt = np.dtype([('id', np.int_), ('word', np.object), ('prob', np.float_)])

    # SortedProb stores the probabilities in decreasing order
    sortedProb = sorted(prob, reverse=True)

    # If n is 0 then we want all the words
    if n >= 0:
        sortedProb = sortedProb[:n]
        # ret = np.empty(n, dtype=dt)
        ret = np.zeros([n, 2], dtype=np.float)
    else:
        # ret = np.empty(len(prob), dtype=dt)
        ret = np.zeros([len(prob), 2], dtype=np.float)

    # Storing the value of the previous word so words are not repeated
    itemNum = 0
    prevWord = 0

    # Need to find out the original index to get word
    for i in range(len(sortedProb)):
        p = sortedProb[i]
        if prevWord != p:
            j = prob.index(p) # this always find the first occurence
            itemNum = 1
        else:
            j = nth_item(itemNum, p, prob)
            itemNum = itemNum + 1
        # temp = [j, vocab[j], p]
        # ret[i][0] = j
        # ret[i][1] = vocab[j]
        # ret[i][2] = p
        ret[i][0] = j
        ret[i][1] = p
        prevWord = p
    # print ret
    return ret

def writeTopics(topic):
    #Topic has the form [id, word, percent]
    # TODO how to include the word id? - formatting
    # Maybe output a file for each topic?
    with open("output/topics.txt", "a") as file:
        buffer = ""
        for i in range(len(topic)):
            # buffer = buffer + str(topic[i][1]) + ":" + str(topic[i][2])
            buffer = buffer + vocab[int(topic[i][0])] + ":" + str(topic[i][1])
            if i < len(topic) - 1:
                buffer = buffer + " "
        file.write(buffer)
        file.write("\n")


def clearDocTopics():
    with open("output/doc_topics.txt", "w") as file:
        file.write("")
def writeDocTopics (topics):
    # print topics
    argSort = np.argsort(topics)[::-1]

    with open("output/doc_topics.txt", "a") as file:
        for arg in argSort:
            if (topics[arg] > 0):
                pass
                buffer = str(arg + 1) + ":" + str("%.3f" % round(topics[arg],2)) + " "
                file.write(buffer)
        # file.write(str(n))
        file.write("\n")
    # Compute posterior for each word, prob over Total
    # sum of all words in that document
    # look at topics that are largest
    # Do we want to only look at the top 10 topics or all of them? TODO
def getDocumentTopics(train, topics):
    # topics is 3d array containing [[id, word, percent]]
    clearDocTopics()
    doc = []
    with open(train, "r") as file:
        curLine = 0
        for line in file:
            # if curLine == docNum:
            doc = []
            isFirst = 0
            for pair in line.split(" "):
                if isFirst != 0:
                    pair = pair.split("\n")[0] # Remove any new line symbols
                    pair = pair.split(":") # Split into array
                    pair = map(int, pair) # Make pair into ints
                    doc.append(pair)
                isFirst = isFirst + 1 # We are ignoring the first one
                # print pair
            docTopic = getTopic(doc, train, topics)
            # print "Topics for document " + str(curLine + 1) + ": " + str(docTopic)
            writeDocTopics(docTopic)
            curLine = curLine + 1
  
    # Doc at this point is an array of [index,count]
    # getTopics(doc, train, topics)
    

def getTopic(doc, train, topics):
    probArr = np.zeros([len(topics), len(doc)],dtype=float)
    for i in range(len(doc)):
        # doc[i] is a word
        # This should contain the probability of the word in each topic
        
        for j in range(len(topics)): # for each topic
            if float(doc[i][0]) in topics[j,:, 0]: # looking at the id of each word in the current topic
                index = np.where(topics[j, :, 0] == float(doc[i][0]))
                # prob of word in doc = weight of word in topic * occurrances in doc
                probArr[j][i] = topics[j][index[0][0]][1] * doc[i][1]
    np.set_printoptions(threshold='nan')

    # we want word / sum
    # probArr[i][j] / sum[j]
    probSum = np.sum(probArr, axis=0)
    probTopic = np.zeros([len(probArr)], dtype=float)
    numTopic = 0
    for topic in probArr:
        numTopic = numTopic + 1
        topicSum = 0
        for i in range(len(topic)):
            if probSum[i] > 0:
                postWord = topic[i] / probSum[i]
            else:
                postWord = 0
            # print postWord
            topicSum = topicSum + postWord
        probTopic[numTopic-1] = topicSum
    # print probTopic
    return probTopic

    # maxTopic = np.argmax(probTopic) + 1
    # # print "Topic for document: " + str(maxTopic)
    # return maxTopic

def main():
    # Get command line arguments
    argc = len(sys.argv)
    if argc < 3:
        print "Usage: input_file vocab_doc [num_words]"
        sys.exit()

    filename = sys.argv[1]
    vocab_doc = sys.argv[2]
    if argc == 4:
        num_words = int(sys.argv[3])
    else:
        num_words = -1

    # Parse vocab doc
    parseVocab(vocab_doc)

    topics = []
    # topics = np.array([])
    # print topics

    # Find the topics
    with open(filename) as csvfile:
        for row in csvfile:
            cur_topic = []
            for col in row.split(","):
                cur_topic.append(float(col))
            cur_topic = getWords(cur_topic, vocab, num_words)
            topics.append(cur_topic)
            # topics = np.append(topics, cur_topic, axis=0)

    print "Total topics: " + str(len(topics))

    # print topics
    topics = np.array(topics)
    # print topics

    # Clears the topics file
    clearTopics()

    # print topics
    for i in range(len(topics)):
        # print "Topic " + str(i) + ": " + str(topics[i])
        # Writes to the topic file
        writeTopics(topics[i])
    print "Wrote to topics"

    # TODO Figure out how to get ALL document topics
    # CURRENTLY HARDCODED

    getDocumentTopics('output/train.ldac', topics)
    print "Wrote to doc_topics"


#Call the main function to start the program
main()

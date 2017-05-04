import sys
import glob

import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
wnl = WordNetLemmatizer()

import re

np.set_printoptions(threshold='nan')

vocab = []

vocabCount = [] # Counts the number of occurences of a word
vocabAppearance = [] # Counts the number of documents vocab appears in

minCount = 5
minAppearance = 3

# Information on what words are invalid
# invalidSentences TODO
invalidLength = 3 # Stores the min length of a word
invalidCharacters = ['.', ',', '"', "'", '-', '_', '(', ')', '<', '>', '[', ']', '?', ':', ';',
                    '!', '@', '#', '$', '%', '^','&', '*', '/', '\r','\n',
                    '1','2','3','4','5','6','7','8','9','0'] # Stores characters to be removed
invalidWords = ["that","with","from","were", "would", "this", "only", "until",
                "they", "their", "have", "which", "also", "could", "these", "such", 'been',
                "into", "than", "then", "more", "about", "will", "company", "business",
                "there", "shall", "other", "through", "when", "even", "most", "between",
                "over", "some", "well", "what", "where", "like", "including", "them",
                "because", "most", "while", "after", "each", "said", 
                "january", "february", "march", "april", "may", "june", "july", "august",
                "september", "october", "november", "december"] # Stores any words to be ignored

wordsF = ["she", "her", "hers"]
wordsM = ["he", "him", "his"]
countF = []
countM = []

# Empties train file
def clearTrain():
    with open("output/train.ldac", "w") as file:
        file.write("")
    print "Cleared train"

# Cleans out any bad characters from a word, and returns that word
def cleanWord(word):
    # Remove any non-unicode characters
    word=word.decode('utf-8','ignore').encode("utf-8")
    # Remove any invalid characters
    for c in word:
        if c in invalidCharacters:
            wordParts = word.split(c)
            # print wordParts
            word = ""
            for i in wordParts:
                word = word + i
    return word

def getWordGender(word, docNum):
    # TODO count num F words vs M words
    # countF = 0
    # countM = 0
    if len(countF) <= docNum:
        countF.append(0)
        countM.append(0)
        # genderClass.append(0.5)
    if word in wordsF:
        countF[docNum] = countF[docNum] + 1
    elif word in wordsM:
        countM[docNum] = countM[docNum] + 1
    # print "Doc " + str(docNum) + " is " + str(genderClass[docNum]) + " percent female."


# Removes any invalid characters from word, or returns empty string if word is invalid
def getValidWord(word):
    # print "Original word: " + word
    # Check word is sufficiently long
    if len(word) <= invalidLength:
        return ""
    # Change word to lower case
    word = word.lower()
    # Getting the root of the word with NLTK
    try:
        word = wnl.lemmatize(word)
    except UnicodeDecodeError:
        print "Word " + word + "could not be lemmatized."

    # Check if word itself is invalid
    if word in invalidWords:
        return ""

    # print "Valid Word: " + word
    return word

def parseDoc(filename, docNum):
    cur_words = []  # index
    cur_counts = []  # count
    cur_total = 0  # total count

    cur_stats = np.empty(shape=(0, 2), dtype=int)

    # Read from file
    try:
        with open(filename) as textfile:
            for line in textfile:
                # for line in textPart.split('\r'):
                # print line.split('\r')
                # TODO Remove line if copyright/everywhere
                # print line + '\n'
                line = re.sub("Copyright .* Harvard Business School." , '', line)
                line = re.sub("This document is authorized for use only by .*", '', line)
                line = re.sub("Do Not Copy or Post", '', line)
                # print line
                # raw_input("Press Enter to continue...\n")\
                
                for word in line.split(" "):
                    # Removes any invalid characters from word
                    # or returns empty string if word is invalid
                    word = cleanWord(word)
                    # TODO get gender
                    getWordGender(word, docNum)
                    word = getValidWord(word)
                    if word != "":
                        # Check if word is already in vocabulary
                        i = vocab.index(word) if word in vocab else -1
                        # if i >= 0:
                            # print "Check vocab: " + vocab[i] + ", " + word
                        if i >= 0:
                            # Word exists in vocab
                            # Increment the total count for the word
                            vocabCount[i] = vocabCount[i] + 1
                            # Increment word count for specific word
                            # Check if word is already in current vocabulary
                            j = np.where(cur_stats[:, 0] == (i))[0][0] if (i) in cur_stats[:, 0] else -1
                            # if j>= 0:
                                # print "Check current vocab: " + str(i) + ", " + str(cur_stats[j])
                            # print j
                            if j >= 0:
                                # Seeing word again - Count just needs to be incremented
                                # Increment the word count for current doc
                                cur_stats[j][1] = cur_stats[j][1] + 1
                            else:
                                # First time seeing word in this document
                                # Count needs to be created for this document
                                cur_stats = np.append(
                                    cur_stats, np.array([[i, 1]]), axis=0)
                                # Increment the total vocab appearance
                                vocabAppearance[i] = vocabAppearance[i] + 1
                        else:
                            # New Word - append to vocab
                            vocab.append(word)
                            vocabCount.append(1)
                            vocabAppearance.append(1)
                            cur_stats = np.append(
                                cur_stats, np.array([[len(vocab)-1, 1]]), axis=0)
        # print cur_stats[:20]
        # raw_input("Press Enter to continue...")
        return cur_stats

    except IOError:
        print "File " + filename + " does not exist"
        return NULL

def removeWords2(file_trains, minCount, minAppearance):
    print "Removing Words..."
    # Version of remove words that removes words from the vocab and all the file_trains
    # Sort the trains by id so that it is easier to remove from
    for train in file_trains:
        # print train[:5]
        train.view('int, int').sort(order=['f0'], axis=0)
        # print train[:5]
        # raw_input("Press Enter to continue...")

    numRemoved = 0
    i = 0
    while i < len(vocab):
        # print file_trains[0][:5]
        # raw_input("Press Enter to continue...")
        # print "i " + str(i) + ", len vocab " + str(len(vocab))
        if (vocabCount[i] < minCount) or (vocabAppearance[i] < minAppearance):
            # Remove the word
            # print "Removing " + vocab[i] + " at index " + str(i)
            del vocab[i]
            del vocabCount[i]
            del vocabAppearance[i]

            for j in range(len(file_trains)):
                # Check if the word is in the train
                # Remove i from the train
                # print train[:5]
                file_trains[j] = file_trains[j][(file_trains[j][:, 0] != (i + numRemoved))]
                # print train[:5]
                # raw_input("Press Enter to continue...")

            numRemoved = numRemoved + 1
        else:
            if numRemoved > 0:
                # print "Words removed: " + str(numRemoved)
                for train in file_trains:
                    for j in range(len(train)):
                        if train[j][0] >= i:
                            train[j][0] = train[j][0] - numRemoved
                numRemoved = 0
            i = i + 1

# def removeWords(train, minCount, minAppearance):
#     # We are removing words that are insignifacnt, they do not appear enougth times or in enough documents.
#     toRemove = []

#     for i in train[:, 0]:
#         if vocabCount[i] < minCount:
#             # print "Word " + vocab[i] + " only has " + str(vocabCount[i]) + " counts."
#             toRemove.append(i)
#         elif vocabAppearance[i] < minAppearance:
#             # print "Word " + vocab[i] + " only has " + str(vocabAppearance[i]) + " appearances."
#             toRemove.append(i)
#         # Word is insignifacant - remove it
#     toRemove = sorted(toRemove, reverse=True)
#     for i in toRemove:
#         train = train[(train[:, 0] != i)]
#     # raw_input("Press Enter to continue...")
#     return train

def formatTrain(train):
    # Formatting of train line
    # print train
    # raw_input("Press Enter to continue...")
    buffer = str(len(train))
    for i in range(len(train)):
        buffer = buffer + " " + str(train[i][0]) + ":" + str(train[i][1])
    # print buffer
    # At this point buffer is one line of train file
    return buffer

def writeToTrain(line):
    with open("output/train.ldac", "a") as file:
        file.write(line)
        file.write("\n")
    # print "Wrote line to train"

# def cleanVocab():
#     for i in reversed(range(len(vocab))):
#         if vocabCount[i] < minCount or vocabAppearance[i] < minAppearance:
#             vocab.pop(i)
#             vocabCount.pop(i)
#             vocabAppearance.pop(i)


def writeVocab():
    with open("output/vocab.txt", "w") as file:
        buffer = ""
        for word in vocab:
            buffer = buffer + word
            if vocab.index(word) < len(vocab) - 1:
                buffer = buffer + "\n"
        file.write(buffer)
        file.write("\n")
    # print vocab
    print "Wrote Vocab"

def getGenderStats(files):
    percentF = 0.5;
    with open("output/genders.txt", "w") as file:
        for i in range(len(countF)):
            percentF = 0.5;
            if (countF[i] + countM[i]) != 0:
                percentF = float(countF[i]) / (countF[i] + countM[i])
            # print "Doc " + str(i) + " is " + str(percentF) + " percent female."
            if (percentF > 0.7):
                file.write("Doc " + files[i] + " is mostly female (" + str(percentF*100) + "%).\n")
                # print "Doc " + str(i) + " is mostly female (" + str(percentF) + "%).\n"
            elif (percentF < 0.3):
                file.write("Doc " + files[i] + " is mostly male. (" + str((1 - percentF)*100) + "%).\n")
                # print "Doc " + str(i) + " is mostly male. (" + str(1 - percentF) + "%).\n"
            else:
                file.write("Doc " + files[i] + " is unknown.\n")
                # print "Doc " + str(i) + " is unknown.\n"
    print ("Wrote genders.")
        

def main(argv):
    numFiles = 0
    # Get command line arguments
    argc = len(argv)
    if argc < 1:
        print "Usage: input_file"
        sys.exit()

    # Allows users to specify all files in a directory
    files = [] # list of file names
    for i in range(argc):
        files.extend(glob.glob(argv[i]))


    # Clear the train file in case of old data
    clearTrain()

    # file_trains = np.empty(shape=[0,1,2], dtype=int); # this doesn't work because we do no know how many words a doc may have
    file_trains = []

    # Allowing user to specify multiple docs at the same time
    for i in range(len(files)):
        input_file = files[i]
        train = parseDoc(input_file, i)

        # file_trains = np.append(file_trains, np.array([[train_line]]), axis = 0)
        if train.any():
            file_trains.append(train)
            numFiles = numFiles + 1
            print "Parsed file: " + files[i]
        else:
            print "ERROR"
        # print file_trains
        # print "Parsed Files"
        # print len(file_trains)
    print "Parsed " + str(numFiles) + " files."

    removeWords2(file_trains, minCount, minAppearance)

    for j in range(len(file_trains)):
        # file_trains[j] = removeWords(file_trains[j], minCount, minAppearance)
        trainLine = formatTrain(file_trains[j])
        # print "Formatted train"
        # If file opened sucessfully store the word data in train file
        writeToTrain(trainLine)
    print "Wrote train"
    # Writes the discovered vocabulary to file

    writeVocab()
    print "Found " + str(len(vocab)) + " valid words."

    getGenderStats(files)


#Call the main function to start the program
# main()
if __name__ == '__main__':
    # Doing this so that main can be called from both the
    # command line as well as another script as a function call
    main(sys.argv[1:])

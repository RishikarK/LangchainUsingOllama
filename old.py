import ollama
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


# Sample text
text = """
No-show for first flight
Changes*: Not applicable (See reissue conditions)Reissue: Allowed with restrictionsPenalty fee for ticket reissue between: 29055 INR / 29055 INR (at today exchange rates 29055 INR / 29055 INR)New travel dates and change request must be prior to: Thursday, February 06, 2025Maximum Reissue penalty fee for entire ticket: 29055 INR
No-show for subsequent flight(s)
Changes*: Not applicable (See reissue conditions)Reissue: Allowed with restrictionsPenalty fee for ticket reissue between: 10795 INR / 10795 INR (at today exchange rates 10795 INR / 10795 INR)New travel dates and change request must be prior to: Wednesday, February 19, 2025Maximum Reissue penalty fee for entire ticket: 29055 INR
Prior to Departure of first flight
Changes*: Not applicable (See reissue conditions)Reissue: Allowed with restrictionsPenalty fee for ticket reissue between: 0 INR / 10795 INR (at today exchange rates 0 INR / 10795 INR)New travel dates must be prior to: Thursday, February 06, 2025Maximum Reissue penalty fee for entire ticket: 29055 INR
After departure of first flight
Changes*: Not applicable (See reissue conditions)Reissue: Allowed with restrictionsPenalty fee for ticket reissue between: 0 INR / 10795 INR (at today exchange rates 0 INR / 10795 INR)New travel dates must be prior to: Wednesday, February 19, 2025Maximum Reissue penalty fee for entire ticket: 29055 INR
"""

# Tokenize words and sentences
words = word_tokenize(text)
sentences = sent_tokenize(text)

# Remove stopwords
stopWords = set(stopwords.words("english"))
freqTable = dict()

for word in words:
    word = word.lower()
    if word in stopWords:
        continue
    if word in freqTable:
        freqTable[word] += 1
    else:
        freqTable[word] = 1

# Calculate sentence scores
sentenceValue = dict()
for sentence in sentences:
    for word, freq in freqTable.items():
        if word in sentence.lower():
            if sentence in sentenceValue:
                sentenceValue[sentence] += freq
            else:
                sentenceValue[sentence] = freq

# Calculate average score
sumValues = sum(sentenceValue.values())
average = int(sumValues / len(sentenceValue))

# Generate summary
summary = ""
for sentence in sentences:
    if sentence in sentenceValue and sentenceValue[sentence] > (1.2 * average):
        summary += " " + sentence

print(summary)

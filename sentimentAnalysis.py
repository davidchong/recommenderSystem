from textblob import TextBlob

pos_count = 0
pos_correct = 0

line = input("Text here: ")
analysis = TextBlob(line)

# detect if it is english or not
if (analysis.sentiment.polarity == 0) & (analysis.sentiment.subjectivity == 0):
    line_t = analysis.translate(to='en')

    print(str(line_t))
    print("Polarity score: " + str(line_t.sentiment.polarity))
    print("Subjectivity score: " + str(line_t.sentiment.subjectivity))

else:
    print("Polarity score: " + str(analysis.sentiment.polarity))
    print("Subjectivity score: " + str(analysis.sentiment.subjectivity))
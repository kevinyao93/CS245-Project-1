'''
The file only contains word and tagging. If we only care about LOC, the
tagging would only have LOC and for all the other things they are O
'''
with open('train.txt','r') as f:
    texts=[]
    tags=[]
    line=f.readline()
    temp_words=[]
    temp_tags=[]
    while line:
        if "-DOCSTART-" not in line and line != "\n":
            tt=line.split(" ")
            temp_words.append(tt[0])
            tt[-1]=tt[-1].rstrip()
            temp_tags.append(tt[-1])
        elif line=="\n" and temp_words!=[]:
            texts.append(" ".join(temp_words))
            tags.append(" ".join(temp_tags))
            temp_words=[]
            temp_tags=[]
        line=f.readline()
    f.close()

with open('annotated_train_sentences.txt','w') as f:
    for i in range(len(texts)):
        f.write(texts[i]+'\n')
f.close()

with open('annotated_train_tags.txt','w') as f:
    for i in range(len(tags)):
        f.write(tags[i]+'\n')
f.close()



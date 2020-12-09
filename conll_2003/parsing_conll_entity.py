#created by Zhaoyi Huang
#open the dict_core first and create a list of result
entities=[]
with open('BERT_conll_full_phrases.txt','r') as f:
    entities=f.readlines()
f.close()
for i in range(len(entities)):
    entities[i]=entities[i].rstrip()
    entities[i]=entities[i].lower()
entities=set(entities)

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
        elif line=="\n" and temp_words!=[]:
            #because the last element is always a
            #punctuation, so no need to check it
            i=0
            while i < len(temp_words):
                #trigram then bigram then unigram
                if i<len(temp_words)-2:
                    temp_trigram=(temp_words[i]+" "+temp_words[i+1]+" "+temp_words[i+2]).lower()
                    if temp_trigram in entities:
                        temp_tags.append("B-LOC")
                        temp_tags.append("I-LOC")
                        temp_tags.append("I-LOC")
                        i+=3
                        continue
                if i<len(temp_words)-1:
                    temp_bigram=(temp_words[i]+" "+temp_words[i+1]).lower()
                    if temp_bigram in entities:
                        temp_tags.append("B-LOC")
                        temp_tags.append("I-LOC")
                        i+=2    
                        continue
                if temp_words[i].lower() in entities:
                    temp_tags.append("B-LOC")
                    i+=1
                else:
                    temp_tags.append("O")
                    i+=1   
            assert(len(temp_words)==len(temp_tags))
            texts.append(" ".join(temp_words))
            tags.append(" ".join(temp_tags))
            
            temp_words=[]
            temp_tags=[]
        line=f.readline()
f.close()

#create setneces file 
with open('train_new_bert_sentences.txt','w') as f:
    for i in range(len(texts)):
        f.write(texts[i]+'\n')
f.close()

#create tags to put in the file
with open('train_new_bert_tags.txt','w') as f:
    for i in range(len(tags)):
        f.write(tags[i]+'\n')
f.close()





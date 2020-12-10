tags=[]
sentences=[]
entities=[]
with open('./all_train_sentences/train_sentences_broad.txt','r') as f:
    sentences=f.readlines()
f.close()

with open('./data/expanded_candidate.txt','r') as f:
    entities=f.readlines()
f.close()
for i in range(len(entities)):
    entities[i]=entities[i].rstrip()
    entities[i]=entities[i].lower()
entities=set(entities)

for a in range(len(sentences)):
    sent=sentences[a].strip().split()
    temp_tags=[]
    i=0
    while i<len(sent):
        if i<len(sent)-2:
            temp_trigram=(sent[i]+" "+sent[i+1]+" "+sent[i+2]).lower()
            if temp_trigram in entities:
                temp_tags.append("B-LOC")
                temp_tags.append("I-LOC")
                temp_tags.append("I-LOC")
                i+=3
                continue
        if i<len(sent)-1:
            temp_bigram=(sent[i]+" "+sent[i+1]).lower()
            if temp_bigram in entities:
                temp_tags.append("B-LOC")
                temp_tags.append("I-LOC")
                i+=2    
                continue
        if sent[i].lower() in entities:
            temp_tags.append("B-LOC")
            i+=1
        else:
            temp_tags.append("O")
            i+=1   
    assert(len(sent)==len(temp_tags))
    tags.append(" ".join(temp_tags))
    temp_tags=[]

with open("train_broad_tag_by_entity.txt",'w') as f:
    for i in range(len(tags)):
        f.write(tags[i]+'\n')
f.close()

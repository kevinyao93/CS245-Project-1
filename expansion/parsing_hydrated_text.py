#made by Zhaoyi Huang
import re
import zipfile
tweets=[]
my_zip = zipfile.ZipFile("./all_train_sentences/2020-03-22_clean-hydrated.txt.zip", "r")
storage_path = '.'
my_zip.extract("2020-03-22_clean-hydrated.txt", storage_path)
with open ("2020-03-22_clean-hydrated.txt",'r') as f:
    tweets=f.readlines()
f.close()

# def find_emoji(text):
#     EMOJI_PATTERN = re.compile(
#     "["
#     "\U0001F1E0-\U0001F1FF"  # flags (iOS)
#     "\U0001F300-\U0001F5FF"  # symbols & pictographs
#     "\U0001F600-\U0001F64F"  # emoticons
#     "\U0001F680-\U0001F6FF"  # transport & map symbols
#     "\U0001F700-\U0001F77F"  # alchemical symbols
#     "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
#     "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
#     "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
#     "\U0001FA00-\U0001FA6F"  # Chess Symbols
#     "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
#     "\U00002702-\U000027B0"  # Dingbats
#     "\U000024C2-\U0001F251" 
#     "]+"
# )
#     return EMOJI_PATTERN.search(text)

# for i in range(len(tweets)-1,-1,-1):
#     if find_emoji(tweets[i])!=None:
#         tweets.pop(i)

#remove all http from the twitter--> as they are useless
for i in range(len(tweets)):
	try:
		j=tweets[i].index("http")
		tweets[i]=tweets[i][:j]
	except:
		continue

for i in range(len(tweets)):
    tweets[i]=re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", tweets[i])

for i in range(len(tweets)):
    tweets[i]=tweets[i].rstrip()

tweets = list(filter(None, tweets))
    


#now do the taggings, using the expanded entity set
tags=[]
with open('./data/expanded_candidate.txt','r') as f:
    entities=f.readlines()
f.close()
for i in range(len(entities)):
    entities[i]=entities[i].rstrip()
    entities[i]=entities[i].lower()
entities=set(entities)

print(tweets[0].split(" "))
for a in range(len(tweets)):
    sent=tweets[a].strip().split()
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

with open("train_twitter_tags_0322.txt",'w') as f:
    for i in range(len(tags)):
        f.write(tags[i]+'\n')
f.close()

with open('train_twitter_sentences_0322.txt','w') as f:
     for i in range(len(tweets)):
         f.write(tweets[i]+'\n')
f.close()
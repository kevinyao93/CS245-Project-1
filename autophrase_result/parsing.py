#made by:Zhaoyi Huang
import os
arr = next(os.walk('.'))[2]

for file in arr:
    if len(file.split("_"))==2 and file!=".DS_Store":
        generation=file.split("_")[1]
        phrases=[]
        with open(file,'r') as f:
            line=f.readline()
            while line:
                temp=line.split("	")
                if len(temp[1].split(" "))<=2:
                    phrases.append(temp[1])
                line=f.readline()
        f.close()
        with open(generation,'w') as new:
            for item in phrases:
                new.write("%s" % item)
        print("generating now")
        new.close()
# phrases=[]
# with open('AutoPhrase_conll.txt','r') as f:
#     line=f.readline()
#     while line:
#         temp=line.split("	")
#         if len(temp[1].split(" "))<=3:
#             phrases.append(temp[1])
#         line=f.readline()
# f.close()
# print(len(phrases))
# print(phrases[0])
# with open('conll_list.txt','w') as file:
#     for item in phrases:
#         file.write("%s" % item)
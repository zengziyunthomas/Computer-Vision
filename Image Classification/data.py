import os

CLASS_NUM = 256
TRAIN_NUM = 15

directions = os.listdir("256_ObjectCategories")
directions.sort()

file_list=[]
for i in range(256):
    file_list.append([])

count = 0
for i in directions:
    for _,_,j in os.walk(os.path.join("256_ObjectCategories",i)):
        for k in j:
            file_list[count].append(os.path.join(os.path.join("256_ObjectCategories",i),k))
    count+=1
# get data for train
with open(r'train_15.txt','w',encoding='utf-8') as f:
    for i in range(CLASS_NUM):
        for j in range(TRAIN_NUM):
            f.write('./'+file_list[i][j]+' '+str(i)+'\n')

# get data for test
with open(r'test_15.txt','w',encoding='utf-8') as f:
    for i in range(CLASS_NUM):
        for j in range(TRAIN_NUM,len(file_list[i])):
            f.write('./'+file_list[i][j]+' '+str(i)+'\n')

# get the label of each data
# with open(r'label_60.txt','w',encoding='utf-8') as f:
#     for i in range(CLASS_NUM):
#         f.write(str(i+1)+'\n')

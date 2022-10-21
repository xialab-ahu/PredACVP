
inf1 = (r"./data/relgen.fasta")
inf = (r"./data/acvp-test.fasta")
def readwrite1(file):
    f=open(file,'r')
    pos = []
    i = 1
    for line in f.readlines():
        if i % 2 == 0:
            pos.append(line.strip())
        i = i + 1
    return pos
pos = readwrite1(inf)
# print(pos)
print(len(pos))
pos1 = readwrite1(inf1)
# print(len(pos1))
print(len(pos1))
import numpy as np
import pandas as pd
temp = []
temp1 = []
list2=[]
for i in pos:
    if not i in pos1:
        list2.append(i)
print(len(list2))
list2 = pd.DataFrame(list2,columns=None)
print(list2)
# list2.to_csv('data/test/testneg.txt',index=None,columns=None)
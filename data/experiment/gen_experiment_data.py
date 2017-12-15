import collections
import random
# randomly shuffle data
ins = open("instructions.txt", 'r')
com = open("commands.txt", 'r')

Data = [ line for line in ins ]
Label = [ line for line in com ]

e1 = list(zip(Data[0:177], Label[0:177]))
e2 = list(zip(Data[0:30], Label[0:30]))
e3 = list(zip(Data[178:278], Label[178:278]))

random.shuffle(e1)

data, label = zip(*e1)

with open('e1_data.txt','w') as dataRand:
    for line in data:
        dataRand.write( line )

with open('e1_label.txt','w') as labelRand:
    for line in label:
        labelRand.write( line )

random.shuffle(e2)

data, label = zip(*e2)

with open('e2_data.txt','w') as dataRand:
    for line in data:
        dataRand.write( line )

with open('e2_label.txt','w') as labelRand:
    for line in label:
        labelRand.write( line )

random.shuffle(e3)

data, label = zip(*e3)

with open('e3_data.txt','w') as dataRand:
    for line in data:
        dataRand.write( line )

with open('e3_label.txt','w') as labelRand:
    for line in label:
        labelRand.write( line )

ins.close()
com.close()
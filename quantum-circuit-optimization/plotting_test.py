import os

import plotting
from collections import defaultdict


# V = [(10,12,True),(10,11,True),(10,13,True), (10,14,True),(10,15,False),(10,16,False),(10,17,False),(10,18,False),
#      (11,12,True),(11,19,False),(11,13,True), (11,14,True),(11,15,False),(11,16,False),(11,17,False),(11,18,False),
#      (12,19,False),(12,21,False),(12,13,True), (12,14,True),(12,15,False),(12,16,False),(12,17,False),(12,18,False),
#      (13,19,False),(13,21,False),(13,23,False), (13,14,True),(13,15,False),(13,16,False),(13,17,False),(13,18,False),
#      (14,19,False),(14,21,False),(14,23,False), (14,24,False),(14,15,False),(14,16,False),(14,17,False),(14,18,False),
#      (15,19,True),(15,21,False),(15,23,False), (15,24,False),(15,25,False),(15,16,False),(15,17,False),(15,18,False),
#      (16,19,True),(16,21,False),(16,23,False), (16,24,False),(16,25,False),(16,26,False),(16,17,False),(16,18,False),
#      (17,19,True),(17,21,True),(17,23,False), (17,24,False),(17,25,False),(17,26,False),(17,27,False),(17,18,False),
#      (18,19,True),(18,21,True),(18,23,False), (18,24,False),(18,25,False),(18,26,False),(18,27,False),(18,28,False),
#      (19,22,True),(19,21,True),(19,23,True), (19,24,False),(19,25,False),(19,26,False),(19,27,False),(19,28,False)]
from save_data import load_object

directory = 'generated_circuits/'
V = []

for filename in sorted(os.listdir(directory)):
    if 'circuit.pickle_' in filename:
        cir = load_object(directory+filename)
        V.append((9,len(cir),True))
    elif 'circuit1.pickle_' in filename:
        cir = load_object(directory+filename)
        V.append((6,len(cir),True))
    elif 'circuit2.pickle_' in filename:
        cir = load_object(directory+filename)
        V.append((12,len(cir),True))
    elif 'circuit3.pickle_' in filename:
        cir = load_object(directory+filename)
        V.append((15,len(cir),True))
    elif 'circuit4.pickle_' in filename:
        cir = load_object(directory+filename)
        V.append((18,len(cir),True))

print(V)
P = defaultdict(float)

for i in V:
     P[i] = i[1]-i[0]
print(P)

plotting.plot_value_function(P, title="Circuit Depth of 5 circuits")

import argparse
import re
import csv

path = "C:/Users/jiali/Downloads/s-pectra-master/s-pectra-master/cfm_pair.csv"
f = open(path, 'r')

weight = []
molecule = []
intensity = []
energy = [0,1,2]
##molecule = {k:[] for k in mol}

f.readline()
lines = f.readlines()

for line in lines:
    mol = line.strip().split(',')[0]
    ins = line.strip().split(',')[3]
    we = line.strip().split(',')[4]
    if mol not in molecule:
        molecule.append(mol)
    if we not in weight:
        weight.append(we)
    if ins not in intensity:
        intensity.append(ins)

num_molecule = 4
ins_tuple = {k:[] for k in range(num_molecule*3)}  
wei_tuple = {k:[] for k in range(num_molecule*3)}        
for line in lines:
    mo = line.strip().split(',')[0]
    en = line.strip().split(',')[1]
    ins = line.strip().split(',')[3]
    we = line.strip().split(',')[4]

    for num_mol,mol in enumerate(molecule):
        for num_ene,ene in enumerate(energy):
            if mo == mol and en == str(ene):
                ins_tuple[3*num_mol+num_ene].append(ins)
                wei_tuple[3*num_mol+num_ene].append(we)
print(ins_tuple)
print(wei_tuple)
num_ins = len(ins_tuple[0])
for i in range(num_molecule*3):

    for j in range(num_ins-1):

        for we in weight:
            if wei_tuple[i][j] < we < wei_tuple[i][j+1]:
                wei_tuple[i].insert( j,we )
                ins_tuple[i].insert( j,'0' )
print(ins_tuple)
print(wei_tuple)
##plt.plot(t, s)
##plt.show()
f.close()


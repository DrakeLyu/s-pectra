
import argparse
import re
import csv


parser = argparse.ArgumentParser()
parser.add_argument('-u')
parser.add_argument('-m')
parser.add_argument('-o')

args = parser.parse_args()

uniprot_member = []
molecule_member = []
with open(args.u, 'r',newline='') as f:
    f.readline()
    for line in f.readlines():
        data = re.split(',',line)
        uniprot = data[6].strip('\n').strip('\r')
        if uniprot not in uniprot_member:
            uniprot_member.append( uniprot )


uniprot_data = {k:[] for k in uniprot_member}
with open(args.u, 'r',newline='') as f:
    f.readline()
    for line in f.readlines():    
        data = re.split(',',line)
        uniprot = data[6].strip('\n').strip('\r')
        molecule = data[0]
        uniprot_data[uniprot].append(molecule)
        if molecule not in molecule_member:
            molecule_member.append( molecule )

molecule_data = {k:[] for k in molecule_member}
with open(args.m, 'r',newline='') as f:
    f.readline()
    for line in f.readlines():    
        data = re.split(',',line)
        fragment = data[0]
        molecule = data[2]
        if molecule in molecule_member:
            molecule_data[molecule].append(fragment)

all_data = ['uniprot,molecule,fragment']
for i in list(uniprot_data.keys()):
    for j in uniprot_data[i]:
        one_data = []
        one_data.append(i)
        one_data.append(j)
        for k in molecule_data[j]:
            one_data.append(k)
        str_one_data = ','.join(one_data)
        all_data.append(str_one_data)   

with open(args.o,'wt',newline='') as f:
    f_csv = csv.writer(f,delimiter=',')
    for i in all_data:
        f_csv.writerow(i.split(','))


print("Completed")



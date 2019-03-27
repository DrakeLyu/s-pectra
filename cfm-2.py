
import pandas as pd
import argparse
import re
import csv


parser = argparse.ArgumentParser()
parser.add_argument('-uniprot')
parser.add_argument('-molecule')
parser.add_argument('-output')

args = parser.parse_args()

uniprot_member = []
molecule_member = []
with open('./test_uniprot.csv', 'r',newline='') as f:
    f.readline()
    for line in f.readlines():
        data = re.split(',',line)
        uniprot = data[6].strip('\n').strip('\r')
        if uniprot not in uniprot_member:
            uniprot_member.append( uniprot )


uniprot_data = {k:[] for k in uniprot_member}
with open('./test_uniprot.csv', 'r',newline='') as f:
    f.readline()
    for line in f.readlines():    
        data = re.split(',',line)
        uniprot = data[6].strip('\n').strip('\r')
        molecule = data[0]
        uniprot_data[uniprot].append(molecule)
        if molecule not in molecule_member:
            molecule_member.append( molecule )

molecule_data = {k:[] for k in molecule_member}
with open('./test_molecule_fragment.csv', 'r',newline='') as f:
    f.readline()
    for line in f.readlines():    
        data = re.split(',',line)
        fragment = data[0]
        molecule = data[2]
        if molecule in molecule_member:
            molecule_data[molecule].append(fragment)

all_data = []
for i in list(uniprot_data.keys()):
    all_data.append(i)
    all_data.append(uniprot_data[i])
    for j in uniprot_data[i]:
        all_data.append(molecule_data[j])
        all_data.append('\n')
print(all_data)
df_pair = pd.DataFrame()
df_mol = pd.DataFrame(all_data)
df_mol.to_csv('./new_mol.csv', index=False)

print("Completed")



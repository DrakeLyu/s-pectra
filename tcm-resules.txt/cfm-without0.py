import os, sys
from os.path import isfile, join
import pickle
import pandas as pd
from collections import defaultdict


class Mol:
    def __init__(self, name):
        self.name = name
        self.fragments = []
        self.energy_bands = []  # energy band 0, 1, 2


class EnergyEntry:
    def __init__(self, index=0, intensity=0.0):
        self.frag_local_index = index
        self.intensity = intensity


class Fragment:
    def __init__(self, s="", w=0.0, lid=0, gid=0):
        self.smile = s
        self.weight = w
        self.local_id = lid
        self.global_id = gid


class CFMProcessor:
    frag_file = "fragment_gid.pk"

    def __init__(self, path):
        self.path = path
        self.mols = []
        self.frag_data = defaultdict(list)
        self.fragment_gid = {}
        if os.path.exists(self.frag_file):
            f = open(self.frag_file, 'rb')
            self.fragment_gid = pickle.load(f)
            f.close()

    def read_file(self):
        filenames = [f for f in os.listdir(self.path) if isfile(join(self.path, f))]
        for file in filenames:
            f = open(join(self.path, file), 'r')
            lines = f.readlines()
            index = lines.index('\n')
            rear_part = lines[index+1:]
            front_part = lines[:index]
            rear_part = [line.strip().split(' ') for line in rear_part]
            index = [i for (i, line) in enumerate(front_part) if line[0] == 'e']
            front_part = [front_part[1:index[1]], front_part[index[1]+1: index[2]], front_part[index[2]+1:]]

            mol = Mol(file.split('.')[0])
            mol.fragments = [Fragment(line[2], float(line[1]), int(line[0])) for line in rear_part]
            self.update(mol.fragments)
            
            for part in front_part:
                e = [line.strip().split(' ') for line in part]
                EnergyEntry_list = []
                for line in e:
                    for i in range(1,int(len(line)/2-1)+1):  # to add with same weight but different intensity fragment
                        EnergyEntry_list.append(EnergyEntry(int(line[1+i]),float(line[1+i+int(len(line)/2-1)].lstrip('(').strip(')'))))
                mol.energy_bands.append(EnergyEntry_list)
                #mol.energy_bands.append([EnergyEntry(int(line[2]), float(line[1])) for line in e])

            self.mols.append(mol)
            f.close()

    def update(self, fragments):
        for frag in fragments:
            if frag.local_id == 0:  # filter NO. 0 fragment
                continue
            gid = self.fragment_gid.get(frag.smile)
            if gid is not None:
                frag.global_id = gid
            else:
                frag.global_id = len(self.fragment_gid) + 1
                self.fragment_gid[frag.smile] = frag.global_id
                self.frag_data['fragment_id'].append(frag.global_id)
                self.frag_data['smile'].append(frag.smile)
                self.frag_data['weight'].append(frag.weight)

    def write_file(self):
        if len(self.frag_data) > 0:
            df_frag = pd.DataFrame(self.frag_data)
            df_frag.to_csv('./new_fragment1.csv', index=False)
            f = open(self.frag_file, 'wb')
            pickle.dump(self.fragment_gid, f)
            f.close()

        df_pair = pd.DataFrame()
        keys = ['mol', 'energy', 'fragment_id', 'intensity', 'weight']
        for mol in self.mols:

            mol_data = {k: [] for k in keys}
            for i, band in enumerate(mol.energy_bands):
                for entry in band:
                    if entry.frag_local_index == 0:  # filter NO. 0 fragment
                        continue
                    mol_data['mol'].append(mol.name)
                    mol_data['energy'].append(i)
                    frag = mol.fragments[entry.frag_local_index]
                    mol_data['fragment_id'].append(frag.global_id)
                    mol_data['intensity'].append(entry.intensity)
                    mol_data['weight'].append(frag.weight)
            df_pair = df_pair.append(pd.DataFrame(mol_data, columns=keys), ignore_index=True)
        df_pair.to_csv('./cfm_pair1.csv', index=False)

        mol_data.clear()
        mol_data['mol_name'] = [mol.name for mol in self.mols]
        df_mol = pd.DataFrame(mol_data)
        df_mol.to_csv('./new_mol1.csv', index=False)


def main():
    if len(sys.argv) <= 1:
        print("please give the file path as argument")
        path = './hmdb'
    else:
        path = sys.argv[1]
    processor = CFMProcessor(path)
    processor.read_file()
    processor.write_file()

if __name__ == '__main__':
    main()

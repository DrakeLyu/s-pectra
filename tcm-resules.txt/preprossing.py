import argparse
import re
import csv
import  matplotlib.pyplot as plt
import random
path = "D:/spectra/nn/cfm_pair1.csv"
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
    ins = float(line.strip().split(',')[3])
    we = float(line.strip().split(',')[4])
    if mol not in molecule:
        molecule.append(mol)
    if we not in weight:
        weight.append(we)
    if ins not in intensity:
        intensity.append(ins)


num_molecule = len(molecule)
num_energy = len(energy)
ins_tuple = {k:[] for k in range(num_molecule*3)}  
wei_tuple = {k:[] for k in range(num_molecule*3)}        
for line in lines:
    mo = line.strip().split(',')[0]
    en = int(line.strip().split(',')[1])
    ins = float(line.strip().split(',')[3])
    we = float(line.strip().split(',')[4])

    for num_mol,mol in enumerate(molecule):
        for num_ene,ene in enumerate(energy):
            if mo == mol and en == int(ene):
                ins_tuple[3*num_mol+num_ene].append(ins)
                wei_tuple[3*num_mol+num_ene].append(we)

weight.sort()
ins_all = []

for i in range(num_molecule*num_energy):
    ins = []
    for j in range(len(weight)):
        if weight[j] in wei_tuple[i]:
            k = wei_tuple[i].index(weight[j])
            ins.append( ins_tuple[i][k] )
            #print(ins_tuple[i][k])
        else:
            ins.append( 0 )
    
    ins_all.append(ins)


##fig, vax = plt.subplots(1, 1, figsize=(12, 6))
##
##vax.vlines( wei_tuple[10],[0], ins_tuple[10])
##plt.show()

first_row = ['molecule']
n=0
while(n<3):
    for i in weight:
        first_row.append(i)
    n+=1

with open('./out.csv','wt',newline='') as f:

    f_csv = csv.writer(f,delimiter=',')
    f_csv.writerow(first_row)

    for i in range(len(ins_all)):
        if i % 3 == 0:
            info =[]
            info.append( molecule[i//3])
            for j in ins_all[i]:
                info.append(j)
            for j in ins_all[i+1]:
                info.append(j)
            for j in ins_all[i+2]:
                info.append(j)
            f_csv.writerow( info )

with open('./out.csv','r',newline='') as f:
    f.readline()
    X = f.readlines()


x = []
y = []
data_num = 800
for m in range(data_num//2):
    ran1 = random.randint(0,len(X)-1) 

    num1 = random.random()  
 
    x1 = X[ran1].strip('\r\n').split(',')

    x_list = []
    y_list = []
    for i in range(1,len(x1)):
        x_list.append(float(x1[i])*num1)
    y_list.append(x1[0])
    
    y.append(y_list)
    x.append(x_list)
    
for m in range(data_num):
    ran1 = random.randint(0,len(X)-1) 
    ran2 = random.randint(0,len(X)-1) 

    num1 = random.random()  
    num2 = random.random()  
 
    x1 = X[ran1].strip('\r\n').split(',')
    x2 = X[ran2].strip('\r\n').split(',')
    x_list = []
    y_list = []
    for i in range(1,len(x1)):
        x_list.append(float(x1[i])*num1+float(x1[i])*num1)
    y_list.append(x1[0])
    y_list.append(x2[0])
    y.append(y_list)
    x.append(x_list)
    
for m in range(data_num):
    ran1 = random.randint(0,len(X)-1) 
    ran2 = random.randint(0,len(X)-1)
    ran3 = random.randint(0,len(X)-1)
    num1 = random.random()
    num2 = random.random()  
    num3 = random.random() 

    x1 = X[ran1].strip('\r\n').split(',')
    x2 = X[ran2].strip('\r\n').split(',')
    x3 = X[ran3].strip('\r\n').split(',')
    x_list = []
    y_list = []
    for i in range(1,len(x1)):
        x_list.append(float(x1[i])*num1+float(x2[i])*num2+float(x3[i])*num3)
    y_list.append(x1[0])
    y_list.append(x2[0])
    y_list.append(x3[0])

    y.append(y_list)
    x.append(x_list)

for m in range(data_num):
    ran1 = random.randint(0,len(X)-1) 
    ran2 = random.randint(0,len(X)-1)
    ran3 = random.randint(0,len(X)-1)
    ran4 = random.randint(0,len(X)-1)
    num1 = random.random()
    num2 = random.random()  
    num3 = random.random() 
    num4 = random.random()
    
    x1 = X[ran1].strip('\r\n').split(',')
    x2 = X[ran2].strip('\r\n').split(',')
    x3 = X[ran3].strip('\r\n').split(',')
    x4 = X[ran4].strip('\r\n').split(',')
    x_list = []
    y_list = []
    for i in range(1,len(x1)):
        x_list.append(float(x1[i])*num1+float(x2[i])*num2+float(x3[i])*num3+float(x4[i])*num4)
    y_list.append(x1[0])
    y_list.append(x2[0])
    y_list.append(x3[0])
    y_list.append(x4[0])
    
    y.append(y_list)
    x.append(x_list)

with open('./x_mix.csv','wt',newline='') as f:
    f_csv = csv.writer(f,delimiter=',')
    for i in x:
        f_csv.writerow(i)
        
with open('./y_mix.csv','wt',newline='') as f:
    f_csv = csv.writer(f,delimiter=',')
    for i in y:
        f_csv.writerow(i)    
f.close()


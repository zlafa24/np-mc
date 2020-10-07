import pickle

mol_dict = pickle.load(open('molecule_dict.pickle','rb'))
for key,mol in mol_dict.items():
    print(mol.molID)
    mol.impropers = []
       
with open('molecule_dict.pickle','wb') as file:     
    pickle.dump(mol_dict,file)
  
mol_dict = pickle.load(open('molecule_dict.pickle','rb'))
for key,mol in mol_dict.items():
    print(mol.molID,mol.impropers)

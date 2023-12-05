import  os
import cv2
import numpy as np
import torch
import rdkit.Chem as Chem
from rdkit import Chem
from Filter import dataset
from Filter.constants import RGROUP_SYMBOLS, ABBREVIATIONS
import zipfile
from Filter import MolScribe
def atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol
device = torch.device('cuda')
path = './logs/chemseg_pix_sdg_2023_09_08_19_02_14/Model/test.pth'
filter = MolScribe(path, device)
filter.decoder.compute_confidence = True
def filter_stucture(list_png):
    os.makedirs('not_bonds',exist_ok=True)
    os.makedirs('one_bonds',exist_ok=True)
    os.makedirs('overlap',exist_ok=True)
    os.makedirs('letter',exist_ok=True)
    for imf in list_png:
        image = cv2.imread(imf)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        abl_transform=dataset.get_transforms(384, augment=False)
        images=[abl_transform(image=image, keypoints=[])['image']]
        images = torch.stack(images, dim=0).to(device)
        features, hiddens=filter.encoder(images)
        batch_predictions = filter.decoder.decode(features)
        smiles=batch_predictions[0]['chartok_coords']['smiles']
        coords, symbols,atom_scores=(batch_predictions[0]['chartok_coords']['coords'],batch_predictions[0]['chartok_coords']['symbols'],batch_predictions[0]['chartok_coords']['atom_scores'])
        edges=batch_predictions[0]['edges']
        mol = Chem.RWMol()
        n = len(symbols)
        ids = []
        for i in range(n):
            symbol = symbols[i]
            if symbol[0] == '[':
                symbol = symbol[1:-1]
            if symbol in RGROUP_SYMBOLS:
                atom = Chem.Atom("*")
                if symbol[0] == 'R' and symbol[1:].isdigit():
                    atom.SetIsotope(int(symbol[1:]))
                Chem.SetAtomAlias(atom, symbol)
            elif symbol in ABBREVIATIONS:
                atom = Chem.Atom("*")
                Chem.SetAtomAlias(atom, symbol)
            else:
                try:  
                    atom = Chem.AtomFromSmiles(symbols[i])
                    atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
                except:  
                    atom = Chem.Atom("*")
                    Chem.SetAtomAlias(atom, symbol)
            if atom.GetSymbol() == '*':
                atom.SetProp('molFileAlias', symbol)
            idx = mol.AddAtom(atom)
            assert idx == i
            ids.append(idx)
        for i in range(n):
            for j in range(i + 1, n):
                if edges[i][j] == 1:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                elif edges[i][j] == 2:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.DOUBLE)
                elif edges[i][j] == 3:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.TRIPLE)
                elif edges[i][j] == 4:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.AROMATIC)
                elif edges[i][j] == 5:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINWEDGE)
                elif edges[i][j] == 6:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINDASH)
        pred_smiles = '<invalid>'
        id_bonds=[(i, bond) for i, bond in enumerate(mol.GetBonds())]
        if not (np.array(edges)>0).any():
            os.system("mv -f "+imf+" not_bonds")
            print(f'no bonds in this molecule ???, not molecule@@\n{imf}')
        x_ys={f"{x},{y}" for x,y in coords}
        ys={f"{y}" for x,y in coords}
        xs={f"{x}" for x,y in coords}
        sy_unique={}
        if len(set(symbols))==1:
            if len(coords) - len(x_ys)>=3:
                os.system("mv -f "+imf+" overlap")
                print(f'only atom C this molecule with 3 overlay ???, not molecule@@\n{imf}')
        if len(set(xs))<=3 and len(set(ys))<=3:
            os.system("mv -f "+imf+" letter")
            print(f'coordY:just{set(ys)} lines in this molecule ???, may be not molecule at all')
        if len(id_bonds)==1:
            os.system("mv -f "+imf+" one_bonds")
            print(f'just 1 bond in this molecule ???, not molecule@@\n{imf}')
def main(data_path):
    img_list = [os.path.join(data_path, img) for img in os.listdir(data_path) if img.endswith(".png")]
    filter_stucture(img_list[:])
if __name__ == '__main__':
    data_path = './testdata/'
    main(data_path)
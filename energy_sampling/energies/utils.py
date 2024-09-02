
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints, rdMolTransforms
from rdkit.Geometry.rdGeometry import Point3D
import numpy as np
import torch

def conformations_to_torsions(xyz, tas, rd_conf):
    # energies = self.model((an_bs, xyz.reshape(-1, self.data_ndim//3, 3))).energies
    torsions = []
    for conf in xyz:
        rd_conf.set_atom_positions(conf)
        tas = torch.tensor(rd_conf.get_freely_rotatable_tas_values())
        torsions.append(tas)
    torsions = torch.stack(torsions)
    return torsions

def bas_bls_to_conformations(xyz, bonds, bas, rd_conf, device):
    confs = []
    for transformation_set in xyz:
        rotation_set = transformation_set[:len(bas)]
        bond_lengths = transformation_set[len(bas):]
        for i, bond_angle in enumerate(bas):
            rd_conf.set_bond_angle(bond_angle, rotation_set[i])
        rd_conf.set_bond_lengths(bond_lengths, bonds)
        xyz = torch.tensor(rd_conf.get_atom_positions()).to(device, dtype=torch.float32)
        confs.append(xyz)
    confs = torch.stack(confs)
    return confs

def torsions_to_conformations(xyz, tas, rd_conf, device):
    # energies = self.model((an_bs, xyz.reshape(-1, self.data_ndim//3, 3))).energies
    confs = []
    for transformation_set in xyz:
        rotation_set = transformation_set[:len(tas)]
        # bond_lengths = transformation_set[len(tas):]
        for i, torsion in enumerate(tas):
            rd_conf.set_torsion_angle(torsion, rotation_set[i])
        # rd_conf.set_bond_lengths(bond_lengths, bonds)
        xyz = torch.tensor(rd_conf.get_atom_positions()).to(device, dtype=torch.float32)
        confs.append(xyz)
    confs = torch.stack(confs)
    return confs

def get_torsion_angles_atoms_list(mol):
    return [x[0][0] for x in TorsionFingerprints.CalculateTorsionLists(mol)[0]]

def get_torsion_angles_values(conf, torsion_angles_atoms_list):
    return [
        torch.tensor(rdMolTransforms.GetDihedralRad(conf, *ta))
        for ta in torsion_angles_atoms_list
    ]

def get_all_torsion_angles(mol, conf):
    ta_atoms = get_torsion_angles_atoms_list(mol)
    ta_values = get_torsion_angles_values(conf, ta_atoms)
    return {k: v for k, v in zip(ta_atoms, ta_values)}

def embed_mol_and_get_conformer(mol, extra_opt=False):
    """Embed RDkit mol with a conformer and return the RDKit conformer object
    (which is synchronized with the RDKit molecule object)
    :param mol: rdkit.Chem.rdchem.Mol object defining the molecule
    :param extre_opt: bool, if True, an additional optimisation of the conformer will be performed
    """
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    if extra_opt:
        AllChem.MMFFOptimizeMolecule(mol, confId=0, maxIters=1000)
    return mol.GetConformer()

def is_hydrogen_ta(mol, ta):
    """
    Simple check whether the given torsion angle is 'hydrogen torsion angle', i.e.
    it effectively influences only positions of some hydrogens in the molecule
    """

    def is_connected_to_three_hydrogens(mol, atom_id, except_id):
        atom = mol.GetAtomWithIdx(atom_id)
        neigh_numbers = []
        for n in atom.GetNeighbors():
            if n.GetIdx() != except_id:
                neigh_numbers.append(n.GetAtomicNum())
        neigh_numbers = np.array(neigh_numbers)
        return np.all(neigh_numbers == 1)

    first = is_connected_to_three_hydrogens(mol, ta[1], ta[2])
    second = is_connected_to_three_hydrogens(mol, ta[2], ta[1])
    return first or second

def remove_duplicate_tas(tas_list):
    """
    Remove duplicate torsion angles from a list of torsion angle tuples.

    Args
    ----
    tas_list : list of tuples
        A list of torsion angle tuples, each containing four values:
        (atom1, atom2, atom3, atom4).

    Returns
    -------
    list of tuples: A list of unique torsion angle tuples, where duplicate angles have been removed.
    """
    tas = np.array(tas_list)
    clean_tas = []
    considered = []
    for row in tas:
        begin = row[1]
        end = row[2]
        if not (begin, end) in considered and not (end, begin) in considered:
            if begin > end:
                begin, end = end, begin
            duplicates = tas[np.logical_and(tas[:, 1] == begin, tas[:, 2] == end)]
            duplicates_reversed = tas[
                np.logical_and(tas[:, 2] == begin, tas[:, 1] == end)
            ]
            duplicates_reversed = np.flip(duplicates_reversed, axis=1)
            duplicates = np.concatenate([duplicates, duplicates_reversed], axis=0)
            assert duplicates.shape[-1] == 4
            duplicates = duplicates[
                np.where(duplicates[:, 0] == duplicates[:, 0].min())[0]
            ]
            clean_tas.append(duplicates[np.argmin(duplicates[:, 3])].tolist())
            considered.append((begin, end))

    return clean_tas

def get_rotatable_ta_list(mol):
    """
    Find unique rotatable torsion angles of a molecule. Torsion angle is given by a tuple of adjacent atoms'
    indices (atom1, atom2, atom3, atom4), where:
    - atom2 < atom3,
    - atom1 and atom4 are minimal among neighbours of atom2 and atom3 correspondingly.

    Torsion angle is considered rotatable if:
    - the bond (atom2, atom3) is a single bond,
    - none of atom2 and atom3 are adjacent to a triple bond (as the bonds near the triple bonds must be fixed),
    - atom2 and atom3 are not in the same ring.

    Args
    ----
    mol : RDKit Mol object
        A molecule for which torsion angles need to be detected.

    Returns 
    -------
    list of tuples: A list of unique torsion angle tuples corresponding to rotatable bonds in the molecule.
    """
    torsion_pattern = "[*]~[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]~[*]"
    substructures = Chem.MolFromSmarts(torsion_pattern)
    torsion_angles = remove_duplicate_tas(list(mol.GetSubstructMatches(substructures)))
    nonring, ring = TorsionFingerprints.CalculateTorsionLists(mol, )
    def collect_4tuples(output):
        result = []
        for item in output:
            tuples_list, _ = item
            result.extend(tuples_list)
        return result

    ring_tas = collect_4tuples(ring)
    nonring_tas = collect_4tuples(nonring)
    # collect bonds from the ring torsion angles (atom1, atom2, atom3, atom4) -> [(atom1, atom2), (atom2, atom3), (atom3, atom4)]
    ring_bonds = [(ta[0], ta[1]) for ta in ring_tas] + [(ta[1], ta[2]) for ta in ring_tas] + [(ta[2], ta[3]) for ta in ring_tas]
    nonring_bonds = [(ta[1], ta[2]) for ta in nonring_tas] + [(ta[2], ta[3]) for ta in nonring_tas] + [(ta[0], ta[1]) for ta in nonring_tas]

    # add also inverse bonds
    ring_bonds.extend([(b[1], b[0]) for b in ring_bonds])
    nonring_bonds.extend([(b[1], b[0]) for b in nonring_bonds])

    torsion_angles.extend(nonring_tas)
    torsion_angles = remove_duplicate_tas(torsion_angles)
    #torsion_angles = [ta for ta in torsion_angles if not is_hydrogen_ta(mol, ta)]
    print('Torsion Angles:', torsion_angles)
    return torsion_angles, ring_bonds, nonring_bonds

def find_bond_angles(bonds):
    # add all reverse bonds to bonds
    bonds = list(bonds)
    for bond in list(bonds):
        bonds.append((bond[1], bond[0]))
    triples = set()
    for bond1 in bonds:
        i, j = bond1
        for bond2 in bonds:
            k, l = bond2
            if k == j and l != i:
                triples.add((i, j, bond2[1]))
            elif l == j and k != i:
                triples.add((i, j, bond2[0]))
    return list(triples)

class RDKitConformer:
    def __init__(self, smiles):
        """
        :param atom_positions: numpy.ndarray of shape [num_atoms, 3] of dtype float64
        """
        self.smiles = smiles
        self.rdk_mol = self.get_mol_from_smiles(smiles)
        self.rdk_conf = self.embed_mol_and_get_conformer(self.rdk_mol, extra_opt=True)

        self.set_atom_positions(self.rdk_conf.GetPositions())
        self.freely_rotatable_tas, self.ring_bonds, self.nonring_bonds = get_rotatable_ta_list(self.rdk_mol)

        self.hydrogen_indices = [self.rdk_mol.GetAtomWithIdx(i).GetAtomicNum() == 1 for i in range(self.rdk_mol.GetNumAtoms())]

        print(self.ring_bonds+self.nonring_bonds)
        self.bonds = self.get_mol_bonds(self.rdk_mol)
        self.bond_lengths = self.get_bond_lengths(self.rdk_mol)
        self.bond_angles = find_bond_angles(self.bonds)
        print('Number of bonds:', len(self.bonds))
        print(self.bond_angles)

    def __deepcopy__(self, memo):
        atom_positions = self.get_atom_positions()
        cls = self.__class__
        new_obj = cls.__new__(
            cls, atom_positions, self.smiles, self.freely_rotatable_tas
        )
        return new_obj

    def get_mol_from_smiles(self, smiles):
        """Create RDKit molecule from SMILES string
        :param smiles: python string
        :returns: rdkit.Chem.rdchem.Mol object"""
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        return mol
    
    def get_mol_bonds(self, mol):
        bonds = []
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if begin_idx in self.hydrogen_indices or end_idx in self.hydrogen_indices:
                continue
            elif (begin_idx, end_idx) in self.ring_bonds:
                continue
            elif (begin_idx, end_idx) in self.nonring_bonds:
                continue
            bonds.append((begin_idx, end_idx))
        return bonds
    
    def get_bond_lengths(self, mol):
        bond_lengths = []
        bonds = mol.GetBonds()
        for bond in bonds:
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if begin_idx in self.hydrogen_indices or end_idx in self.hydrogen_indices:
                continue
            elif (begin_idx, end_idx) in self.ring_bonds:
                continue
            elif (begin_idx, end_idx) in self.nonring_bonds:
                continue
            l = rdMolTransforms.GetBondLength(self.rdk_conf, begin_idx, end_idx)
            bond_lengths.append(float(l))
        return bond_lengths
    
    def set_bond_lengths(self, bond_lengths, bonds):
        bond_lengths = bond_lengths.cpu().numpy()
        bond_lengths = bond_lengths + self.bond_lengths
        for i, (bond, length) in enumerate(zip(bonds, bond_lengths)):
            rdMolTransforms.SetBondLength(self.rdk_conf, *bond, float(length))
        self.bond_lengths = bond_lengths


    def embed_mol_and_get_conformer(self, mol, extra_opt=False):
        """Embed RDkit mol with a conformer and return the RDKit conformer object
        (which is synchronized with the RDKit molecule object)
        :param mol: rdkit.Chem.rdchem.Mol object defining the molecule
        :param extre_opt: bool, if True, an additional optimisation of the conformer will be performed
        """
        AllChem.EmbedMolecule(mol)
        if extra_opt:
            AllChem.MMFFOptimizeMolecule(mol, confId=0, maxIters=10000)
        return mol.GetConformer()

    def set_atom_positions(self, atom_positions):
        """Set atom positions of the self.rdk_conf to the input atom_positions values
        :param atom_positions: 2d numpy array of shape [num atoms, 3] with new atom positions
        """
        for idx, pos in enumerate(atom_positions):
            x,y,z = pos
            self.rdk_conf.SetAtomPosition(idx, Point3D(x.item(),y.item(),z.item()))

    def get_atom_positions(self):
        """
        :returns: numpy array of atom positions of shape [num_atoms, 3]
        """
        return self.rdk_conf.GetPositions()

    def get_atomic_numbers(self):
        """Get atomic numbers of the atoms as 1d numpy array
        :returns: numpy array of atomic numbers of shape [num_atoms,]"""
        atomic_numbers = [atom.GetAtomicNum() for atom in self.rdk_mol.GetAtoms()]
        return np.array(atomic_numbers)

    def get_n_atoms(self):
        return self.rdk_mol.GetNumAtoms()

    def set_bond_angle(self, bond_angle, value):
        rdMolTransforms.SetAngleRad(self.rdk_conf, 
                                    iAtomId=bond_angle[0], 
                                    jAtomId=bond_angle[1], 
                                    kAtomId=bond_angle[2], 
                                    value=float(value))

    def set_torsion_angle(self, torsion_angle, value):
        rdMolTransforms.SetDihedralRad(self.rdk_conf, *torsion_angle, float(value))

    def get_all_torsion_angles(self):
        """
        :returns: a dict of all tostion angles in the molecule with their values
        """
        return get_all_torsion_angles(self.rdk_mol, self.rdk_conf)

    def get_freely_rotatable_tas_values(self):
        """
        :returns: a list of values of self.freely_rotatable_tas
        """
        return get_torsion_angles_values(self.rdk_conf, self.freely_rotatable_tas)


    def randomize_freely_rotatable_tas(self):
        """
        Uniformly randomize torsion angles defined by self.freely_rotatable_tas
        """
        for torsion_angle in self.freely_rotatable_tas:
            increment = np.random.uniform(0, 2 * np.pi)
            self.increment_torsion_angle(torsion_angle, increment)

    def increment_torsion_angle(self, torsion_angle, increment):
        """
        :param torsion_angle: tuple of 4 integers defining the torsion angle
        :param increment: a float value of the increment of the angle (in radians)
        """
        initial_value = rdMolTransforms.GetDihedralRad(self.rdk_conf, *torsion_angle)
        self.set_torsion_angle(torsion_angle, initial_value + increment)
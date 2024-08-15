
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints, rdMolTransforms
from rdkit.Geometry.rdGeometry import Point3D
import numpy as np
import torch

def torsions_to_conformations(xyz, tas, rd_conf, device):
    # energies = self.model((an_bs, xyz.reshape(-1, self.data_ndim//3, 3))).energies
    confs = []
    for rotation_set in xyz:
        for i, torsion in enumerate(tas):
            rd_conf.set_torsion_angle(torsion, rotation_set[i])
        xyz = torch.tensor(rd_conf.get_atom_positions()).to(device, dtype=torch.float32)
        confs.append(xyz)
    confs = torch.stack(confs)
    return confs

def get_torsion_angles_atoms_list(mol):
    return [x[0][0] for x in TorsionFingerprints.CalculateTorsionLists(mol)[0]]

def get_torsion_angles_values(conf, torsion_angles_atoms_list):
    return [
        np.float32(rdMolTransforms.GetDihedralRad(conf, *ta))
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
    #torsion_pattern = "[*]~[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]~[*]"
    #substructures = Chem.MolFromSmarts(torsion_pattern)
    #torsion_angles = remove_duplicate_tas(list(mol.GetSubstructMatches(substructures)))
    nonring, ring = TorsionFingerprints.CalculateTorsionLists(mol, )

    def collect_4tuples(output):
        result = []
        for item in output:
            tuples_list, _ = item
            result.extend(tuples_list)
        return result

    torsion_angles = remove_duplicate_tas(collect_4tuples(nonring))
    torsion_angles = [ta for ta in torsion_angles if not is_hydrogen_ta(mol, ta)]
    return torsion_angles

class RDKitConformer:
    def __init__(self, smiles):
        """
        :param atom_positions: numpy.ndarray of shape [num_atoms, 3] of dtype float64
        """
        self.smiles = smiles
        self.rdk_mol = self.get_mol_from_smiles(smiles)
        self.rdk_conf = self.embed_mol_and_get_conformer(self.rdk_mol, extra_opt=True)

        self.set_atom_positions(self.rdk_conf.GetPositions())
        self.freely_rotatable_tas = get_rotatable_ta_list(self.rdk_mol)

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
            self.rdk_conf.SetAtomPosition(idx, Point3D(*pos))

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
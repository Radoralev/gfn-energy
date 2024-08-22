"""Wrapper for semi-empirical QM energies with XTB.
"""

__all__ = ["XTBEnergy", "XTBBridge"]


import warnings
import torch
import numpy as np
from .base import _BridgeEnergy, _Bridge
import os
print(os.environ.get('LD_LIBRARY_PATH'))

from tblite.interface import Calculator


class XTBBridge(_Bridge):
    """Wrapper around XTB for semi-empirical QM energy calculations.

    Parameters
    ----------
    numbers : np.ndarray
        Atomic numbers
    temperature : float
        Temperature in Kelvin.
    method : str
        The semi-empirical method that is used to compute energies.
    solvent : str
        The solvent. If empty string, perform a vacuum calculation.
    verbosity : int
        0 (muted), 1 (minimal), 2 (full)
    err_handling : str
        How to deal with exceptions inside XTB. One of `["ignore", "warning", "error"]`

    Attributes
    ----------
    n_atoms : int
        The number of atoms in this molecules.
    available_solvents : List[str]
        The solvent models that are available for computations in xtb.
    available_methods : List[str]
        The semiempirical methods that are available for computations in xtb.

    Examples
    --------
    Setting up an XTB energy for a small peptide from bgmol
    >>> from bgmol.systems import MiniPeptide
    >>> from bgflow import XTBEnergy, XTBBridge
    >>> import numpy as np
    >>> import torch
    >>> system = MiniPeptide("G")
    >>> numbers = np.array([atom.element.number for atom in system.mdtraj_topology.atoms])
    >>> target = XTBEnergy(XTBBridge(numbers=numbers, temperature=300, solvent="water"))
    >>> xyz = torch.tensor(system.positions)
    >>> energy = target.energy(xyz)

    Notes
    -----
    Requires the xtb-python program (installable with `conda install -c conda-forge xtb-python`).

    """
    def __init__(
            self,
            numbers: np.ndarray,
            temperature: float,
            method: str = "GFN2-xTB",
            solvent: str = "",
            verbosity: int = 0,
            err_handling: str = "warning"
    ):
        self.numbers = numbers
        self.temperature = temperature
        self.method = method
        self.solvent = solvent
        self.verbosity = verbosity
        self.err_handling = err_handling
        super().__init__()

    @property
    def n_atoms(self):
        return len(self.numbers)

class XTBEnergy(_BridgeEnergy):
    """Semi-empirical energy computation with XTB.

    Parameters
    ----------
    xtb_bridge : XTBBridge
        The wrapper object.
    two_event_dims : bool
        Whether to use two event dimensions.
        In this case, the energy call expects positions of shape (*batch_shape, n_atoms, 3).
        Otherwise, it expects positions of shape (*batch_shape, n_atoms * 3).
    """
    pass


_BOLTZMANN_CONSTANT_HE = 3.1668115634556076e-06  # in hartree / kelvin
_BOHR_RADIUS = 0.0529177210903  # nm


def _bohr2nm(x):
    return x * _BOHR_RADIUS


def _nm2bohr(x):
    return x / _BOHR_RADIUS


def _per_bohr2per_nm(x):
    return _nm2bohr(x)


def _hartree2kbt(x, temperature):
    kbt = _BOLTZMANN_CONSTANT_HE * temperature
    return x / kbt


def _hartree_per_bohr2kbt_per_nm(x, temperature):
    return _per_bohr2per_nm(_hartree2kbt(x, temperature))
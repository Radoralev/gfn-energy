import contextlib
import os
import re
import subprocess
import warnings

import numpy as np



# This is a hotfix for tblite (used for the conformer generation) not
# importing correctly unless it is being imported first.
try:
    from tblite import interface
except:
    pass

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import numpy as np
import numpy.typing as npt
import torch
from joblib import Parallel, delayed
from torch import Tensor
from wurlitzer import pipes

METHODS = {"gfn2": "gfn 2", "gfnff": "gfnff"}


def _get_energy(file):
    normal_termination = False
    energy = np.nan
    with open(file) as f:
        for l in f:
            if "TOTAL ENERGY" in l:
                try:
                    energy = float(re.search(r"[+-]?(?:\d*\.)?\d+", l).group())
                except:
                    return np.nan
            if "normal termination of xtb" in l:
                normal_termination = True
    if not normal_termination:
        return np.nan
    return energy

def run_gfn_xtb(
    filepath,
    filename,
    gfn_version="gfnff",
    opt=False,
    gfn_xtb_config: str = None,
    remove_scratch=True,
    solvent=False,
):
    """
    Runs GFN_XTB/FF given a directory and either a coord file or all coord files will be run

    :param filepath: Directory containing the coord file
    :param filename: if given, the specific coord file to run
    :param gfn_version: GFN_xtb version (default is 2)
    :param opt: optimization or single point (default is opt)
    :param gfn_xtb_config: additional xtb config (default is None)
    :param remove_scratch: remove xtb files
    :return:
    """
    xyz_file = os.path.join(filepath, filename)

    # optimization vs single point
    if opt:
        opt = "--opt"
    else:
        opt = ""

    # cd to filepath
    starting_dir = os.getcwd()
    os.chdir(filepath)

    file_name = str(xyz_file.split(".")[0])
    cmd = "xtb --{} {} {} {} --iterations 1000".format(
        str(gfn_version), xyz_file, opt, str(gfn_xtb_config or "")
    )

    if solvent:
        cmd += f"--alpb water "

    # run XTB
    with open(file_name + ".out", "w") as fd:
        subprocess.run(cmd, shell=True, stdout=fd, stderr=subprocess.STDOUT)

    # check XTB results
    if os.path.isfile(os.path.join(filepath, "NOT_CONVERGED")):
        # optimization not converged
        warnings.warn(
            "xtb --{} for {} is not converged, using last optimized step instead; proceed with caution".format(
                str(gfn_version), file_name
            )
        )

        # remove files
        if remove_scratch:
            os.remove(os.path.join(filepath, "NOT_CONVERGED"))
            os.remove(os.path.join(filepath, "xtblast.xyz"))
            os.remove(os.path.join(filepath, file_name + ".out"))
        energy = np.nan

    elif opt and not os.path.isfile(os.path.join(filepath, "xtbopt.xyz")):
        # other abnormal optimization convergence
        warnings.warn(
            "xtb --{} for {} abnormal termination, likely scf issues, using initial geometry instead; proceed with caution".format(
                str(gfn_version), file_name
            )
        )
        if remove_scratch:
            os.remove(os.path.join(filepath, file_name + ".out"))
        energy = np.nan

    else:
        # normal convergence
        # get energy
        energy = _get_energy(file_name + ".out")
        if remove_scratch:
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(filepath, file_name + ".out"))
                os.remove(os.path.join(filepath, "gfnff_charges"))
                os.remove(os.path.join(filepath, "gfnff_adjacency"))
                os.remove(os.path.join(filepath, "gfnff_topo"))
                os.remove(os.path.join(filepath, "xtbopt.log"))
                os.remove(os.path.join(filepath, "xtbopt.xyz"))
                os.remove(os.path.join(filepath, "xtbtopo.mol"))
                os.remove(os.path.join(filepath, "wbo"))
                os.remove(os.path.join(filepath, "charges"))
                os.remove(os.path.join(filepath, "xtbrestart"))
    os.chdir(starting_dir)
    return energy



def _write_xyz_file(
    elements: npt.NDArray, coordinates: npt.NDArray, file_path: str
) -> None:
    num_atoms = len(elements)
    with open(file_path, "w") as f:
        f.write(str(num_atoms) + "\n")
        f.write("\n")

        for i in range(num_atoms):
            element = elements[i]
            x, y, z = coordinates[i]
            line = f"{int(element)} {x:.6f} {y:.6f} {z:.6f}\n"
            f.write(line)


def get_energy(numbers, positions, method="gfnff", solvent=False):
    directory = TemporaryDirectory()
    file_name = "input.xyz"

    _write_xyz_file(numbers, positions, str(Path(directory.name) / "input.xyz"))
    with pipes():
        energy = run_gfn_xtb(directory.name, file_name, gfn_version=method, solvent=solvent)
    directory.cleanup()

    if np.isnan(energy):
        return 0.0

    return energy


def optimize_conformation(numbers, positions, method="gfnff", solvent=True):
    directory = TemporaryDirectory()
    file_name = "input.xyz"

    # Write the input xyz file
    _write_xyz_file(numbers, positions, str(Path(directory.name) / file_name))

    # Run xtb with optimization and solvent
    with pipes():
        energy = run_gfn_xtb(
            directory.name, file_name, gfn_version=method, opt=True, solvent=solvent
        )

    # Read optimized geometry from xtbopt.xyz
    optimized_xyz_file = Path(directory.name) / "xtbopt.xyz"
    if optimized_xyz_file.exists():
        elements_opt, positions_opt = _read_xyz_file(str(optimized_xyz_file))
    else:
        # Optimization failed, return original positions.
        elements_opt = numbers
        positions_opt = positions
        warnings.warn("Optimization failed, returning original positions.")

    # Clean up temporary directory
    directory.cleanup()

    return elements_opt, positions_opt, energy

def _read_xyz_file(file_path: str):
    # Map element symbols to atomic numbers
    element_symbol_to_atomic_number = {
        'H': 1, 'He': 2,
        'Li': 3, 'Be': 4, 'B': 5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10,
        'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18,
        'K':19, 'Ca':20, 'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30,
        'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36,
        'Rb':37, 'Sr':38, 'Y':39, 'Zr':40, 'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46, 'Ag':47, 'Cd':48,
        'In':49, 'Sn':50, 'Sb':51, 'Te':52, 'I':53, 'Xe':54,
        # Add more elements as needed
    }

    with open(file_path, "r") as f:
        lines = f.readlines()

    num_atoms = int(lines[0])
    # Skip the second line (comment)
    elements = []
    positions = []
    for line in lines[2:2+num_atoms]:
        tokens = line.strip().split()
        if len(tokens) != 4:
            continue
        element_symbol = tokens[0]
        x, y, z = map(float, tokens[1:4])
        atomic_number = element_symbol_to_atomic_number.get(element_symbol)
        if atomic_number is None:
            raise ValueError(f"Unknown element symbol: {element_symbol}")
        elements.append(atomic_number)
        positions.append([x, y, z])

    positions = np.array(positions)
    elements = np.array(elements)
    return elements, positions

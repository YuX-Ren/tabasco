from rdkit import Chem
from typing import List
from torch import Tensor
import os
import numpy as np
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

def largest_component(molecules: List[Chem.Mol]) -> List[Chem.Mol]:
    """Return the largest connected component for each molecule.

    Args:
        molecules: Iterable of RDKit Mol objects with 3-D coordinates.

    Returns:
        list[Chem.Mol]: Each entry is the fragment with the most atoms for
        the corresponding input molecule.

    From: https://github.com/jostorge/diffusion-hopping/blob/main/diffusion_hopping/analysis/util.py
    """
    return [
        max(
            Chem.GetMolFrags(mol, asMols=True),
            key=lambda x: x.GetNumAtoms(),
            default=mol,
        )
        for mol in molecules
    ]


def attempt_sanitize(mol: Chem.Mol):
    """Run `Chem.SanitizeMol` and return `None` on failure.

    Args:
        mol: RDKit Mol to sanitize.

    Returns:
        Chem.Mol | None: Sanitised molecule or `None` if RDKit raises.

    Credits: Charles Harris
    """
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        print(f"Sanitization failed: {e}")
        return None


def reorder_molecule_by_smiles(mol: Chem.Mol) -> Chem.Mol | None:
    """Renumber atoms to follow canonical SMILES indexing.

    Args:
        mol: RDKit Mol. If `None`, the function returns `None`.

    Returns:
        Chem.Mol | None: Renumbered copy of the molecule.

    Raises:
        ValueError: If canonicalisation or substructure matching fails.
    """

    if mol is None:
        return None

    mol_copy = Chem.Mol(mol)
    canonical_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol_copy))

    if canonical_mol is None:
        raise ValueError("Failed to canonicalize molecule")

    match = mol_copy.GetSubstructMatch(canonical_mol)
    if not match:
        raise ValueError("Failed to find substructure match")

    return Chem.RenumberAtoms(mol_copy, match)


def write_xyz_file(coords: Tensor, atom_types: List[str], filename: str) -> None:
    """Write an XYZ file.

    Args:
        coords: Array-like of shape `(N, 3)` with Cartesian coordinates.
        atom_types: Sequence of length `N` with element symbols.
        filename: Path to the output `.xyz` file.
    """
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, "w") as f:
        f.write(out)


def save_aa_coords(aatypes_seq_list, coords_list, savedir=None):


    with open(os.path.join(savedir, "sequence.fasta"), 'w') as file:
        for idx, aa_seq in enumerate(aatypes_seq_list):
            file.write(f">seq{idx} \n")
            file.write(aa_seq + " \n")

    with open(os.path.join(savedir, "seq_len.txt"), 'w') as file:
        for idx, aa_seq in enumerate(aatypes_seq_list):
            file.write(f">seq{idx} {len(aa_seq)} \n")

    savedir = os.path.join(savedir, "pdbs", "")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for i in range(len(coords_list)):
        fname = os.path.join(savedir, f"generated_{i}.pdb")
        write_coords_to_pdb(coords_list[i], fname)

def write_coords_to_pdb(coords: np.ndarray, out_fname: str) -> str:
    """
    Write the coordinates to the given pdb fname
    """
    # Create a new PDB file using biotite
    # https://www.biotite-python.org/tutorial/target/index.html#creating-structures
    # assert len(coords) % 3 == 0

    atoms = []
    for i, ca_coord in enumerate(coords):

        atom = struc.Atom(
            ca_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i + 1,
            res_name="GLY",
            atom_name="CA",
            element="C",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )

        atoms.extend([atom])
    full_structure = struc.array(atoms)


    sink = PDBFile()
    sink.set_structure(full_structure)
    sink.write(out_fname)
    return out_fname
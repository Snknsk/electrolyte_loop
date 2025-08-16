"""Generate a Packmol input file using consolidated 'polymer_matrix.xyz' and 'solvent.xyz' files."""
import os
import logging
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolToXYZBlock

# --- Add ASE imports for nanoparticles ---
from ase import Atoms
from ase.cluster.icosahedron import Icosahedron
from ase.build import bulk
import ase.io

# Define logger before any usage
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
box_size = 70.0  # default box size in Ångstroms

def generate_polymer_chain_from_smiles(smiles: str, repeat_units: int = 3) -> Chem.Mol | None:
    """
    Generate a linear polymer by stitching together multiple monomers.
    Each monomer must have two '[*]' atoms as the attachment points (head and tail).
    """
    try:
        monomer = Chem.MolFromSmiles(smiles)
        if monomer is None:
            logger.warning("⚠️ RDKit failed to parse SMILES with dummy atoms.")
            return None

        # Find dummy atoms
        dummy_atoms = [atom.GetIdx() for atom in monomer.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummy_atoms) != 2:
            logger.warning("⚠️ Monomer must have exactly two '[*]' dummy atoms.")
            return None

        # Identify their neighbors
        neighbor_indices = []
        for dummy_idx in dummy_atoms:
            neighbors = monomer.GetAtomWithIdx(dummy_idx).GetNeighbors()
            if len(neighbors) != 1:
                logger.warning("⚠️ Each dummy atom must have only one neighbor.")
                return None
            neighbor_indices.append(neighbors[0].GetIdx())

        # Remove dummy atoms
        editable = Chem.EditableMol(monomer)
        editable.RemoveAtom(max(dummy_atoms))  # remove higher index first
        editable.RemoveAtom(min(dummy_atoms))
        core = editable.GetMol()

        # Track indices of connection points
        head_idx, tail_idx = neighbor_indices

        # Because atoms are removed, adjust tail_idx if needed
        if max(dummy_atoms) < tail_idx:
            tail_idx -= 1
        if max(dummy_atoms) < head_idx:
            head_idx -= 1
        if min(dummy_atoms) < tail_idx:
            tail_idx -= 1
        if min(dummy_atoms) < head_idx:
            head_idx -= 1

        polymer = core
        last_tail = tail_idx

        for _ in range(repeat_units - 1):
            new_core = Chem.Mol(core)
            polymer = Chem.CombineMols(polymer, new_core)
            editable = Chem.EditableMol(polymer)

            offset = polymer.GetNumAtoms() - core.GetNumAtoms()
            editable.AddBond(last_tail, offset + head_idx, order=Chem.rdchem.BondType.SINGLE)

            last_tail = offset + tail_idx
            polymer = editable.GetMol()

        polymer = Chem.AddHs(polymer)
        return polymer

    except Exception as e:
        logger.error(f"❌ Polymer generation failed for SMILES: {smiles} | {e}")
        return None

lines = []
lines.append("nloop0 1000")
lines.append("tolerance 2.0")
lines.append("filetype xyz")
lines.append("structure polymer_matrix.xyz")
lines.append("  number 1")
lines.append(f"  inside box 0. 0. 0. {box_size} {box_size} {box_size}.")
lines.append("end structure\n")

# Solvent
def optimize_molecule(mol, ff='UFF'):
    # from rdkit.Chem import AllChem  # Already imported at top
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if ff == 'UFF':
            AllChem.UFFOptimizeMolecule(mol)
        elif ff == 'MMFF':
            AllChem.MMFFOptimizeMolecule(mol)
        return mol
    except Exception as e:
        print(f"⚠️ {ff} optimization failed: {e}")
        return mol

# --- Add molecule placement function ---
import numpy as np
def place_molecule_in_box(existing_coords, mol, box_size, min_dist=2.0, max_attempts=200):
    num_atoms = mol.GetNumAtoms()
    conf = mol.GetConformer()

    for _ in range(max_attempts):
        shift = np.random.uniform(0, box_size, size=3)
        mol_coords = np.array([list(conf.GetAtomPosition(i)) + shift for i in range(num_atoms)])

        too_close = False
        for new_pos in mol_coords:
            for existing_pos in existing_coords:
                if np.linalg.norm(new_pos - existing_pos) < min_dist:
                    too_close = True
                    break
            if too_close:
                break
        if not too_close:
            return mol_coords  # Accept

    return None  # Fail

# --- New: random rotation and translation with collision avoidance ---
def random_translate_and_rotate(mol, box_size):
    conf = mol.GetConformer()
    # Apply random rotation
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    psi = np.random.uniform(0, 2 * np.pi)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    Ry = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])
    Rz = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx

    # Apply rotation
    for i in range(mol.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        new_pos = R @ pos
        conf.SetAtomPosition(i, new_pos.tolist())

    # Apply random translation within box
    shift = np.random.uniform(0.1 * box_size, 0.9 * box_size, size=3)
    for i in range(mol.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        conf.SetAtomPosition(i, (pos + shift).tolist())

    return mol

# Build multiple chains with variable length, but do not place yet
# --- HSP model import ---
try:
    from hansen_solubility import predict_hsp
except ImportError:
    def predict_hsp(smiles):
        return {'d': 0.0, 'p': 0.0, 'h': 0.0}
    logger.warning("hansen_solubility.predict_hsp not found; using dummy function.")

# --- LigParGen/OPLS-AA integration ---
import requests
import tempfile
import shutil
import subprocess
import certifi
import argparse

# Global flag to skip LigParGen attempts if requested or if unavailable
SKIP_LIGPARGEN = False

def call_ligpargen(smiles, name, output_dir="ligpargen_params", ff="OPLSAA"):
    """
    Query LigParGen (web API) for OPLS-AA force field parameters for a given SMILES.
    Saves .itp/.top/.prm or .lmp/.xml/.prmtop in the output_dir.
    If LigParGen is unavailable or SSL fails, fallback to local MMFF or UFF with clear notice.
    """
    global SKIP_LIGPARGEN
    os.makedirs(output_dir, exist_ok=True)
    if SKIP_LIGPARGEN:
        print(f"⚠️ LigParGen unavailable, skipping and falling back to local UFF/MMFF.")
        # fallback below
    else:
        # LigParGen web API endpoint (as of 2024)
        LIGPARGEN_URL = "https://zarbi.chem.yale.edu/ligpargen/process_smiles.cgi"
        payload = {
            "smiles": smiles,
            "molname": name,
            "FF": ff,
            "charge": "auto",
            "submit": "Submit"
        }
        try:
            r = requests.post(LIGPARGEN_URL, data=payload, timeout=90, verify=certifi.where())
            if r.status_code != 200:
                print(f"❌ LigParGen query failed for {name}")
                print(f"⚠️ LigParGen unavailable, skipping and falling back to local UFF/MMFF.")
                SKIP_LIGPARGEN = True
                # fallback below
            else:
                # LigParGen returns a page with a download link for the parameter zip
                import re
                match = re.search(r'href="([^"]+\.zip)"', r.text)
                if not match:
                    print(f"❌ LigParGen did not return a zip for {name}")
                    print(f"⚠️ LigParGen unavailable, skipping and falling back to local UFF/MMFF.")
                    SKIP_LIGPARGEN = True
                    # fallback below
                else:
                    zip_url = "https://zarbi.chem.yale.edu/ligpargen/" + match.group(1)
                    zip_data = requests.get(zip_url, timeout=60, verify=certifi.where()).content
                    # Save and extract zip
                    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmpf:
                        tmpf.write(zip_data)
                        tmpf.flush()
                        shutil.unpack_archive(tmpf.name, output_dir)
                    print(f"✅ LigParGen params for {name} saved to {output_dir}")
                    return output_dir
        except requests.exceptions.SSLError as ssl_err:
            print(f"⚠️ LigParGen SSL error for {name}: {ssl_err}")
            print(f"⚠️ LigParGen unavailable, skipping and falling back to local UFF/MMFF.")
            SKIP_LIGPARGEN = True
            # fallback below
        except Exception as e:
            print(f"⚠️ LigParGen exception for {name}: {e}")
            print(f"⚠️ LigParGen unavailable, skipping and falling back to local UFF/MMFF.")
            SKIP_LIGPARGEN = True
            # fallback below
    # Fallback: try MMFF, then UFF, and save to XYZ with fallback note
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        ff_used = None
        try:
            AllChem.MMFFOptimizeMolecule(mol)
            ff_used = "MMFF"
        except Exception:
            AllChem.UFFOptimizeMolecule(mol)
            ff_used = "UFF fallback"
        xyz_block = MolToXYZBlock(mol)
        fallback_xyz = os.path.join(output_dir, f"{name}_UFF_fallback.xyz")
        with open(fallback_xyz, "w") as f:
            f.write(f"{mol.GetNumAtoms()}\n{ff_used} conformer\n")
            f.write(xyz_block)
        print(f"📦 Saved {fallback_xyz} | FF used: {ff_used}")
        return None
    except Exception as e2:
        print(f"❌ UFF/MMFF fallback failed for {name}: {e2}")
        return None

# --- Improved conformer embedding and optimization with detailed logging ---
def embed_and_optimize(mol: Chem.Mol, name: str = "", max_attempts: int = 5) -> 'Optional[Chem.Mol]':
    """
    Robust conformer embedding and optimization for an RDKit molecule.
    Tries ETKDGv3 multiple times, then falls back to random coordinates if needed.
    Returns the optimized molecule on success, or None on failure.
    Adds detailed logging for each stage.
    """
    from rdkit.Chem import AllChem
    import logging
    log = logging.getLogger("embed_and_optimize")
    # Log the SMILES string being embedded
    smiles = Chem.MolToSmiles(mol)
    # --- BEGIN PATCH: Detailed logging before embedding ---
    print(f"🔬 Embedding molecule: {name} | SMILES: {smiles}")
    print(f"{mol.GetNumAtoms()} atoms | {Chem.GetSSSR(mol)} rings")
    print(smiles)
    # --- END PATCH
    log.info(f"🔬 Embedding molecule: {name} | SMILES: {smiles}")
    atom_count = mol.GetNumAtoms()
    ring_count = mol.GetRingInfo().NumRings()
    log.info(f"📏 Atom count: {atom_count}, Ring count: {ring_count}")

    # Skip embedding for small or empty molecules
    if atom_count < 2:
        log.warning(f"⚠️ Molecule too small for embedding (atom count: {atom_count}). Skipping.")
        print(f"⚠️ Molecule too small for embedding (atom count: {atom_count}). Skipping.")
        return None
    mol = Chem.AddHs(mol)
    # Try ETKDG embedding
    for i in range(max_attempts):
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = np.random.randint(1, 1e6)
            ret = AllChem.EmbedMolecule(mol, params)
            if ret == 0:
                log.info(f"✅ ETKDG embedding succeeded on attempt {i+1} for {name}")
                print(f"✅ ETKDG embedding succeeded on attempt {i+1} for {name}")
                # Try MMFF optimization
                try:
                    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
                    if mp is not None:
                        ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
                        if ff is not None:
                            ff.Minimize()
                            log.info(f"✅ MMFF optimization succeeded for {name}")
                            print(f"✅ MMFF optimization succeeded for {name}")
                            return mol
                        else:
                            log.warning(f"❌ MMFF force field creation failed for {name}")
                            raise ValueError("MMFF force field could not be created.")
                    else:
                        log.warning(f"❌ MMFF properties creation failed for {name}")
                        raise ValueError("MMFF properties could not be created.")
                except Exception as mmff_e:
                    log.warning(f"⚠️ MMFF optimization failed for {name}: {mmff_e}, falling back to UFF.")
                    print(f"⚠️ MMFF fallback optimization failed for {name}: {mmff_e}")
                    try:
                        AllChem.UFFOptimizeMolecule(mol)
                        log.info(f"✅ UFF optimization succeeded for {name}")
                        print(f"✅ UFF optimization succeeded for {name}")
                        return mol
                    except Exception as uff_e:
                        log.warning(f"❌ Fallback random embedding and UFF optimization failed for {name}: {uff_e}")
                        print(f"❌ Fallback random embedding and UFF optimization failed for {name}: {uff_e}")
                        continue
            else:
                log.warning(f"❌ ETKDG embedding failed on attempt {i+1} for {name} (ret={ret})")
                print(f"❌ ETKDG embedding failed on attempt {i+1} for {name} (ret={ret})")
        except Exception as e:
            log.error(f"⚠️ Attempt {i + 1} failed for {name}: {e}")
            print(f"⚠️ Attempt {i + 1} failed for {name}: {e}")
            continue

    # Final fallback using random coordinates if nothing else worked
    try:
        log.warning(f"⚠️ ETKDG failed for {name}, falling back to random coordinates...")
        print(f"⚠️ ETKDG failed for {name}, falling back to random coordinates...")
        mol = Chem.AddHs(mol)
        ret = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        if ret == 0:
            log.info(f"✅ Random coordinate embedding succeeded for {name}")
            print(f"✅ Random coordinate embedding succeeded for {name}")
        else:
            log.error(f"❌ Random coordinate embedding failed for {name} (ret={ret})")
            print(f"❌ Random coordinate embedding failed for {name} (ret={ret})")
        try:
            mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
            if mp is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
                if ff is not None:
                    ff.Minimize()
                    log.info(f"✅ MMFF optimization succeeded for {name}")
                    print(f"✅ MMFF optimization succeeded for {name}")
                    return mol
                else:
                    log.warning(f"❌ MMFF force field creation failed after random coords for {name}")
            else:
                log.warning(f"❌ MMFF properties creation failed after random coords for {name}")
        except Exception as mmff_e:
            log.warning(f"⚠️ MMFF fallback optimization failed for {name}: {mmff_e}")
            print(f"⚠️ MMFF fallback optimization failed for {name}: {mmff_e}")
            try:
                AllChem.UFFOptimizeMolecule(mol)
                log.info(f"✅ UFF optimization succeeded for {name}")
                print(f"✅ UFF optimization succeeded for {name}")
                return mol
            except Exception as uff_e:
                log.error(f"❌ Fallback random embedding and UFF optimization failed for {name}: {uff_e}")
                print(f"❌ Fallback random embedding and UFF optimization failed for {name}: {uff_e}")
                return None
    except Exception as e:
        log.error(f"❌ Fallback random embedding failed for {name}: {e}")
        print(f"❌ Fallback random embedding failed for {name}: {e}")
        return None

    # If still here, check for conformer issues and log
    try:
        nconf = mol.GetNumConformers()
        log.warning(f"❌ No valid conformer found for {name} after all attempts. NumConformers={nconf}")
        print(f"❌ No valid conformer found for {name} after all attempts. NumConformers={nconf}")
        if nconf > 0:
            ids = [c.GetId() for c in mol.GetConformers()]
            log.warning(f"Conformer IDs: {ids}")
            print(f"Conformer IDs: {ids}")
    except Exception as conf_e:
        log.error(f"Error checking conformer IDs: {conf_e}")
        print(f"Error checking conformer IDs: {conf_e}")
    return None

def get_bounding_box(mol):
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    return min_coords, max_coords

# --- Placement logic with variable staggered spacing and minimum distance check ---
placed_coords = []
placed_chains = []

def is_nonoverlapping(new_coords, existing_coords, cutoff=2.5):
    for atom in new_coords:
        if len(existing_coords) > 0 and np.any(np.linalg.norm(np.array(existing_coords) - atom, axis=1) < cutoff):
            return False
    return True

def try_place_molecule(mol, spacing=3.5, max_attempts=100):
    num_atoms = mol.GetNumAtoms()
    # Check for at least one conformer before proceeding
    if mol.GetNumConformers() == 0:
        logger.warning(f"❌ No conformer found for molecule: skipping placement")
        return False
    for attempt in range(max_attempts):
        # Random rotation
        conf = mol.GetConformer()
        # Random rotation matrix
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        psi = np.random.uniform(0, 2 * np.pi)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        Ry = np.array([
            [np.cos(phi), 0, np.sin(phi)],
            [0, 1, 0],
            [-np.sin(phi), 0, np.cos(phi)]
        ])
        Rz = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        R = Rz @ Ry @ Rx
        coords = []
        for i in range(num_atoms):
            pos = np.array(conf.GetAtomPosition(i))
            coords.append(pos)
        coords = np.array(coords)
        coords = coords @ R.T
        # Random translation, with variable spacing
        shift = np.random.uniform(spacing, box_size - spacing, size=3)
        coords += shift
        if is_nonoverlapping(coords, placed_coords, cutoff=2.5):
            # Set positions
            for i in range(num_atoms):
                conf.SetAtomPosition(i, coords[i].tolist())
            placed_coords.extend(coords)
            placed_chains.append(mol)
            return True
    return False

if __name__ == "__main__":
    import numpy as np
    # Inverse-log random repeat unit generator
    def inverse_log_random(min_val=3, max_val=20, size=1):
        r = np.random.uniform(0.01, 1.0, size)
        vals = min_val + (max_val - min_val) * (1 - np.log(r) / np.log(100))
        return np.clip(vals.astype(int), min_val, max_val)

    polymer_df = pd.read_csv("/Users/eddypoon/Desktop/electrolyte_loop/01_data_raw/data_collection/raw/polymer_electrolytes.csv")
    smiles_list = polymer_df["SMILES"].dropna().tolist()

    # Track current polymer index in a state file
    state_file = "/Users/eddypoon/Desktop/electrolyte_loop/02_data_cleaned/polymer_index.txt"
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            i = int(f.read().strip())
    else:
        i = 0

    polymer_chains = []
    placed_chains = []
    if i < len(smiles_list):
        smiles = smiles_list[i]
        # --- Atom filter: skip if any atom not in allowed set, but allow dummy atoms (atomic number 0) ---
        from rdkit.Chem import Descriptors
        allowed_atomic_nums = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # H, C, N, O, F, P, S, Cl, Br, I
        while True:
            mol_check = Chem.MolFromSmiles(smiles)
            if mol_check is not None:
                atomic_nums = [atom.GetAtomicNum() for atom in mol_check.GetAtoms()]
                # Check only real atoms, ignore dummy atoms with atomic number 0
                if all(num in allowed_atomic_nums or num == 0 for num in atomic_nums):
                    break
                else:
                    logger.warning(f"⚠️ Skipping SMILES with disallowed atoms: {smiles}")
            else:
                logger.warning(f"⚠️ Skipping invalid SMILES: {smiles}")
            i += 1
            if i >= len(smiles_list):
                logger.info("✅ All polymers processed.")
                exit(0)
            smiles = smiles_list[i]

        # Use inverse-log random repeat units
        repeat_units = inverse_log_random(min_val=3, max_val=20)[0]
        atoms = generate_polymer_chain_from_smiles(smiles, repeat_units=repeat_units)
        # --- Get polymer_name for output ---
        if atoms:
            # Try to get a name property, or fallback to index
            if atoms.HasProp("_Name"):
                polymer_name = atoms.GetProp("_Name")
            else:
                polymer_name = f"polymer_{i:03d}"
            # Estimate base monomer size for repeat calculation
            # (Try to parse from SMILES, or fallback to repeat_units)
            # For now, just use repeat_units variable as supplied
            # Embed and optimize
            polymer_chains.append(atoms)
            atoms = embed_and_optimize(atoms, name=polymer_name)
            if atoms is None or atoms.GetNumConformers() == 0:
                logger.warning(f"⚠️ Embedding or optimization failed for polymer {i}")
            else:
                # Try to place molecule in box
                success = try_place_molecule(atoms, spacing=3.5)
                if not success:
                    logger.warning(f"❌ Could not place polymer {i} after multiple tries.")
                else:
                    placed_chains.append(atoms)
                # Save to properly named xyz file
                os.makedirs("/Users/eddypoon/Desktop/electrolyte_loop/02_data_cleaned/polymer_matrices", exist_ok=True)
                filename = f"/Users/eddypoon/Desktop/electrolyte_loop/02_data_cleaned/polymer_matrices/{polymer_name}_n{repeat_units}.xyz"
                with open(filename, "w") as f:
                    f.write(f"{atoms.GetNumAtoms()}\nGenerated single polymer\n")
                    conf = atoms.GetConformer()
                    for atom in atoms.GetAtoms():
                        pos = conf.GetAtomPosition(atom.GetIdx())
                        f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
                print(f"📦 Saved polymer chain to {filename}")
        else:
            logging.warning(f"⚠️ Skipped NoneType polymer for SMILES: {smiles}")
        # Save next index
        with open(state_file, "w") as f:
            f.write(str(i + 1))
    else:
        logger.info("✅ All polymers processed.")

    # Write placed_chains to polymer_matrix.xyz and retain box size logic
    # --- Add support for nanoparticles and solvent ---
    def generate_dummy_nanoparticle(radius=6, structure="icosahedral", center=(0, 0, 0), element="Si"):
        """
        Generate a dummy nanoparticle as an ASE Atoms object.
        structure: "icosahedral", "crystal", or "amorphous"
        """
        if structure == "icosahedral":
            # icosahedral cluster (magic numbers: 13, 55, 147, ...)
            # ASE's Icosahedron uses "n" parameter for number of shells, radius is in Ångstroms
            # We'll estimate number of shells from radius
            # Si atomic radius ~1.1 Å, so n=2-4 for small NP
            shell = max(1, int(radius // 2.5))
            cluster = Icosahedron(element, shell)
            cluster.translate(center)
            return cluster
        elif structure == "crystal":
            # Crystalline spherical nanoparticle (bulk cut to sphere)
            # Use diamond cubic for Si
            a = 5.43  # Si lattice constant in Å
            slab = bulk(element, 'diamond', a=a, cubic=True).repeat((5, 5, 5))
            # Cut to sphere
            positions = slab.get_positions()
            center_arr = np.array(center)
            mask = np.linalg.norm(positions - center_arr, axis=1) <= radius
            atoms = Atoms([atom for i, atom in enumerate(slab) if mask[i]])
            atoms.set_positions(positions[mask])
            return atoms
        elif structure == "amorphous":
            # Generate a random sphere of atoms (very crude amorphous model)
            from ase.ga.utilities import get_random_positions
            n_atoms = int((4/3)*np.pi*radius**3 * 0.045)  # ~0.045 atoms/Å^3 for Si
            pos = get_random_positions(n_atoms, radius*2, min_dist=2.0)
            atoms = Atoms(element * n_atoms, positions=pos - np.mean(pos, axis=0) + center)
            return atoms
        else:
            raise ValueError("Unknown structure type for nanoparticle.")

    # Place polymer matrix and nanoparticles, and combine into final file
    polymer_matrix_atoms = []
    box_size_arr = np.array([70.0, 70.0, 70.0])  # default box size
    if placed_chains:
        # Gather all atom coordinates from all placed chains
        all_coords = []
        for mol in placed_chains:
            conf = mol.GetConformer()
            for j in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(j)
                all_coords.append([pos.x, pos.y, pos.z])
        all_coords = np.array(all_coords)
        if len(all_coords) == 0:
            logger.error("❌ No polymer chains placed successfully. Skipping XYZ generation.")
            os.makedirs("/Users/eddypoon/Desktop/electrolyte_loop/02_data_cleaned/polymer_matrices", exist_ok=True)
            with open("/Users/eddypoon/Desktop/electrolyte_loop/02_data_cleaned/polymer_matrices/polymer_matrix.xyz", "w") as f:
                f.write("0\n\n")
        else:
            min_coords = np.min(all_coords, axis=0)
            max_coords = np.max(all_coords, axis=0)
            buffer = 10.0  # Ångstroms
            box_length = 70.0
            box_size_arr = np.array([box_length, box_length, box_length])
            box_center = box_size_arr / 2
            centroid = np.mean(all_coords, axis=0)
            shift = box_center - centroid
            # Center all polymer chains in the box
            for mol in placed_chains:
                conf = mol.GetConformer()
                for j in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(j)
                    conf.SetAtomPosition(j, [pos.x + shift[0], pos.y + shift[1], pos.z + shift[2]])
            print(f"📦 Final box size: {box_size_arr}")
            # Gather all polymer atoms for output
            for mol in placed_chains:
                conf = mol.GetConformer()
                for atom in mol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    polymer_matrix_atoms.append((atom.GetSymbol(), [pos.x, pos.y, pos.z]))

    # --- Generate nanoparticles ---
    nanoparticle_dir = "/Users/eddypoon/Desktop/electrolyte_loop/02_data_cleaned/polymer_matrices/nanoparticles"
    os.makedirs(nanoparticle_dir, exist_ok=True)
    nanoparticles = []
    nanoparticle_xyz_files = []
    # Example: generate 3 nanoparticles with different structures and radii
    nanoparticle_specs = [
        {"radius": 6, "structure": "icosahedral", "center": (20, 20, 20), "element": "Si", "name": "np_ico"},
        {"radius": 8, "structure": "crystal", "center": (50, 50, 30), "element": "Si", "name": "np_crystal"},
        {"radius": 5, "structure": "amorphous", "center": (35, 55, 55), "element": "Si", "name": "np_amorphous"},
    ]
    for spec in nanoparticle_specs:
        np_atoms = generate_dummy_nanoparticle(radius=spec["radius"], structure=spec["structure"], center=spec["center"], element=spec["element"])
        nanoparticles.append(np_atoms)
        xyz_path = os.path.join(nanoparticle_dir, f"{spec['name']}.xyz")
        ase.io.write(xyz_path, np_atoms)
        nanoparticle_xyz_files.append(xyz_path)
        print(f"🟠 Wrote nanoparticle: {xyz_path} | {len(np_atoms)} atoms")

    # --- (Optional) Generate dummy NMP solvent molecule using ASE (for demonstration) ---
    # In practice, you would use a real solvent molecule and placement logic.
    nmp_atoms = Atoms('C5H9NO', positions=[
        [60, 15, 15], [61, 15, 15], [60, 16, 15], [60, 15, 16], [61, 16, 15],  # C
        [62, 15, 15], [60, 17, 15], [60, 15, 17], [62, 16, 15], [61, 17, 15],  # H
        [60, 15, 18],  # N
        [63, 15, 15]   # O
    ])
    solvent_xyz_path = "/Users/eddypoon/Desktop/electrolyte_loop/02_data_cleaned/polymer_matrices/nmp_solvent.xyz"
    ase.io.write(solvent_xyz_path, nmp_atoms)
    print(f"🟦 Wrote dummy NMP solvent: {solvent_xyz_path}")

    # --- Combine all components into a single final XYZ file ---
    final_xyz_path = "/Users/eddypoon/Desktop/electrolyte_loop/02_data_cleaned/polymer_matrices/polymer_matrix_with_nanoparticles.xyz"
    all_atoms = []
    # Add polymer matrix atoms
    all_atoms.extend(polymer_matrix_atoms)
    # Add solvent atoms (read from xyz)
    if os.path.exists(solvent_xyz_path):
        with open(solvent_xyz_path) as f:
            lines = f.readlines()
            for line in lines[2:]:  # skip first two lines (header)
                parts = line.strip().split()
                if len(parts) == 4:
                    all_atoms.append((parts[0], [float(parts[1]), float(parts[2]), float(parts[3])]))
    # Add nanoparticles atoms (read from xyz files)
    for nanop_xyz in nanoparticle_xyz_files:
        with open(nanop_xyz) as f:
            lines = f.readlines()
            for line in lines[2:]:
                parts = line.strip().split()
                if len(parts) == 4:
                    all_atoms.append((parts[0], [float(parts[1]), float(parts[2]), float(parts[3])]))
    # Write all atoms to the final XYZ file
    with open(final_xyz_path, "w") as f:
        f.write(f"{len(all_atoms)}\n")
        f.write("Polymer matrix + NMP solvent + nanoparticles\n")
        for symbol, pos in all_atoms:
            f.write(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
    print(f"✅ Saved combined structure to {final_xyz_path}")

    # --- Remove any Packmol call from this script (none present) ---
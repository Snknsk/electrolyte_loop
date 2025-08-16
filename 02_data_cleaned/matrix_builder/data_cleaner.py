import logging

# --- Nanoparticle/ceramic bead packing control ---
include_nanoparticles = True
num_nanoparticles = 5
import os
import subprocess
import certifi
# --- ASE nanoparticle generation and PACKMOL helpers ---
from ase.build import bulk, make_supercell
from ase.io import write
from ase.cluster.icosahedron import Icosahedron
import subprocess

# 🔒 Force the use of the certifi certificate bundle for SSL verification
os.environ['SSL_CERT_FILE'] = certifi.where()
print("✅ Using cert file:", certifi.where())
from pymatgen.core.structure import Structure
from pymatgen.core.sites import Site
import random
import numpy as np

# --- Add function to generate small crystalline beads from Materials Project data ---
def make_crystalline_beads(mp_csv_path: str, num_beads: int = 3, bead_radius: float = 7.5):
    import pandas as pd
    from pymatgen.analysis.local_env import VoronoiNN

    df = pd.read_csv(mp_csv_path)
    bead_structs = []

    for _ in range(num_beads):
        # Pick a random entry
        entry = df.sample(1).iloc[0]
        structure_dict = eval(entry['structure'])  # safely use literal_eval if you prefer

        try:
            structure = Structure.from_dict(structure_dict)

            # Expand to a 2x2x2 supercell to make sure we can crop
            structure.make_supercell([2, 2, 2])

            # Get a central point (random site or centroid)
            center = random.choice(structure).coords

            # Extract sphere of atoms around center
            sphere = structure.get_sites_in_sphere(center, r=bead_radius)
            bead = Structure.from_sites([site[0] for site in sphere])

            # Random rotation
            angle = np.radians(random.uniform(0, 360))
            axis = np.random.normal(size=3)
            # Normalize axis
            axis = axis / np.linalg.norm(axis)
            rot_matrix = Structure.get_rot_matrix(angle, axis)
            bead.apply_operation(rot_matrix)

            bead_structs.append(bead)
        except Exception as e:
            print(f"❌ Failed to process structure {entry['material_id']}: {e}")
            continue

    return bead_structs
import ast
import pandas as pd
from pymatgen.core import Structure, Molecule
def make_filler_bead(structure_str: str, radius: float = 10.0) -> Molecule:
    """
    Convert a Materials Project structure string (Pymatgen dict) into a spherical bead as a Pymatgen Molecule.
    """
    structure_dict = ast.literal_eval(structure_str)
    structure = Structure.from_dict(structure_dict)
    structure.make_supercell([3, 3, 3])  # can increase for larger beads

    center = structure.lattice.get_cartesian_coords([0.5, 0.5, 0.5])
    selected_sites = [
        site for site in structure.sites
        if np.linalg.norm(site.coords - center) <= radius
    ]
    bead = Molecule([site.specie for site in selected_sites],
                    [site.coords for site in selected_sites])
    return bead
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolToXYZBlock
import numpy as np

# --- Add get_mol_bounds function for bounding box calculation ---
from rdkit.Chem import rdMolTransforms
def get_mol_bounds(mol, conf_id=0):
    conf = mol.GetConformer(conf_id)
    coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    min_x, min_y, min_z = map(min, zip(*coords))
    max_x, max_y, max_z = map(max, zip(*coords))
    return (min_x, min_y, min_z), (max_x, max_y, max_z)

def generate_multiple_nanoparticles(n: int = 3, radius: float = 5.0):
    """
    Generate n nanoparticles as pymatgen Molecule objects and return as a list.
    """
    import pandas as pd
    from pymatgen.core.structure import Structure
    nanoparticles = []
    df = pd.read_csv("/Users/eddypoon/Desktop/electrolyte_loop/01_data_raw/data_collection/raw/materials_project.csv")
    sampled = df.sample(n)
    for i, row in enumerate(sampled.itertuples()):
        try:
            struct_dict = eval(row.structure)
            struct = Structure.from_dict(struct_dict)
            struct.make_supercell([2, 2, 2])
            center = random.choice(struct).coords
            sphere = struct.get_sites_in_sphere(center, r=radius)
            bead = Structure.from_sites([s[0] for s in sphere])
            # Convert to Molecule for easy xyz output
            from pymatgen.core import Molecule
            mol = Molecule([site.specie for site in bead.sites], [site.coords for site in bead.sites])
            nanoparticles.append(mol)
        except Exception as e:
            print(f"❌ Failed to generate nanoparticle {i}: {e}")
    return nanoparticles

def generate_packmol_input(box_size=60.0):
    """
    Generate a Packmol input file using consolidated 'polymer_matrix.xyz' and 'solvent.xyz' files.
    """
    lines = []
    lines.append("nloop0 1000")
    lines.append("tolerance 2.0")
    lines.append("filetype xyz")
    lines.append("output packed_matrix.xyz\n")
    # Polymer matrix
    lines.append("structure polymer_matrix.xyz")
    lines.append("  number 1")
    lines.append(f"  inside box 0. 0. 0. {box_size} {box_size} {box_size}.")
    lines.append("end structure\n")
    # Solvent
    lines.append("structure solvent.xyz")
    lines.append("  number 1")
    lines.append(f"  inside box 0. 0. 0. {box_size} {box_size} {box_size}.")
    lines.append("end structure\n")
    with open("packmol_input.inp", "w") as f:
        for line in lines:
            f.write(line + "\n")


"""
tolerance 2.0
filetype xyz
output packed_matrix.xyz
seed 12345

structure solvent_output/nmp.xyz
  number 200
  inside box 0. 0. 0. 80. 80. 80.
end structure

    # Add all polymer chain files
    for file in sorted(os.listdir("matrix_output")):
        if file.endswith(".xyz") and "chain" in file:
structure matrix_output/{file}
  number 1
  inside box 0. 0. 0. 80. 80. 80.
end structure

    # Add nanoparticles
    for i in range(3):
structure nanoparticle_{i}.xyz
  number 2
  inside box 0. 0. 0. 80. 80. 80.
end structure
"""


# Import logger for HSP prediction
import logging
logger = logging.getLogger(__name__)


def join_monomers_by_atommap(monomer_smiles: str, repeat: int) -> Chem.Mol:
    print("✅ join_monomers_by_atommap called")

    try:
        mol = Chem.MolFromSmiles(monomer_smiles)
        if mol is None:
            print("❌ MolFromSmiles returned None")
            return None

        # Map wildcards
        head = None
        tail = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == '*' and atom.HasProp('molAtomMapNumber'):
                mapnum = atom.GetIntProp('molAtomMapNumber')
                if mapnum == 1:
                    head = atom.GetIdx()
                elif mapnum == 2:
                    tail = atom.GetIdx()
        if head is None or tail is None:
            print("❌ Could not find both head and tail wildcards")
            return None

        monomer = Chem.RWMol(mol)
        monomer.RemoveAtom(max(head, tail))
        monomer.RemoveAtom(min(head, tail))

        monomer = monomer.GetMol()
        Chem.SanitizeMol(monomer)

        polymer = Chem.RWMol(monomer)
        prev_tail_idx = None
        offset = monomer.GetNumAtoms()

        for i in range(1, repeat):
            frag = Chem.RWMol(monomer)
            amap = {}

            for atom in frag.GetAtoms():
                new_idx = polymer.AddAtom(atom)
                amap[atom.GetIdx()] = new_idx

            for bond in frag.GetBonds():
                polymer.AddBond(amap[bond.GetBeginAtomIdx()],
                                amap[bond.GetEndAtomIdx()],
                                bond.GetBondType())

            # Join the last atom of the previous copy to the first atom of the new copy
            tail_idx = offset - 1
            head_idx = offset
            polymer.AddBond(tail_idx, head_idx, Chem.BondType.SINGLE)
            offset += monomer.GetNumAtoms()

        final_polymer = polymer.GetMol()
        Chem.SanitizeMol(final_polymer)
        print("✅ Polymer successfully built and sanitized")
        print(f"🧪 Final polymer atom count: {final_polymer.GetNumAtoms()}")
        return final_polymer

    except Exception as e:
        print(f"❌ Exception during join_monomers_by_atommap: {e}")
        return None


if __name__ == "__main__":
    print("⚡ Script started")

    # Set of indices (0-based) for crystalline chains
    crystalline_indices = {0, 2, 5}  # set of indices (0-based) for crystalline chains

    test_smiles = "[*:1]CCO[*:2]"
    # Polymer chain count and chain lengths (randomized)
    num_polymer_chains = 30
    chain_lengths = np.random.randint(15, 40, size=num_polymer_chains)

    os.makedirs("matrix_output", exist_ok=True)

    from rdkit.Chem.rdMolTransforms import ComputeCentroid
    from rdkit.Chem import AllChem

    # --- Add molecule optimization function ---
    def optimize_molecule(mol, ff='UFF'):
        from rdkit.Chem import AllChem
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
    polymer_chains = []
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

    for idx in range(num_polymer_chains):
        is_cryst = idx in crystalline_indices
        # --- PATCH: restrict repeat count range to embeddable values ---
        repeat_count = random.randint(5, 12)
        crystal_style = 'crystalline' if is_cryst else 'amorphous'
        logger.info(f"🔁 [{idx}] Using repeat count: {repeat_count} | {crystal_style}")
        polymer_smiles = test_smiles  # or use a variable if different per chain
        # For debugging, log the SMILES string for this chain
        logger.debug(f"SMILES for chain_{idx}_len{repeat_count}: {polymer_smiles}")

        polymer = join_monomers_by_atommap(polymer_smiles, repeat_count)
        if not polymer:
            print(f"❌ Skipping chain {idx+1}")
            continue

        # Use robust conformer embedding and optimization
        name = f"chain_{idx+1}_len{repeat_count}"
        mol = embed_and_optimize(polymer, name)
        if mol is None:
            print(f"❌ Chain {idx+1} failed conformer embedding/optimization (including fallback).")
            continue
    used_ligpargen = False
    try:
        ligpargen_dir = call_ligpargen(Chem.MolToSmiles(mol), name)
        if ligpargen_dir is not None:
            used_ligpargen = True
        AllChem.AlignMolConformers(mol)
        xyz_block = MolToXYZBlock(mol)
        filename = f"matrix_output/chain_{idx+1}_len{repeat_count}.xyz"
        with open(filename, "w") as f:
            f.write(xyz_block)
        print(f"📦 Saved {filename} | FF used: {'OPLS-AA' if used_ligpargen else 'UFF fallback'}")
        polymer_chains.append((mol, is_cryst, idx))
    except Exception as e:
        print(f"⚠️ Failed to save chain {idx+1}: {e}")

    # --- Estimate box size dynamically based on molecule content ---
    # (polymer_chains and nmp_molecules will be combined later, but for box size estimation, use both)
    # If nmp_molecules not yet created, just use polymer_chains for now and add NMP below.
    # We'll add NMP molecules after box size is estimated.
    # Dynamically estimate box size from molecule count and target density
    # First, prepare for NMP molecules (but not yet created)
    nmp_smiles = "CN1CCCC1=O"  # SMILES for N-Methyl-2-pyrrolidone
    num_nmp = 1000  # Increase solvent density
    # --- Random 3D coordinate generator with minimum separation ---
    def random_positions(n, box_size, min_dist=5):
        positions = []
        while len(positions) < n:
            candidate = np.random.uniform(0, box_size, size=3)
            if all(np.linalg.norm(candidate - p) > min_dist for p in positions):
                positions.append(candidate)
        return positions
    # For box estimate, create dummy NMP mols for atom count
    nmp_mol = Chem.MolFromSmiles(nmp_smiles)
    nmp_mol = Chem.AddHs(nmp_mol)
    # --- LigParGen for NMP solvent molecule ---
    used_ligpargen_nmp = False
    try:
        ligpargen_nmp_dir = call_ligpargen(nmp_smiles, "NMP")
        if ligpargen_nmp_dir is not None:
            used_ligpargen_nmp = True
    except Exception as ssl_err:
        print(f"⚠️ LigParGen SSL error/exception for NMP: {ssl_err}")
        print(f"⚠️ LigParGen unavailable, skipping and falling back to local UFF/MMFF.")
        # fallback is now handled in call_ligpargen
    nmp_atoms = nmp_mol.GetNumAtoms() if nmp_mol is not None else 0
    # Combine all molecules for box estimation
    all_molecules = polymer_chains + [nmp_mol]*num_nmp if nmp_mol is not None else polymer_chains
    total_atoms = sum(mol.GetNumAtoms() for mol in (m[0] if isinstance(m, tuple) else m for m in all_molecules))
    atom_volume_estimate = 20.0  # Å³ per atom, rough guess
    total_volume = total_atoms * atom_volume_estimate
    box_size = total_volume ** (1/3)  # cubic root to get box side length
    logger.info(f"📦 Estimated box size: {box_size:.2f} Å based on {total_atoms} atoms")

    # --- Print force field status for NMP ---
    print(f"📦 Saved matrix_output/NMP_UFF_fallback.xyz | FF used: {'OPLS-AA' if used_ligpargen_nmp else 'UFF fallback'}")

    # --- Optional: OPLS-AA energy minimization: can be run after placement using OpenMM/ParmEd/LAMMPS ---
    # Here, only embedding is performed; minimization with OPLS-AA should be done after box building.

    # --- Place all polymer chains (randomized, collision-aware, evenly spread) ---
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

    # Place all polymers with staggered spacing
    polymer_list = [polymer for polymer, _, _ in polymer_chains]
    placed_polymers = []
    for i, polymer in enumerate(polymer_list):
        # Only embed (do not UFF/MMFF optimize, as OPLS-AA params will be used)
        AllChem.EmbedMolecule(polymer, AllChem.ETKDG())
        success = try_place_molecule(polymer, spacing=3.5 + i * 0.5)
        if not success:
            logger.warning(f"❌ Could not place polymer {i} after multiple tries.")
        else:
            placed_polymers.append(polymer)

    # --- Compute bounding box for all placed polymers and adjust box_size accordingly ---
    # Gather all atom coordinates from all placed polymers
    all_coords = []
    for mol in placed_polymers:
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            all_coords.append([pos.x, pos.y, pos.z])
    all_coords = np.array(all_coords)
    min_coords = np.min(all_coords, axis=0)
    max_coords = np.max(all_coords, axis=0)
    buffer = 10.0  # Ångstroms
    # Increase the simulation box size for more dispersion
    box_length = 70.0  # in Å, allows for dispersion
    box_size = np.array([box_length, box_length, box_length])
    box_center = box_size / 2
    centroid = np.mean(all_coords, axis=0)
    shift = box_center - centroid
    # Center all polymer chains in the box
    for mol in placed_polymers:
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, [pos.x + shift[0], pos.y + shift[1], pos.z + shift[2]])
    print(f"📦 Final box size: {box_size}")

    # Save all chains into one XYZ file as the polymer matrix
    with open("matrix_output/polymer_matrix.xyz", "w") as f:
        total_atoms = sum(mol.GetNumAtoms() for mol in placed_polymers)
        f.write(f"{total_atoms}\n\n")
        for mol in placed_polymers:
            conf = mol.GetConformer()
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
    print("🧩 Saved full matrix to polymer_matrix.xyz")

    # --- Add NMP solvent molecules after all polymers have been inserted ---
    nmp_molecules = []
    nmp_solvent = Chem.MolFromSmiles(nmp_smiles)
    nmp_solvent = Chem.AddHs(nmp_solvent)
    AllChem.EmbedMolecule(nmp_solvent, AllChem.ETKDG())
    # Do not optimize with UFF/MMFF; OPLS-AA params from LigParGen will be used
    solvent_list = [Chem.Mol(nmp_solvent) for _ in range(num_nmp)]
    # Use new box_center from centering step above
    # Try to place solvents using increased max_attempts and updated box_size/center
    for i, solvent in enumerate(solvent_list):
        AllChem.EmbedMolecule(solvent, AllChem.ETKDG())
        conf = solvent.GetConformer()
        centroid = np.mean([conf.GetAtomPosition(j) for j in range(solvent.GetNumAtoms())], axis=0)
        rand_offset = np.random.uniform(
            low=[-0.5 * s for s in box_size],
            high=[0.5 * s for s in box_size],
            size=3
        )
        target_center = np.array(box_center) + rand_offset
        translation = target_center - centroid
        for j in range(solvent.GetNumAtoms()):
            pos = np.array(conf.GetAtomPosition(j))
            conf.SetAtomPosition(j, (pos + translation).tolist())
        success = try_place_molecule(solvent, spacing=2.8, max_attempts=500)
        if not success:
            logger.warning(f"⚠️ Could not place solvent molecule {i}")
        else:
            nmp_molecules.append(solvent)
    print(f"✅ Placed {len(nmp_molecules)} solvent molecules (NMP) in box.")

    # Combine all molecules into one mol object using collision-aware, randomized placement
    from rdkit.Chem import rdmolops
    all_mols = placed_polymers + nmp_molecules

    # --- Add ceramic bead substructures as nanoscale crystallites from materials_project.csv ---
    from pymatgen.core import Structure
    ceramic_df = pd.read_csv("/Users/eddypoon/Desktop/electrolyte_loop/01_data_raw/data_collection/raw/materials_project.csv")

    bead_structures = []
    for i in range(3):  # Add 3 ceramic beads
        struct = Structure.from_str(ceramic_df.iloc[i]["structure"], fmt="json")
        coords = struct[0].coords
        bead = struct.get_sites_in_sphere(coords, r=5)  # ~1 nm bead
        bead_struct = Structure.from_sites([s[0] for s in bead])
        bead_struct.translate_sites(range(len(bead_struct)), np.random.uniform(10, 60, size=3))
        bead_structures.append(bead_struct)

    # Add the ceramic bead atoms to your final structure merger before output
    all_atoms = list(all_mols)
    all_atoms += bead_structures

    final_mol = None
    for mol in all_atoms:
        if final_mol is None:
            final_mol = mol
        else:
            final_mol = rdmolops.CombineMols(final_mol, mol)

    # --- Add ceramic beads using new function ---
    def add_ceramic_beads(box, bead_xyz_file, n_beads=5, spacing=10.0):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import random
        bead = Chem.MolFromXYZFile(bead_xyz_file)
        for i in range(n_beads):
            conf = Chem.Conformer(bead.GetNumAtoms())
            for j in range(bead.GetNumAtoms()):
                pos = bead.GetConformer().GetAtomPosition(j)
                conf.SetAtomPosition(j, (pos.x + random.uniform(-spacing, spacing),
                                         pos.y + random.uniform(-spacing, spacing),
                                         pos.z + random.uniform(-spacing, spacing)))
            bead.AddConformer(conf, assignId=True)
            box.append(bead)

    # Example call after placing polymers and NMPs
    add_ceramic_beads(all_mols, "path/to/bead.xyz", n_beads=5)

    # Save updated matrix with solvent
    def write_xyz(mol, filename):
        conf = mol.GetConformer()
        with open(filename, "w") as f:
            f.write(f"{mol.GetNumAtoms()}\nGenerated polymer matrix with NMP\n")
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                f.write(f"{atom.GetSymbol()} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}\n")

    write_xyz(final_mol, "matrix_output/polymer_matrix_with_nmp.xyz")
    print("🧩 Saved polymer matrix with NMP to polymer_matrix_with_nmp.xyz")

    # --- Save topologies for simulation (LAMMPS/OpenMM) using LigParGen-generated files ---
    # For LAMMPS: use the .lmp file from LigParGen (load in your LAMMPS script)
    # For OpenMM: use the .prmtop/.inpcrd or .xml from LigParGen (load in your OpenMM script)
    print("🔗 Use LigParGen-generated .itp/.top/.prm/.lmp/.prmtop/.xml files for force field assignment in your simulation engine.")

    # --- Final system summary ---
    num_polymers = len(placed_polymers)
    num_solvents = len(nmp_molecules)
    num_beads = 5  # as set above
    print(f"🔍 Final system summary:\n   ✅ Polymers placed: {num_polymers}\n   ✅ Solvents placed: {num_solvents}\n   ✅ Ceramic clusters placed: {num_beads}")



    # --- Generate small crystalline beads from Materials Project data and save as XYZ ---
    mp_csv_path = "/Users/eddypoon/Desktop/electrolyte_loop/01_data_raw/data_collection/raw/materials_project.csv"
    beads = make_crystalline_beads(mp_csv_path, num_beads=3, bead_radius=7.5)

    for i, bead in enumerate(beads):
        bead.to(fmt="xyz", filename=f"bead_{i+1}.xyz")
        print(f"📦 Saved bead_{i+1}.xyz with {len(bead)} atoms")


    # ------------- New: Add ceramic beads from Materials Project structures (pymatgen/ase) -------------
    from pymatgen.core.structure import Structure
    import pandas as pd
    import random

    def load_materials_project_structures(csv_path: str, n_samples: int = 5) -> list[Structure]:
        df = pd.read_csv(csv_path)
        struct_col = df.columns[-1]  # Assume structure is in the last column
        structures = []
        for i in range(len(df)):
            try:
                s_dict = eval(df[struct_col].iloc[i])
                s = Structure.from_dict(s_dict)
                structures.append(s)
            except Exception as e:
                continue
        return random.sample(structures, min(n_samples, len(structures)))

    def add_ceramic_beads(polymer_system, ceramics: list[Structure], n_beads: int = 1, bead_radius: float = 6.0):
        from ase.build import make_supercell
        from ase import Atoms
        from pymatgen.io.ase import AseAtomsAdaptor
        import numpy as np

        bead_structures = []
        for s in ceramics[:n_beads]:
            supercell = make_supercell(AseAtomsAdaptor().get_atoms(s), np.eye(3) * 2)  # Make it slightly bigger
            atoms = supercell.copy()
            # Use box_size variable if available, else default to 100
            try:
                box_dim = box_size if isinstance(box_size, (float, int)) else float(np.max(box_size))
            except Exception:
                box_dim = 100.0
            atoms.translate(np.random.uniform(low=0, high=box_dim, size=3))  # Random position in box
            bead_structures.append(atoms)

        # Combine the beads into the polymer system (assuming ASE Atoms)
        # If polymer_system is an ASE Atoms object, sum them
        try:
            from ase import Atoms
            if isinstance(polymer_system, Atoms):
                for bead in bead_structures:
                    polymer_system += bead
                return polymer_system
            else:
                # Not ASE Atoms, just return as is (or handle conversion as needed)
                return polymer_system
        except ImportError:
            return polymer_system

    # Usage example (uncomment and adapt as needed for your workflow):
    # ceramics = load_materials_project_structures("/Users/eddypoon/Desktop/electrolyte_loop/01_data_raw/data_collection/raw/materials_project.csv")
    # final_system = add_ceramic_beads(final_system, ceramics, n_beads=3)



# 🔥 Clean up any conflicting input or output files
for fname in ["packmol_input.inp", "packed_matrix.xyz", "nanoparticle.xyz"]:
    if os.path.exists(fname):
        os.remove(fname)

### --- Consolidate polymer chains and NMP molecules into single files for Packmol ---
box_size = 60  # Default box size, can be adjusted if needed


# --- Polymer chain generation from SMILES in CSV ---
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.io import read
from io import StringIO
from typing import Optional

def generate_polymer_chain_from_smiles(smiles: str, chain_length: int = 10) -> Optional[Atoms]:
    try:
        # Remove wildcard atoms before RDKit processing
        smiles = smiles.replace("*", "")
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
        xyz_str = Chem.MolToXYZBlock(mol)
        atoms = read(StringIO(xyz_str), format="xyz")
        return atoms
    except Exception as e:
        logging.warning(f"⚠️ Failed to build polymer chain from SMILES: {smiles} — {e}")
        return None

polymer_df = pd.read_csv("/Users/eddypoon/Desktop/electrolyte_loop/01_data_raw/data_collection/raw/polymer_electrolytes.csv")
smiles_list = polymer_df["SMILES"].dropna().tolist()

polymer_chains = []
for smiles in smiles_list:
    # Remove wildcard atoms before processing
    smiles = smiles.replace("*", "")
    atoms = generate_polymer_chain_from_smiles(smiles)
    if atoms:
        polymer_chains.append(atoms)
    else:
        logging.warning(f"⚠️ Skipped NoneType polymer for SMILES: {smiles}")

# Write valid polymer chains to XYZ
with open("polymer_matrix.xyz", "w") as f:
    for atoms in polymer_chains:
        try:
            xyz_str = atoms.write("xyz", format="xyz", plain=True)
        except Exception:
            # fallback: try via ase.io.write to string
            from ase.io import write as ase_write
            import io
            buf = io.StringIO()
            ase_write(buf, atoms, format="xyz")
            xyz_str = buf.getvalue()
        f.write(f"{xyz_str.strip()}\n")

# Generate NMP/solvent molecules as a single file
nmp_smiles = "CN1CCCC1=O"
from rdkit import Chem
from rdkit.Chem import AllChem
# Remove wildcard atoms before RDKit processing (not expected for NMP, but for generality)
nmp_smiles = nmp_smiles.replace("*", "")
nmp_mol = Chem.MolFromSmiles(nmp_smiles)
nmp_mol = Chem.AddHs(nmp_mol)
AllChem.EmbedMolecule(nmp_mol)
nmp_molecules = [nmp_mol] * 10  # Example: 10 solvent molecules; adjust as needed
with open("solvent.xyz", "w") as f:
    for i, mol in enumerate(nmp_molecules):
        xyz_block = Chem.MolToXYZBlock(mol)
        f.write(f"{xyz_block.strip()}\n")

# Generate Packmol input referencing consolidated files
generate_packmol_input(box_size=box_size)

# 🧪 Debug output: show actual Packmol input
print("🧪 DEBUG: Contents of packmol_input.inp:")
with open("packmol_input.inp") as f:
    print(f.read())

# ✅ Run Packmol with the intended input
try:
    subprocess.run("packmol < packmol_input.inp", shell=True, check=True)
    print("✅ PACKMOL ran successfully.")
except subprocess.CalledProcessError:
    print("❌ PACKMOL failed.")
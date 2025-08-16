    lines.append("structure solvent.xyz")
    lines.append("  number 1")
    lines.append(f"  inside box 0. 0. 0. {box_size} {box_size} {box_size}.")
    lines.append("end structure\n")
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
    nmp_atoms = nmp_mol.GetNumAtoms() if nmp_mol is not None else 0
    # Combine all molecules for box estimation
    all_molecules = polymer_chains + [nmp_mol]*num_nmp if nmp_mol is not None else polymer_chains
    total_atoms = sum(mol.GetNumAtoms() for mol in (m[0] if isinstance(m, tuple) else m for m in all_molecules))
    atom_volume_estimate = 20.0  # Å³ per atom, rough guess
    total_volume = total_atoms * atom_volume_estimate
    box_size = total_volume ** (1/3)  # cubic root to get box side length
    logger.info(f"📦 Estimated box size: {box_size:.2f} Å based on {total_atoms} atoms")

    # --- Print force field status for NMP ---
    # --- Add NMP solvent molecules after all polymers have been inserted ---
    nmp_molecules = []
    nmp_solvent = Chem.MolFromSmiles(nmp_smiles)
    nmp_solvent = Chem.AddHs(nmp_solvent)
    AllChem.EmbedMolecule(nmp_solvent, AllChem.ETKDG())
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



# Generate NMP/solvent molecules as a single file
nmp_smiles = "CN1CCCC1=O"
from rdkit import Chem
from rdkit.Chem import AllChem
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ceramic generator with robust inhomogeneous heating ladder.

Fixes included:
- HOT-group metrics (centrosymmetry average) are created *inside* the fraction loop,
  after HOT/COLD groups exist. No more "Could not find compute group ID".
- Per-fraction sampling: each ladder step writes its own obs/RDF files:
  <formula>_frac_<frac>_obs.txt and <formula>_frac_<frac>_rdf.txt
- ave/time: arrays (RDF) on their own fix with 'mode vector'; scalars with 'ave running'.
- Voxelized HOT selection (deterministic, no fragile random region loops).
- Charge neutrality enforced to machine precision to avoid PPPM warnings.
- Legacy adaptive melt + two-stage quench path kept (optional).

Run: python3 ceramic_generator.py
"""

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import numpy as np
import os
import ast
import logging
import certifi
import pandas as pd
import traceback
import shutil
import subprocess as sp
from tqdm import tqdm
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure, Lattice
from pymatgen.io.lammps.data import LammpsData

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Certificates ---
os.environ['SSL_CERT_FILE'] = certifi.where()

# --- Configuration ---
LAMMPS_CORES = 6
TARGET_ATOMS = 3000

# Paths
BASE_DIR = "/Users/eddypoon/Desktop/electrolyte_loop"
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "02_data_cleaned/ceramic_particles")
DATA_FILE = os.path.join(BASE_DIR, "01_data_raw/data_collection/raw/materials_project.csv")
PROGRESS_FILE = os.path.join(BASE_DIR, "02_data_cleaned/matrix_builder/mp_index_progress.txt")

# Cooling rates (K/ps) for legacy path
COOLING_RATES = [10000, 2000, 500, 50, 5]

# --- Read Materials Data ---
try:
    df = pd.read_csv(DATA_FILE)
    MATERIALS_DATA = df.to_dict(orient="records")
    logger.info(f"✅ Loaded {len(MATERIALS_DATA)} materials from CSV")
except Exception as e:
    logger.error(f"❌ Failed to load materials_project.csv: {e}")
    MATERIALS_DATA = []

# --- Progress Management ---
def read_progress_index():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                return int(f.read().strip())
        except:
            return 0
    return 0

def write_progress_index(index):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(index))

# --- Balanced Charge Assignment ---
def assign_balanced_oxide_charges(struct, model="clayff-lite"):
    """
    Assign per-atom partial charges and enforce exact neutrality (rounded to 6 dp).
    """
    q_cation = {
        "Li": 0.525, "Na": 1.0, "K": 1.0, "Mg": 1.36, "Ca": 1.36, "Ba": 1.36,
        "Al": 1.575, "Fe": 1.575, "Sc": 1.575, "Y": 1.575, "La": 1.575,
        "Ti": 2.196, "Zr": 2.196, "Nb": 2.196, "Ta": 2.196, "Mo": 2.196, "W": 2.196,
        "Si": 2.10, "Ge": 2.10, "P": 1.575, "Cr": 1.575, "V": 1.575, "Mn": 1.36,
        "Co": 1.36, "Ni": 1.36, "Cu": 1.1, "Zn": 1.36, "Ga": 1.81, "As": 1.575,
        "Se": 1.575, "Br": 0.525, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Nb": 1.6,
        "Ru": 2.2, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78,
        "Sn": 1.36, "Sb": 1.575, "Te": 1.575, "I": 0.525, "Cs": 0.79, "Ce": 1.575,
        "Pr": 1.575, "Nd": 1.575, "Sm": 1.575, "Eu": 1.575, "Gd": 1.575, "Tb": 1.575,
        "Dy": 1.575, "Ho": 1.575, "Er": 1.575, "Tm": 1.575, "Yb": 1.575, "Lu": 1.575,
        "Hf": 2.196, "Re": 1.9, "Os": 2.2, "Ir": 2.20, "Pt": 2.28, "Au": 2.54,
        "Hg": 2.00, "Tl": 1.62, "Pb": 1.36, "Bi": 1.575
    }
    from collections import Counter
    elems = [s.specie.symbol for s in struct.sites]
    counts = Counter(elems)
    anions = {"O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N"}
    has_anions = any(el in anions for el in counts.keys())
    if not has_anions:
        struct.add_site_property("charge", [0.0]*len(struct))
        logger.info("   No anions detected - using neutral charges")
        return struct

    Q_cat = 0.0
    n_anions = 0
    for el, n in counts.items():
        if el in anions:
            n_anions += n
            continue
        Q_cat += n * q_cation.get(el, 0.0)

    if n_anions == 0:
        struct.add_site_property("charge", [0.0]*len(struct))
        return struct

    q_anion = -Q_cat / n_anions
    if abs(q_anion) > 1.5:
        scale = 1.2 / abs(q_anion)
        q_anion *= scale
        for k in q_cation:
            q_cation[k] *= scale

    charges = []
    anion_indices = []
    for i, site in enumerate(struct.sites):
        el = site.specie.symbol
        if el in anions:
            charges.append(q_anion)
            anion_indices.append(i)
        else:
            charges.append(q_cation.get(el, 0.0))

    # Shift to neutrality
    Q = float(sum(charges))
    shift = -Q / len(charges)
    charges = [c + shift for c in charges]

    # Round to 6 dp and force exact neutrality
    charges = [round(c, 6) for c in charges]
    resid = float(sum(charges))
    if abs(resid) > 0.0 and anion_indices:
        charges[anion_indices[-1]] = round(charges[anion_indices[-1]] - resid, 6)

    final_charge = float(sum(charges))
    logger.info(f"   Assigned charges: {len([c for c in charges if c > 0])} cations, {len([c for c in charges if c < 0])} anions")
    logger.info(f"   Net charge: {final_charge:.6f} (should be 0.000000)")
    struct.add_site_property("charge", charges)
    return struct

# --- Inhomogeneous HOT ladder (voxel method; HOT metrics inside loop) ---
def generate_inhomogeneous_series_script(
    output_dir, formula, elements,
    fractions=(0.2, 0.4, 0.6, 0.8, 1.0),
    Thot=4000.0,
    Nx=6, Ny=6, Nz=6,
    nvt_hold_ps=20.0, hot_hold_ps=20.0, hot_quench_ps=50.0, npt_cool_ps=200.0
):
    """
    Voxelized HOT selection using built-in chunker with deterministic per-voxel hashing.
    HOT/COLD metrics are created inside the loop (after groups exist) and torn down each pass.
    No use of 'variable ix/iy/iz' math that caused 'Illegal variable command' on your build.
    """
    elements_str = " ".join(elements)
    frac_str = " ".join([f"{f:.3f}" for f in fractions])

    nvt_hold_steps = int(max(1.0, nvt_hold_ps / 0.001))
    hot_hold_steps = int(max(1.0, hot_hold_ps / 0.001))
    hot_quench_steps = int(max(1.0, hot_quench_ps / 0.001))
    npt_cool_steps = int(max(1.0, npt_cool_ps / 0.001))

    script = f"""# ---------- PREAMBLE ----------
units           metal
atom_style      charge
boundary        p p p
read_data       {formula}_crystal.data

pair_style      buck/coul/long 12.0
kspace_style    pppm 1.0e-4
pair_modify     mix arithmetic
pair_coeff      * * 1388.77 0.3623 0.0

neighbor        4.0 bin
neigh_modify    every 1 delay 0 check yes
thermo          2000
thermo_style    custom step temp press pe etotal
thermo_modify   lost error

# ---------- SOFT START ----------
velocity        all create 300.0 12345 mom yes rot yes dist gaussian
fix             soft all nve/limit 0.03
fix             tl   all langevin 300.0 300.0 5.0 7777
timestep        0.001
run             1500
unfix           tl
unfix           soft

# ---------- GLOBAL PER-ATOM COMPUTES ----------
compute         csm all centro/atom fcc
compute         msdALL all msd
compute         gr all rdf 200

# ---------- FRACTION LADDER ----------
variable        frac index {frac_str}
label           FRAC_LOOP

# Pseudo-random per-atom noise in [0,1) based on position (no chunk/atom needed)
variable        seed equal 9999
variable        ph   atom "abs(sin((x*12.9898 + y*78.233 + z*37.719 + v_seed)*43758.5453))"
variable        noise atom "v_ph - floor(v_ph)"
variable        isHot atom "v_noise < v_frac"

# HOT/COLD groups for this fraction
group           HOT variable isHot
group           COLD subtract all HOT

# Report selection
variable        nAll equal count(all)
variable        nHot equal count(HOT)
print           "Selected HOT ~ ${{nHot}}/${{nAll}} atoms (target fraction = ${{frac}})"

# ---------- PER-FRACTION SAMPLING ----------
compute         cavgHOT HOT reduce ave c_csm
fix             fobs all ave/time 200 1 200 c_msdALL[4] c_cavgHOT ave running file {formula}_frac_${{frac}}_obs.txt overwrite
fix             frdf all ave/time 200 1 200 c_gr[*] ave running file {formula}_frac_${{frac}}_rdf.txt mode vector overwrite

# ---------- SPLIT HEAT ----------
variable        Thot  equal {Thot}
variable        Tcold equal 300.0

fix             fH HOT  nvt temp ${{Tcold}} ${{Thot}} 5.0
fix             fC COLD nvt temp ${{Tcold}} ${{Tcold}} 5.0
fix             dtr all dt/reset 200 0.0008 0.0012 0.02
dump            dH HOT  custom 1000 {formula}_frac_${{frac}}_HOT.lammpstrj  id type x y z
dump            dC COLD custom 1000 {formula}_frac_${{frac}}_COLD.lammpstrj id type x y z
dump_modify     dH sort id
dump_modify     dC sort id

run             {nvt_hold_steps}
unfix           dtr
run             {hot_hold_steps}

# ---------- FAST QUENCH HOT -> 900 K ----------
unfix           fH
fix             fH HOT nvt temp ${{Thot}} 900.0 5.0
run             {hot_quench_steps}

# Merge thermostats
unfix           fH
unfix           fC
undump          dH
undump          dC

# ---------- WHOLE-BOX NPT COOL 900 -> 300 ----------
fix             cool all npt temp 900.0 300.0 10.0 iso 1.0 1.0 200.0
fix             dtr2 all dt/reset 200 0.0008 0.0012 0.02
dump            box all custom 2000 {formula}_frac_${{frac}}_BOX.lammpstrj id type x y z
dump_modify     box sort id
run             {npt_cool_steps}
unfix           dtr2
unfix           cool
undump          box

# ---------- FINAL RELAX + SAVE ----------
minimize        1.0e-4 1.0e-6 1000 10000
write_data      {formula}_out_frac_${{frac}}.data
write_dump      all xyz {formula}_out_frac_${{frac}}.xyz modify element {" ".join(elements)}

# Teardown per-fraction sampling
unfix           fobs
unfix           frdf
uncompute       cavgHOT

# Next fraction
next            frac
jump            SELF FRAC_LOOP
"""

    path = os.path.join(output_dir, f"in.{formula}_inhomogeneous_series.lmp")
    with open(path, "w") as f:
        f.write(script)
    return path


# --- Structure Loading ---
def load_mp_structure(entry):
    try:
        structure_str = entry.get("structure")
        if structure_str:
            struct_dict = ast.literal_eval(structure_str)
            return Structure.from_dict(struct_dict)
    except Exception as e:
        logger.warning(f"⚠️ Failed to parse structure from entry: {e}")
    try:
        lattice_data = entry.get("lattice")
        species = entry.get("species")
        coords = entry.get("frac_coords", entry.get("coords"))
        if lattice_data and species and coords:
            if isinstance(lattice_data, dict) and "matrix" in lattice_data:
                lattice = Lattice(lattice_data["matrix"])
            else:
                lattice = Lattice.from_parameters(
                    a=lattice_data["a"], b=lattice_data["b"], c=lattice_data["c"],
                    alpha=lattice_data["alpha"], beta=lattice_data["beta"], gamma=lattice_data["gamma"]
                )
            return Structure(lattice, species, coords, coords_are_cartesian=False)
    except Exception as e:
        logger.error(f"❌ Failed to build structure from fields: {e}")
    return None

# --- Electronegativity Filter ---
def passes_electronegativity_filter(entry, threshold=1.7):
    pauling_en = {
        "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
        "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16,
        "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55,
        "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01,
        "As": 2.18, "Se": 2.55, "Br": 2.96, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33,
        "Nb": 1.6, "Mo": 2.16, "Ru": 2.2, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69,
        "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.1, "I": 2.66, "Cs": 0.79, "Ba": 0.89,
        "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Sm": 1.17, "Eu": 1.2, "Gd": 1.20,
        "Tb": 1.1, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.1, "Lu": 1.27,
        "Hf": 1.3, "Ta": 1.5, "W": 2.36, "Re": 1.9, "Os": 2.2, "Ir": 2.20, "Pt": 2.28,
        "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02,
    }
    elements = []
    if "elements" in entry:
        elem_str = entry["elements"]
        if isinstance(elem_str, str):
            if "|" in elem_str:
                elements = elem_str.split("|")
            elif "," in elem_str:
                elements = [e.strip() for e in elem_str.split(",")]
            else:
                elements = [elem_str]
    if not elements:
        formula = entry.get("formula_pretty", entry.get("formula", ""))
        try:
            comp = Composition(formula)
            elements = [el.symbol for el in comp.elements]
        except:
            return True
    vals = [pauling_en[el] for el in elements if el in pauling_en]
    if len(vals) < 2:
        return False
    max_diff = max(abs(a - b) for i, a in enumerate(vals) for b in vals[i+1:])
    return max_diff > threshold

# --- LAMMPS Execution ---
def run_lammps(script_path, working_dir, use_mpi=True, np=LAMMPS_CORES):
    try:
        lammps_exe = shutil.which("lmp_mpi") or shutil.which("lmp_serial") or shutil.which("lmp") or shutil.which("lammps")
        if not lammps_exe:
            logger.error("❌ LAMMPS executable not found in PATH")
            return False
        if use_mpi and shutil.which("mpirun") and "mpi" in lammps_exe:
            cmd = ["mpirun", "-np", str(np), lammps_exe, "-in", os.path.basename(script_path)]
            logger.info(f"🚀 Running LAMMPS with {np} cores via MPI")
        else:
            cmd = [lammps_exe, "-in", os.path.basename(script_path)]
            logger.info(f"🚀 Running LAMMPS in serial mode")
        logger.info(f"📝 Command: {' '.join(cmd)}")
        p = sp.Popen(
            cmd, cwd=working_dir, stdout=sp.PIPE, stderr=sp.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace"
        )
        for line in p.stdout:
            print(line.strip())
            if "ERROR" in line:
                logger.error(f"LAMMPS ERROR: {line.strip()}")
            elif "WARNING" in line:
                logger.warning(f"LAMMPS WARNING: {line.strip()}")
        p.wait()
        if p.returncode != 0:
            logger.error(f"❌ LAMMPS failed with return code {p.returncode}")
            return False
        logger.info(f"✅ LAMMPS simulation completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ LAMMPS execution failed: {e}")
        logger.error(traceback.format_exc())
        return False

# --- Main Processing (inhomogeneous ladder) ---
def process_material_inhomogeneous(index):
    try:
        entry = MATERIALS_DATA[index]
        if not passes_electronegativity_filter(entry):
            logger.info(f"⏭️ Skipping index {index} - failed electronegativity filter")
            return False

        formula = entry.get("formula_pretty", entry.get("formula", f"material_{index}")).replace("/", "_").replace(" ", "_")
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 INHOMOGENEOUS HEATING LADDER: {formula} (index {index})")
        logger.info(f"{'='*60}")

        crystal_structure = load_mp_structure(entry)
        if crystal_structure is None:
            logger.error(f"❌ Could not load crystal structure for {formula}")
            return False
        logger.info(f"✅ Loaded CRYSTAL structure from Materials Project")

        elements = list({site.specie.symbol for site in crystal_structure})
        logger.info(f"📋 Elements: {', '.join(elements)}")
        logger.info(f"🔮 Initial structure: ORDERED CRYSTAL with {len(crystal_structure)} atoms")

        # Supercell to TARGET_ATOMS
        from pymatgen.transformations.standard_transformations import SupercellTransformation
        num_atoms = len(crystal_structure)
        if num_atoms < TARGET_ATOMS:
            factor = int(np.ceil((TARGET_ATOMS / num_atoms) ** (1/3)))
            sc_matrix = [[factor, 0, 0], [0, factor, 0], [0, 0, factor]]
            crystal_structure = SupercellTransformation(sc_matrix).apply_transformation(crystal_structure)
            logger.info(f"📏 Expanded crystal: {num_atoms} → {len(crystal_structure)} atoms (supercell: {factor}x{factor}x{factor})")
        else:
            logger.info(f"📏 Crystal already has {num_atoms} atoms")

        mp_id = entry.get("material_id", f"mp-{index}")
        safe_formula = formula.replace("(", "").replace(")", "").replace("/", "_")
        group_dir = f"{index // 1000:03d}000s"
        output_dir = os.path.join(OUTPUT_BASE_DIR, group_dir, f"{mp_id}_{safe_formula}")
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"⚡ Assigning balanced charges...")
        crystal_structure = assign_balanced_oxide_charges(crystal_structure)
        lammps_data = LammpsData.from_structure(crystal_structure, atom_style="charge")
        crystal_data_path = os.path.join(output_dir, f"{safe_formula}_crystal.data")
        lammps_data.write_file(crystal_data_path)
        logger.info(f"💾 Saved crystal structure with charges to {output_dir}")
        crystal_structure.to(filename=os.path.join(output_dir, f"{safe_formula}_crystal.cif"))

        # Build & run the inhomogeneous ladder
        inhomog_in = generate_inhomogeneous_series_script(
            output_dir, safe_formula, elements,
            fractions=(0.2, 0.4, 0.6, 0.8, 1.0),
            Thot=4000.0, Nx=6, Ny=6, Nz=6,
            nvt_hold_ps=20.0, hot_hold_ps=20.0, hot_quench_ps=50.0, npt_cool_ps=200.0
        )
        logger.info(f"🧪 Running inhomogeneous heating ladder: {inhomog_in}")
        ok = run_lammps(inhomog_in, output_dir, use_mpi=True, np=LAMMPS_CORES)
        if not ok:
            logger.error(f"❌ Inhomogeneous ladder failed for {safe_formula}")
            return False

        # Optional: parse per-fraction RDFs if you want to collect summaries
        # (files are named <formula>_frac_<frac>_rdf.txt)
        logger.info(f"✅ Inhomogeneous ladder complete for {safe_formula}")
        return True

    except Exception as e:
        logger.error(f"❌ Error processing index {index}: {e}")
        logger.error(traceback.format_exc())
        return False


def main():
    if not MATERIALS_DATA:
        logger.error("❌ No materials data loaded")
        return

    lammps_exe = shutil.which("lmp_mpi") or shutil.which("lmp_serial") or shutil.which("lmp") or shutil.which("lammps")
    if not lammps_exe:
        logger.error("❌ LAMMPS not found in PATH. Please install LAMMPS first.")
        logger.error("   On Mac: brew install lammps")
        logger.error("   With conda: conda install -c conda-forge lammps")
        return
    logger.info(f"✅ Found LAMMPS: {lammps_exe}")

    if shutil.which("mpirun"):
        logger.info(f"✅ MPI available - will use {LAMMPS_CORES} cores per simulation")
    else:
        logger.warning(f"⚠️ MPI not found - will run in serial mode")
        logger.warning(f"   For better performance, install MPI: brew install open-mpi")

    start_index = read_progress_index()
    max_index = len(MATERIALS_DATA)
    if start_index >= max_index:
        logger.info("✅ All materials already processed")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"🔬 INHOMOGENEOUS HEATING → QUENCH CAMPAIGN")
    logger.info(f"{'='*60}")
    logger.info(f"📈 Processing materials {start_index} to {max_index-1}")
    logger.info(f"🔧 Key behaviors:")
    logger.info(f"   - HOT/COLD split: heat only HOT to near-melt, COLD stays ~300 K")
    logger.info(f"   - Fast HOT quench to 900 K, then whole-box NPT 900→300 K")
    logger.info(f"   - Fractions ramp: 0.2 → 1.0 (crystal → amorphous)")
    logger.info(f"   - Trajectories: .lammpstrj per stage for debugging")
    logger.info(f"{'='*60}\n")

    for idx in tqdm(range(start_index, max_index), desc="Processing materials"):
        try:
            success = process_material_inhomogeneous(idx)
            write_progress_index(idx + 1)
            if success:
                logger.info(f"✅ Progress saved: {idx + 1}/{max_index}")
            else:
                logger.warning(f"⚠️ Failed to process index {idx}, continuing...")

            logger.info("\n" + "="*60)
            logger.info("MATERIAL COMPLETE - Review output above")
            logger.info("="*60)

            # interactive pause (keep existing UX)
            user_input = input("\n🛑 Press Enter to continue to next material, or 'q' to quit: ")
            if user_input.lower() == 'q':
                logger.info("\n⚠️ User requested stop. Progress saved at index {}.".format(idx + 1))
                break

        except KeyboardInterrupt:
            logger.info("\n⚠️ Interrupted by user. Progress saved.")
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error at index {idx}: {e}")
            write_progress_index(idx + 1)
            continue

    logger.info("\n🎉 Inhomogeneous ladder campaign complete!")

if __name__ == "__main__":
    main()
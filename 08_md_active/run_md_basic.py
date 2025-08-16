#!/usr/bin/env python3
"""
run_md_basic.py   –   minimal MD driver for FLARE potentials
Usage:
    python run_md_basic.py <input_xyz> <gp_json> [nsteps] [T_K]
"""

import sys, time
from pathlib import Path
from ase.io import read, write
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution as MB
from ase import units

from flare import gp
from flare.utils.ase_calculator import FLARE_Calculator

# ────────────────────────── parse CLI ──────────────────────────
if len(sys.argv) < 3:
    sys.exit("Usage: run_md_basic.py box.xyz gp.json [nsteps] [T_K]")

box_xyz   = Path(sys.argv[1])
gp_file   = Path(sys.argv[2])
nsteps    = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
temp_K    = float(sys.argv[4]) if len(sys.argv) > 4 else 350.0

# ────────────────────────── setup atoms & calculator ───────────
atoms = read(box_xyz)
gp_model = gp.GAPotential.from_file(gp_file)
atoms.calc = FLARE_Calculator(gp_model)

MB(atoms, temperature_K=temp_K)   # initialise velocities

# ────────────────────────── MD engine ──────────────────────────
dt_fs = 1.0                     # time-step in fs
dyn   = VelocityVerlet(atoms, dt_fs * units.fs,
                       trajectory='md_basic.xyz',
                       logfile='md_basic.log')

def status():
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    T    = ekin / (1.5 * units.kB * len(atoms))
    std  = atoms.calc.results.get('std', [0.0])
    print(f"Step {dyn.nsteps:>6} | Epot {epot:9.3f} eV | T {T:6.1f} K | max σ {max(std):.3f} eV/Å")

dyn.attach(status, interval=100)

t0 = time.time()
dyn.run(nsteps)
print(f"\n⏲️  MD finished in {time.time() - t0:.1f} s.  "
      f"Trajectory → md_basic.xyz | Log → md_basic.log")
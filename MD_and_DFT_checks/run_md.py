from ase.io import read
from flare import md
from flare.utils import ASECalculator
from flare.gp import GAPotential
import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
with open(CONFIG_DIR / "md_params.yaml", "r") as f:
    cfg = yaml.safe_load(f)

atoms = read('inputs/poscars/init.xyz')
gp = GAPotential.from_file('data/models/gp_model.json')
calc = ASECalculator(gp)
atoms.set_calculator(calc)

md.run_md(
    atoms,
    timestep=cfg['timestep'],
    temperature=cfg['temperature'],
    md_steps=cfg['md_steps'],
    std_tolerance_factor=cfg['std_tolerance_factor'],
    freeze_atoms=cfg.get('freeze_atoms', []),
    logfile=cfg['logfile'],
    trajectory=cfg['trajectory']
)
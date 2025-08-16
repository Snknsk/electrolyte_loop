from ase.io import read, write
from ase.calculators.espresso import Espresso
import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
with open(CONFIG_DIR / "dft_params.yaml", "r") as f:
    dft_cfg = yaml.safe_load(f)

atoms = read('data/dft_outputs/to_dft.xyz')
calc = Espresso(
    input_data={
        'calculation': dft_cfg['control']['calculation'],
        'ecutwfc': dft_cfg['system']['ecutwfc'],
        'ecutrho': dft_cfg['system']['ecutrho'],
        'occupations': dft_cfg['system']['occupations'],
        'smearing': dft_cfg['system']['smearing'],
        'degauss': dft_cfg['system']['degauss'],
        'conv_thr': dft_cfg['electrons']['conv_thr'],
    },
    pseudopotentials={'Li': 'Li.pbe-n-kjpaw_psl.1.0.0.UPF'},
    pseudo_dir=dft_cfg['control']['pseudo_dir'],
    kpts=dft_cfg['k_points']['grid']
)

atoms.set_calculator(calc)
atoms.get_potential_energy()
atoms.get_forces()

write('data/training_data/labeled.xyz', atoms)
import shutil
import subprocess
import importlib
import sys

def check_executable(name):
    return shutil.which(name) is not None

def check_module(name):
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False

def print_header(title):
    print("\n" + "=" * (len(title) + 4))
    print(f"  {title}")
    print("=" * (len(title) + 4))

def check_lammps():
    print_header("Checking LAMMPS")

    found = False
    if check_executable("lmp"):
        print("✅ Found LAMMPS executable: `lmp`")
        found = True
    elif check_executable("lmp_serial"):
        print("✅ Found LAMMPS executable: `lmp_serial`")
        found = True

    if check_module("lammps"):
        print("✅ Found LAMMPS Python module.")
        found = True

    if not found:
        print("❌ LAMMPS not found.")
        print("📦 Install via Conda:")
        print("    conda install -c conda-forge lammps")
        print("📘 Docs: https://docs.lammps.org/")

def check_quantum_espresso():
    print_header("Checking Quantum ESPRESSO")

    found = False
    for exe in ["pw.x", "cp.x", "ph.x"]:
        if check_executable(exe):
            print(f"✅ Found QE executable: `{exe}`")
            found = True

    if not found:
        print("❌ Quantum ESPRESSO not found.")
        print("📦 Install via Conda:")
        print("    conda install -c conda-forge quantum-espresso")
        print("📘 Docs: https://www.quantum-espresso.org/")

if __name__ == "__main__":
    print("🧪 Environment Check for LAMMPS and Quantum ESPRESSO")
    check_lammps()
    check_quantum_espresso()
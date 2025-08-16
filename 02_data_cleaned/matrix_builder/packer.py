def generate_packmol_input(box_size=60.0):
    """
    lines.append("output packed_matrix.xyz\n")
    # Polymer matrix
    with open("packmol_input.inp", "w") as f:
        for line in lines:
            f.write(line + "\n")


"""
tolerance 2.0
filetype xyz
output packed_matrix.xyz
seed 12345

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
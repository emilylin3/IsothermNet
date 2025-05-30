import subprocess
import os

# List all directories
dir_zeopp = "/home/elin49/zeopp"
path_test_cssr = "/home/elin49/zeopp/CSSR_Files_extra/"          # Path for test .cif files (to be deleted) - zeopp
dir_output_res = "/home/elin49/zeopp/outputs/res"

qmof_cssr = os.listdir(path_test_cssr)    # List of all qmof .cif files 
outputFiles_res = os.listdir(dir_output_res) 

# Remove non-.cssr files - cssr folder
indx_vec = []
for indx in range(len(qmof_cssr)-1):
    if ".cssr" not in qmof_cssr[indx]:
        indx_vec.append(indx)
for i in sorted(indx_vec,reverse=True):
    del qmof_cssr[i]
    
# Remove non-.res files - output folder
indx_vec = []
for indx in range(len(outputFiles_res)-1):
    if ".res" not in outputFiles_res[indx]:
        indx_vec.append(indx)
for i in sorted(indx_vec,reverse=True):
    del outputFiles_res[i]
    
# Remove .cssr files that have already been used (avoid repetition)
for i in outputFiles_res:    
    mof_name = i.replace(".res","")
    if mof_name+".cssr" in os.listdir(path_test_cssr):          # If framework w/ RASPA results is in the folder, delete for next time
        os.remove(path_test_cssr+mof_name+".cssr")

# Start Zeopp loop
os.chdir(dir_zeopp)

# Parameters 
probeDiameter = 2.65                   # Probe diameter [Angstrom] - hard sphere/kinetic diameter of water molecule (2.65 A) - can get from LJ sigma parameter
probeRadius = probeDiameter/2          # Probe radius [Angstrom] 
n_samples = 50000                      # Number of samples 

# Go to directory with all CSSR files
qmof_cssr = os.listdir(path_test_cssr)    # List of all qmof .cssr files 

for i in qmof_cssr:
    qmof = i.replace(".cssr","")     

    # Pore Diameter
    subprocess.run(["./network", "-ha", "-res", "outputs/res/{}.res".format(qmof), "CSSR_Files/{}.cssr".format(qmof)])

    # Surface Area
    subprocess.run(["./network", "-ha", "-sa", "{}".format(str(probeRadius)), "{}".format(str(probeRadius)), "{}".format(str(n_samples)), 
                    "outputs/sa/{}.sa".format(qmof), "CSSR_Files/{}.cssr".format(qmof)])

    # Volume
    subprocess.run(["./network", "-ha", "-vol", "{}".format(str(probeRadius)), "{}".format(str(probeRadius)), "{}".format(str(n_samples)), 
                    "outputs/vol/{}.vol".format(qmof), "CSSR_Files/{}.cssr".format(qmof)])

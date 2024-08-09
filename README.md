# LitE

## Execution Environment:

Operation System: Ubuntu 20.04 (LTS), 64bit.<br />
Physical Memory (RAM) 16.0 GB.<br />


### Prerequisites

Python 3.7.<br />
PyCharm Community Edition 2021.2. <br />
Mininet for VDCE simulation.<br />
An introduction about VDCE problem can be found in below link:<br />
https://www.youtube.com/watch?v=JKB3aVyCMuo&t=506s<br />

### Installation

- ~# git clone https://github.com/mininet/mininet
- ~# cd mininet/
- ~# sudo ./util/install.sh -a
- ~# export PYTHONPATH="$PYTHONPATH:/media/sdn/New Volume/PyCharm Projects - ubuntu/Testing/mininet"
- ~# sudo mn --test pingall
- ~# sudo python run_mininet.py

###   Download LitE and keep it in the drive where Mininet is present. The LitE file contains all executable files related to the proposed and baseline approaches. <br />

- CEVNE.py -> The main file related to the baseline CEVNE approach.<br />
- DROI.py -> The main file related to the baseline DROI approach.<br />
- Energy_Load_Math.py -> The main file related to the proposed LitE approach. <br /> 
- First_Fit.py -> The main file related to the Greedy baseline approach.<br />
- Mininet.py -> The main file related to the PN topology generation.<br />
- SN-Input-File.txt -> The main file related to the PN topology generation inputs. <br />
- VNE-Input-File.txt -> The main file related to the VDCR generation inputs. <br />
- VNE.generator.py -> The main file related to the VDCR topology generation. <br />
- manager.py -> The main file is related to the proposed LitE approach manager module. <br />

## Usage

###  In VNE.generator.py, we can set the various parameters related to Virtual Data Center Requests(VDCRs).<br />

- We can set the minimum and maximum number of VDCR VMs.<br />

- We can set the virtual data center request demands like BandWidth(min, max), CRB(min, max), LocationX(min, max), LocationY(min, max), and Delay(min, max) in vdce. Append function. <br />
- Example: (1, 5, 1, 10, 0, 100, 0, 100, 1, 4)<br />

- Run VNE.generator.py after doing any modifications. <br />

###  In Mininet.py:<br />

- In the Mininet function mention the pickle file related to physical network generation.<br />

- In graph.parameters function set physical network resources like BandWidth(min,max), CRB(min,max), LocationX(min,max), LocationY(min,max), Delay(min,max).<br />
- Example: (500, 1000, 200, 1000, 0, 100, 0, 100, 1, 1)<br />

- Run Mininet.py after doing any modification. <br />

### In the manager.py file, set the VDCR size such as [250, 500, 750, 1000] and mention the number of iterations needed to execute each VDCR size in the iteration variable.<br />

- Finally, run the manager.py file. After successfully running, a SN.pickle file is created (If it already does not exist in the specified path). It has all input parameters related to the physical network parameters, such as CRB, Bandwidth, Delay, and Location.

- Final embedding results are captured in Results.xlsx, which includes values for various metrics for all test scenarios for every iteration.

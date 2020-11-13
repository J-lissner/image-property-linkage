# Image based prediction of the heat conduction tensor

This repository provides a graphical user interface to predict the effective heat conductivity of microstructures and requires only the image data as input. The deployed methods are proposed and validated in [this paper](https://www.mdpi.com/2297-8747/24/2/57).<br>
Execute "predict_kappa.py" with python3 in the terminal to start the intuitive GUI. Longer loading times during startup are due to the loading of the surrogate models. Results and output will be printed inside the GUI, a more detailed description is found in the terminal. <br>
20 example microstructures are given in the subfolder "examples/" with their respective heat conduction tensor written into the "heat_conductivities" file.<br>

Required packages:
- pillow
- tensorflow 1.13+ or 2.2-
- default: os, tkinter
- numpy, matplotlib

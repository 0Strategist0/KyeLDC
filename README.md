# Kye's Final Folder
*Kye Emond, August 2022*

This folder contains the initial results of my work. It has two subdirectories and one other file, which I will explain below:

## LearningDocuments

This is a directory in which I'm including some explanatory documents that I wish I'd had when I started learning about the subjects. It includes:

- **CedarUse.pdf**: A file that quickly summarizes how to access and use Cedar via your terminal on a Mac or Linux computer. 

- More to come

## Code

This directory includes all the important code I've written over the summer, as well as files and directories that are necessary for the code to run. All code is well-documented (in my opinion), so it should be easy enough to read through and figure out what everything does. Some files are too big to be stored on Github, in which case I store them on Google Drive and stick a text file with a link to the drive in its place. This directory includes:

- **Final_Notebook.ipynb**: A Jupyter Notebook that explains and implements the entire process of parameter estimation for the verification binaries. It downloads the LISA data, finds glitches, windows glitches, runs an f-statistic gridsearch, runs MCMC samplers, then analyzes results. It is incomplete at the moment because I haven't had time to run all the MCMC samplers needed to estimate parameters for all detected binaries, but I'm hoping to submit those jobs to Cedar and get back some files I can just stick in a directory to finish off the project. Also, some modules imported at the start likely aren't needed, and I never did finish off my full TODO list. It's pretty close to done though. 

- **Data**: Stores the LDC files. 

- **FinalChains**: Stores the MCMC chains for each source. Currently only contains an example chain, but once the MCMC runs are complete, will hold all the MCMC chains. 

- **FinalGridsearches**: Stores the full gridsearch over all of parameter space. 

- **SmallerSearches**: Stores the small gridsearches around the peaks detected in the full gridsearch. 

- **FinalFunctions.py**: A python file that has all the probability functions (fstat, log likelihood, etc...) stored to be easily imported. 

- **KyeLISAModule** A python module containing several submodules, all used to make LISA data analysis faster and easier. 

## requirements.txt

A pip freeze of the environment I was using to develop my code. It probably has extraneous modules, and the versions most likely don't need to be as specific as they are in the freeze, but it should make it easier for others to set up a similar environment. Do note that some of the modules in requirements.txt need to be installed from the LDC gitlab, so you won't be able to directly use pip install -r requirements.txt. 

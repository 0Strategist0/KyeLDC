# Kye's Final Folder
*Kye Emond, August 2022*

## Files and Directories

This repository contains the initial results of my work. It has two subdirectories and one other file, which I will explain below:

### LearningDocuments

This is a directory in which I'm including some explanatory documents that I wish I'd had when I started learning about the subjects. It includes:

- **CedarUse.pdf**: A file that quickly summarizes how to access and use Cedar via your terminal on a Mac or Linux computer. 

- More to come (Never mind. This was a lie)

### Code

This directory includes all the important code I've written over the summer, as well as files and directories that are necessary for the code to run. All code is well-documented (in my opinion), so it should be easy enough to read through and figure out what everything does. Some files are too big to be stored on Github, in which case I store them on Google Drive or on Cedar and stick a text file with a link to the drive or instructions on accessing it in its place. This directory includes:

- **Final_Notebook.ipynb**: A Jupyter Notebook that explains and implements the entire process of parameter estimation for the verification binaries. It downloads the LISA data, finds glitches, windows glitches, runs an f-statistic gridsearch, runs MCMC samplers, then analyzes results. It is incomplete at the moment because I haven't had time to run all the MCMC samplers needed to estimate parameters for all detected binaries, but I'm hoping to submit those jobs to Cedar and get back some files I can just stick in a directory to finish off the project. Also, some modules imported at the start likely aren't needed, and I never did finish off my full TODO list. It's pretty close to done though. 

- **Data**: Stores the LDC files. 

- **FinalChains**: Stores the MCMC chains for each source. Currently only contains an example chain, but once the MCMC runs are complete, will hold all the MCMC chains. 

- **FinalGridsearches**: Stores the full gridsearch over all of parameter space. 

- **SmallerSearches**: Stores the small gridsearches around the peaks detected in the full gridsearch. 

- **FinalFunctions.py**: A python file that has all the probability functions (fstat, log likelihood, etc...) stored to be easily imported. 

- **KyeLISAModule**: A python module containing several submodules, all used to make LISA data analysis faster and easier. 

### requirements.txt

A pip freeze of the environment I was using to develop my code. It probably has extraneous modules, and the versions most likely don't need to be as specific as they are in the freeze, but it should make it easier for others to set up a similar environment. Do note that some of the modules in requirements.txt need to be installed from the LDC gitlab, so you won't be able to directly use pip install -r requirements.txt. 

## Stuff I Would Have Done With More Time

I ended the summer before I got to do all the research I wanted, sadly, so here is a list of extra stuff I would have done with more time:

- Investigate the effects of changing the important of different glitch cutoffs (explained more in the notebook)

- Optimize and improve my (and other people's) code
  - F-Statistic could be made faster in a language like C++ or Rust, I imagine
  - Log-likelihood is terribly optimized. Find some way to avoid doing repeated fast fourier transforms, possibly use a different language to make it faster
  - emcee and ptemcee are both written in Python without automatic stopping. It would be nice to have a faster version that has stopping conditions
  - My gridsearch function doesn't terminate extra Python processes if it crashes. Kind of a bother for debugging, would be good to add a `with Pool() as pool:` statement in there to prevent that issue

- Look into methods of filling in gaps with data that doesn't impact parameter estimation at all

- Look into methods of glitch subtraction, rather than glitch windowing

- Look into performing analysis in wavelet domain - helps avoid noise stationarity issues, apparently faster, according to a paper

- Spend more time trying to figure out the effects of windowing on parameter estimation

- Spend more time comparing results of my method to results with no artefacts

- Check effects of different window rolloff durations

- Test impact of adding and removing TDI T from analysis more extensively

- Read through some of those F-Stat papers that explain how to construct a hexagonal grid exactly the right size to avoid missing sources

- Implement a better maximum finder to make source detection in F-Stat more reliable

- Find a more rigorous way to detect sources than just "it's above this arbitrary line constructed from a rolling median" (ideally one where you can give a probability of a false detection)

- Honestly rewrite most of the code in a language that handles multithreading and multiprocessing better than Python

- Better visualization on the Jupyter Notebook, to help people understand what the process is actually doing

- Figure out where those extra factors of two were coming from in log-likelihood

- Try to derive F-Statistic myself

- Read more about deriving TDI combinations

- Investigate whether comparing A and E F-Stat grids (or even X, Y and Z) and looking for coincident maxima is a more effective way to detect true sources than looking at data above a certain threshhold in an AE F-Stat grid

- Derive conversion from strain to TDI variables

- Apparently someone wrote a GPU-accelerated, vectorized version of FastGB. Implementing could speed up loglike and fstat (mikekatz04.github.io/GBGPU)

- Characterize glitches that my method found, and those it didn't find. Also look at what noise got found and what didn't

- Plot out percent of glitches caught vs. percent of data cut out, optimize the best choice of data to cut out
  - Maybe use glitch power instead of glitches, since we're more interested in removing their power from the data, not each and every glitch

- LISA Pathfinder glitches modelled using shapelets - use shapelets to remove these glitches?

- Investigate how much my method removes transient GWs

- Look into nested sampling as an alternative to MCMC and PTMCMC

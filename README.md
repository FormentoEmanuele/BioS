# README #

Electrical stimulation is an effective method for artificially modulating the activity of the nervous system. However, current stimulation paradigms fail to reproduce the stochastic and asynchronous properties of natural neural activity. 
In *Formento and D'Anna et al., A biomimetic electrical stimulation strategy to induce asynchronous stochastic neural activity, Journal of Neural Engineering, 2020*, we introduced a novel biomimetic stimulation (BioS) strategy that overcomes these limitations.
This repository contains the code of the neural simulations performed in the manuscript. 

### How do I get set up? ###
* Dependencies
    * python 3.7
        * numpy
        * matplotlib
        * neuron
    * [neuron](http://www.neuron.yale.edu/neuron/download)
        * --with-python

* Configuration

    The folder /mod_files contains the *NEURON* AXNODE.mode file developed by [C. McIntyre et al. 2002](https://doi.org/10.1152/jn.00353.2001) modelling the membrane dynamics of the afferent fibers here implemented. This file needs to be compiled. For this purpose issue the following bash commands:
    
```
#!shell
    cd BioS/  # make sure to be in the repo direcotry
    nrnivmodl ./mod_files
```
    
* Running a simulation

    The different simulations described in the associated paper can be executed by running the python3 *run_** scripts. Refer to the comments inside each script to see the required arguments that need to be passed at launch time. For example to simulate the effect of BioS (as in Figure 2 of the manuscript) issue the following bash command:


```
#!shell
    cd BioS/  # make sure to be in the repo direcotry
    python3 run_afferent_stimulation.py  --bios --stim-amp -0.027 
```

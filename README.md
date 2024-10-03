# 👀 Problem

**What is it?**
World’s first Early Stage Embryo Physical Simulator. The aim is to provide another layer of embryo quality/growth assessment providing reasonably accurate mechanical simulation that takes into account the superset of observable characteristics.

- 

---

# 💭 Parameters that will be taken into account

The majority of parameters that will be modelled into the simulation will be stochastic (i.e they will sampled from appropriately sampled distributiions.)

They most important are the following

### Potential attributes that might be taken into consideration

- Elasticity
- Viscosity
- Surface tension
- Adhesion Forces between blastomeres
- Complex correlations that affect growth (e.g., larger cells divide more slowly)
- Apoptosis(programmed cell death, allowing cells to shrink(?))
- external influences like nutrient gradients, temperature effects, or mechanical constraints to simulate the microenvironment.

The whole simulation will be designed in order to run optimization experiments with a given target in order to tune the various parameters of the simulation

---

## **Limitations**

While this simulation aims to enhance embryo quality assessment, there are inherent limitations. Accurately modeling the complex biological processes of early-stage embryos is challenging due to limited data and the stochastic nature of cellular behaviours. Computational constraints may also limit the resolution or scale of the simulation. Additionally, oversimplifications necessary for feasibility could reduce the model's accuracy, and ethical considerations regarding the simulation of human embryos must be carefully managed.

---

## Appendix

---

1. **Simulation Engine**
    
    The heart of the simulation engine will be implemented with the Nvidia [warp](https://github.com/NVIDIA/warp) package.
    

---

1. **Sampler**
    
    The sampling engine at first will be implemented with cpu based packages with the ambition to be translated to some CUDA interface in order to run in GPU acceleration alongside the core simulation.
    

---

1. **Tuner**
    
    The final ambition is to implement an optimisation module which will take some targets and tune the various simulation parameters in accordance with the real behaviour.
    
    First iteration will be made with trying to aproximate the 2D surface area as seen from top down view 
    

---

Comprehensive List of Simulation Parameters

Stochastic “Biological” Parameters

1. Cell Division Times
2. Cleavage size distribution
3. Probability of fragmentation
4. Cleavage Plane Orientation
5. 

Mechanical Parameters

1. Elasticity
2. Viscosity
3. Adhesion Forces between blastomeres
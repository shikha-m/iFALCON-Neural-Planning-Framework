# iFALCON Neural Planning Framework

A from-scratch Python implementation of the **iFALCON** (Initial Fuzzy Adaptive Logical Cognitive Neural) architecture. This framework provides a computational model for autonomous cognitive planning, integrating Adaptive Resonance Theory (ART) with multi-channel BDI (Belief-Desire-Intention) logic.

## Technical Overview
This project replicates the complex hierarchical planning structures of the iFALCON architecture. It is designed to handle cognitive mapping and decision-making in autonomous agents through:

* **Multi-Channel Processing:** Segregated layers for Belief, Critic, Desire, and Action.
* **Fuzzy Logic Integration:** Utilizing fuzzy AND/OR operations for robust pattern matching.
* **Dynamic Node Recruitment:** Implementing 'Growth' and 'Learning' phases within the neural network.

## Key Technical Features
* **Modular Class Structure:** Object-oriented implementation of F1 (Input), F2 (Category), and F3 (Plan) layers.
* **Algorithmic Mastery:** Custom-built vigilance testing and weight adjustment kernels.
* **Optimized Search:** Efficient node selection and resonance testing.

## Getting Started

This project implements the iFALCON architecture. The implementation can be verified using the included `iFALCON` test class. 

The test class follows a two-step process:
1. **Feeding Phase:** It first feeds known plans into the system to establish the neural patterns.
2. **Testing Phase:** It then tests the system's response for a specific configuration.

### To Run:
You can execute the framework using either **Jupyter Notebook** or **IDLE**:
* **Jupyter:** Run the cells sequentially to initialize the layers and execute the test class.
* **IDLE:** Open the file and click **Run Module (F5)** to see the output in the shell.

## References
*Based on the original research cited in my project documentation:*

* [1] B. Subagdja and A,-H. Tan, “A Self- Organizing Neural Network Architecture for Intentional Planning Agents”,2009
* [2] B. Subagdja and A,-H. Tan, “A Brain-Inspired model of Hierarchical Planner”, 2011
* [3] B. Subagdja and A,-H. Tan, “Planning with ifalcon: Towards a neural-network-based bdi agent architecture” ,2008


## License
Distributed under the **GNU GPL v3 License**. This ensures that the implementation remains open-source and protects the original effort of the author from closed-source commercial exploitation.

---
**Copyright (c) 2019 Shikha Mittal. All rights reserved.**

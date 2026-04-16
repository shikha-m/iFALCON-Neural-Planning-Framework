# iFALCON Neural Planning Framework

A from-scratch Python implementation of the **iFALCON** (Initial Fuzzy Adaptive Logical Cognitive Neural) architecture.This framework provides a computational model for autonomous cognitive planning, integrating Adaptive Resonance Theory (ART) with multi-channel BDI (Belief-Desire-Intention) logic.

## Technical Overview
This project replicates the hierarchical planning structures of the iFALCON architecture, designed to handle cognitive mapping and decision-making in autonomous agents. Unlike traditional symbolic planners, this neural planner is capable of automatically seeking and acquiring plans "on the fly" through experience.

Key architectural components include:
* **Multi-Channel Processing:** Segregated input/output fields for Beliefs, Desires, Critic, and Action.
* **Fuzzy Logic Integration:** Utilization of fuzzy AND operations for robust pattern matching and resonance search.
* **Dynamic Node Recruitment:** Implementation of self-organizing layers that allocate new category neurons when no existing patterns match the input.

## Experimental Validation: Blocks World
The framework was tested using the **Blocks World domain**, a classic AI problem requiring sequential and hierarchical action structures.
* **Learning Phase:** The model was trained on primitive plans to establish initial weighted connections.
* **Execution:** The system demonstrated the ability to traverse search spaces and discover legal intermediate steps to reach a goal configuration.
* **Key Finding:** The research observed that planning quality is sensitive to pre-existing knowledge and initial task configurations.

## Getting Started
This implementation can be verified using the included `iFALCON` test class.

### To Run:
You can execute the framework using **Jupyter Notebook** or **IDLE**:
1. **Feeding Phase:** Known plans are fed to the system to establish neural patterns.
2. **Testing Phase:** The system is tested for a specific block configuration.
*Open the main module and click **Run Module (F5)** to view the execution results.*

## Research Context
This implementation was developed as a **Credit-Bearing Independent Study** during my Master’s in Computer Science at **Shippensburg University**. 

While the research and codebase are entirely my own, the project was conducted under the guidance of **Dr. David J. Mooney**, who provided the initial direction and valuable feedback throughout the study.

*The full technical Research Report is available upon request.*

## References
*Based on the original research cited in the project report:*
* [1] Andrew Sohn, Jean-Luc-Gaudiot, "A Connectionist Approach to Learning Moves in Tower of Hanoi", 1990 
* [2] B. Subagdja and A,-H.Tan, "A Self-Organizing Neural Network Architecture for Intentional Planning Agents", 2009 
* [3] B. Subagdja and A,-H.Tan, "A Brain-Inspired model of Hierarchical Planner", 2011 
* [4] B. Subagdja and A, H. Tan, "Planning with ifalcon: Towards a neural-network-based bdi agent architecture", 2008
  
## License
Distributed under the **GNU GPL v3 License**. This ensures that the implementation remains open-source and protects the original effort of the author from closed-source commercial exploitation.

---
**Copyright (c) 2019 Shikha Mittal. All rights reserved.**

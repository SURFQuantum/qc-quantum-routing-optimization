# Qubit Routing with Reinforcement Learning (RL)

This repository is part of a Bachelor internship thesis on Quantum Circuit Routing Optimization for the company SURF. It contains code for a qubit routing procedure that makes use of Monte Carlo Tree Search (MCTS) guided by a Reinforcement Learning (RL) model. CLick [here](https://github.com/Lizaterdag/quantum-routing-optimization/blob/main/Bachelor_Thesis_Liza_Darwesh_500793232_Amsterdam_University_of_Applied_Science.pdf) for thesis. This repository also contains notes and presentations from the meetings at SURF.

## Intro to Qubit Routing

In order to run a quantum algorithm on a quantum computer, one has to make sure the algorithm is executable on a quantum computer. A quantum algorithm is represented in the form of a quantum circuit, consisting of quantum logic gates. These gates operations on the physical quantum bits inside the quantum computer. There are several things that need to be taking into account to be able to run these quantum circuits on a quantum computer (i.e topology, qubit lifetime, crosstalk etc). This research illuminates an approach on modifying a quantum circuit to satisfy the contrains of the target topology. 

A quantum architecture has a connectivity graph, consisting of physical qubits ("nodes") and links between them. (Logical) qubits inhabit the nodes, and two-qubit gates may only occur between qubits on adjacent nodes in the topology. SWAP-gates must be inserted to move the qubits and satisfy such constraints - this process is known as "routing".

## The Algorithm

![alt text](https://github.com/Lizaterdag/quantum-routing-optimization/blob/main/img/workflow.png)

## Python package versions

This code requires **Python 3.9**, as well as specific versions of a few libraries in order to run. I believe there have been some minor changes to those libraries in recent times, and I have not yet had the chance to update the code in response. I recommend installing the relevant packages by running `pip install -r requirements.txt`, or you could even try fixing the code yourself to be compatible with the latest versions.

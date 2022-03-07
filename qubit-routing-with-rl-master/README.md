# Qubit Routing with Reinforcement Learning (RL)

This repository contains code for a qubit routing procedure that makes use of RL. Originally developed by Matteo G. Pozzi as part of a Master's Thesis for the University of Cambridge Computer Laboratory. A paper on the subject is available here: https://arxiv.org/abs/2007.15957.

## Intro to Qubit Routing

Before quantum circuits can be executed on quantum architectures, they must be be modified to satisfy the contrains of the target topology. Specifically, a quantum architecture has a connectivity graph, consisting of physical qubits ("nodes") and links between them. (Logical) qubits inhabit the nodes, and two-qubit gates may only occur between qubits on adjacent nodes in the topology. SWAP gates must be inserted to move the qubits and satisfy such constraints - this process is known as "routing".

This project uses RL to perform the task of routing qubits. 

![alt text](https://github.com/Lizaterdag/quantum-routing-optimisation/blob/main/img/RL.png)

The agent views the state: which represents the initial location of the qubit, its interaction with another logical qubit, and if it is already possible to carry out a CNOT gate immediatly.

![alt text](https://github.com/Lizaterdag/quantum-routing-optimisation/blob/main/img/topology.png)

![alt text](https://github.com/Lizaterdag/quantum-routing-optimisation/blob/main/img/circuit.png)

If you consider the topology and circuit as an example, then you can see that some of these interactions are not possible (CNOT0,2 and CNOT1,3). These qubits need to be moved. The SWAP operation is used for this.


![alt text](https://github.com/Lizaterdag/quantum-routing-optimisation/blob/main/img/swapcircuit.png)


The agent has to select some swap gates of the remaining qubits that are not involved in the cnot gates. Thats where the simulated annealing comes in. By training a model, we try to carry out the routing. The state representation goes through a function that computes a feature representation, which condenses the state in a fixed length vector of distances (how many qubits do I have to go through). The first entry of the vector is the number of qubits that are one hop away from their target, where the second entry is the number of qubits that are two hops away. 
  We feed the vector through the neural network, which outputs a single continues number which is the Q value (quality of the current state) in the context of Q-Learning, based on the quality of state and next state.
  
![alt text](https://github.com/Lizaterdag/quantum-routing-optimisation/blob/main/img/annealing.png)  

The simulated annealer selects a bunch of swaps to add the action in sequences on possible swap gate
and evaluates the quality. So basically, what is depicted in the diagram above. The annealing 
encourages the exploration of the state space and decreases the temperature so you end up exploring less and less till 
you end up only adding a swap gate when it is benefitting the quality. 

So in short:
The neural network learns a quality function and then the simulated annealing constructs the action that is best to carry out in that step 
by invoking the neural network many times to try and optimize. Qualities come out with the highest 
possible q for a given state. Once we select an action, we carry out and perform the swap and we get a new state and do it all again. 

## Module structure

- The `environments` directory contains classes that represent different types of quantum architecture. These are called "environments", in the RL sense of the word - specifically, they are responsible for generating a new state from a state-action pair, and delivering a reward. The simplest is the "grid" environment, but there are some unique real-world quantum architectures as well, such as the IBM Q20 Tokyo. There is also a `PhysicalEnvironment` class that is responsible for simulating a (_routed_) quantum circuit on a given target architecture, for the purpose of verifying that the hardware constraints are indeed satisfied.
- The `benchmarks` directory contains a series of benchmarks used for the thesis and subsequent paper. In general, to obtain results for different architectures or routing methods, simply import the correct environment and `schedule_swaps` function for the architecture and method you would like to test, respectively.
- The `agents` and `annealers` directories contain the code for the actual RL method, as well as some helper functions for training the models and performing routing.
- The `other_systems` directory contains code for routing with other existing methods, such as Qiskit's `StochasticSwap`. These files simply wrap external library calls into a common format, for easy benchmarking.
- The `realistic_test_set` directory contains a series of `.qasm` files that were used in the paper to benchmark the different routing methods.
- The `utils` directory contains a series of utilities that were useful throughout the thesis, as well as some static heuristics (which were used in the initial phase of the thesis) and a simple implementation of a PER memory tree.

Please note: the "single state" agent was briefly trialled in the thesis but was not used in the paper. The "paired state" agent remains the recommended choice, and is the one evaluated in the paper.


## Python package versions

This code requires **Python 3.7**, as well as specific versions of a few libraries in order to run. I believe there have been some minor changes to those libraries in recent times, and I have not yet had the chance to update the code in response. I recommend installing the relevant packages by running `pip install -r requirements.txt`, or you could even try fixing the code yourself to be compatible with the latest versions.


## Disclaimer

This code has not yet been properly documented. It is the result of cleaning up the original project repository (which was significantly larger and more messy), so apologies in advance if some files have ended up in the wrong places. Please do get in touch if you have any issues running or understanding the code, and I will try my best to help out.

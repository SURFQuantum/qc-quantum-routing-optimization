OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
// This initializes 6 quantum registers and 6 classical registers.

h q[0];
h q[1];
h q[2];
// The first 3 qubits are put into superposition states.

cx q[2], q[4];
x q[3];
cx q[2], q[3];
ccx q[0], q[1], q[3];
x q[0];
x q[1];
ccx q[0], q[1], q[3];
x q[0];
x q[1];
x q[3];
// This applies the secret structure: s=110.

// This measures the second 3 qubits.
// h q[0];
// h q[1];
// h q[2];

// This measures the first 3 qubits.

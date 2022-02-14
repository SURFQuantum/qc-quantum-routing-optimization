OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

h q[0];
ccx q[0],q[2],q[1];
ccx q[1],q[2],q[0];
x q[0];
ccx q[0],q[1],q[2];
x q[0];
x q[2];
ccx q[1],q[2],q[0];
ccx q[0],q[2],q[1];
x q[1];
ccx q[1],q[2],q[0];
x q[1];
x q[2];
ccx q[1],q[2],q[0];
x q[0];
ccx q[0],q[1],q[2];
x q[0];
x q[2];
ccx q[1],q[2],q[0];
x q[2];
ccx q[0],q[2],q[1];
ccx q[1],q[2],q[0]; 
x q[0];
ccx q[0],q[1],q[2];
x q[0];
x q[1];
ccx q[1],q[2],q[0]; 
x q[1];
ccx q[0],q[2],q[1];
ccx q[1],q[2],q[0]; 


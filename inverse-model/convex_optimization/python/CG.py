# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:20:56 2019

@author: shihao

Solve Ax = b using Conjugate Gradient method
"""
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse.linalg

A = np.matrix(hlr)
At = A.getH()
R = np.reshape(R, (22, 1))
b = np.matrix(R)

AtA = A_T * A
Atb = A_T * b

x_r = np.real(B0)
x_i = np.imag(B0)

x_comb = np.concatenate((x_r, x_i), axis = 0)

B = np.real(A)
C = np.imag(A)

u = np.real(b)
v = np.imag(b)

A_up = np.concatenate((B, -C), axis = 1)
A_bt = np.concatenate((C, B), axis = 1)
A_comb = np.concatenate((A_up, A_bt), axis = 0)

b_comb = np.concatenate((u, v), axis = 0)

AtAc = np.matrix.transpose(A_comb) * A_comb
Atbc = np.matrix.transpose(A_comb) * b_comb

P, L, U = sp.linalg.lu(AtA)

L_s = np.linalg.inv(L) * Atb
x = np.linalg.inv(U) * L_s

eigenvalues_A = np.linalg.eigvals(A)
eigenvalues_A_T_A = np.linalg.eigvals(A_T_A)

preconditioner = np.linalg.inv(A_T_A)



def is_symmetric(a, tol=1e-5):
    return np.allclose(a, a.getH(), atol=tol)

def is_pos_def(a):
    return np.all(np.linalg.eigvals(0.5*(a + a.getH())) > 0)

print(is_symmetric(AtAc))
print(is_pos_def(AtAc))

x, info = sp.sparse.linalg.cg(AtAc, Atbc, tol = 1e-11)

plt.figure()
plt.subplot(211)
plt.imshow(np.real(A))
plt.title('Real')
plt.colorbar()
plt.subplot(212)
plt.imshow(np.imag(A))
plt.title('Imag')
plt.colorbar()
plt.suptitle('A Matrix')

plt.figure()
plt.subplot(211)
plt.imshow(np.real(A_T))
plt.title('Real')
plt.colorbar()
plt.subplot(212)
plt.imshow(np.imag(A_T))
plt.title('Imag')
plt.colorbar()
plt.suptitle('A Transpose')

plt.figure()
plt.subplot(211)
plt.imshow(np.real(A_T_A))
plt.title('Real')
plt.colorbar()
plt.subplot(212)
plt.imshow(np.imag(A_T_A))
plt.title('Imag')
plt.colorbar()
plt.suptitle('A Transpose * A')

plt.figure()
plt.subplot(211)
plt.plot(np.real(np.squeeze(B0)), label = 'Ground Truth')
plt.plot(np.real(x[:22]), label = 'Solution')
plt.title('Real')
plt.legend()

plt.subplot(212)
plt.plot(np.imag(np.squeeze(B0)), label = 'Ground Truth')
plt.plot((x[22:]), label = 'Solution')
plt.title('Imag')
plt.legend()

plt.figure()
plt.scatter(np.real(eigenvalues_A_T_A), np.imag(eigenvalues_A_T_A), s = 3)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('EigenValues')

plt.figure()
plt.scatter(np.real(eigenvalues_A), np.imag(eigenvalues_A), s = 3)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('EigenValues')
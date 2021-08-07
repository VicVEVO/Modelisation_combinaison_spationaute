
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 22:52:30 2021

@author: victo
"""

import numpy as np
from numpy.linalg import inv

import seaborn as sns; sns.set_theme()

precision = 70

nlignes , ncolonnes = precision, precision

xmax , tmax = 10 , 10

alpha = 10**-5 # k/rho*c_p #coeff de diffusivité


h_x , h_t = xmax/nlignes , tmax/ncolonnes
r = alpha*h_t/(h_x**2)


def f(x): #la fonction définissant la température U à t = 0
    # return(np.exp(-0.1*x)*100)
    # return(np.sin(x)*100 + 10)
    # return(x**2-3*x+3)
    return(abs(100*np.sin(x/2) + 20))
    # return(x)

def matrice_AB(r, nlignes, ncolonnes): #calcul des deux matrices A et B
    nlignes,ncolonnes = nlignes -2, ncolonnes-2
    A = np.zeros((nlignes,ncolonnes))
    B = np.zeros((nlignes,ncolonnes))
    for i in range(nlignes):
        for j in range(ncolonnes):
            if i == j:
                A[i][j] = 2 + 2 * r
                B[i][j] = 2 - 2 * r
            elif (j == i + 1) or (i == j+1):
                A[i][j] = -r
                B[i][j] = r
    return inv(A), B



def init(f,nlignes,ncolonnes): #initialise la matrice U
    M = np.zeros((nlignes,ncolonnes))
    for i in range(nlignes):       
        M[i][0] = f(h_x*i)
    for i in range(ncolonnes):
        M[0][i] = T0
        M[nlignes-1] = f((nlignes-1)*h_x)
    
    return(M)

def matrice_U(f, nlignes, ncolonnes, r, invA, B): #calcule la matrice U (lignes (resp. colonens) correspondent aux x (resp. t))
    U = init(f,nlignes+2,ncolonnes+2)
    #U = np.around(U,2)
    # print('------------- Matrice initiale: -------------')
    # print(U, '\n') #ctrl+1
    # print(U)
    invAxB = np.dot(invA,B)    #print(U[:,0])
    for t in range(1,ncolonnes+2):
        b = np.zeros(nlignes)
        #print('B_{} initial = {}'.format(t,b_t))
        b[0] = r*U[0,t-1]
        b[nlignes-1] = r*U[nlignes+1,t-1]
        print(b)
# =============================================================================
#         #print(r*U[nlignes+1,t]) 
#         #print('B_{} final = {}'.format(t,b_t))
#         
#         #print(invAxB)
#         #print((U[1:nlignes+1,t-1]))
#         print(np.dot(invAxB,U[1:nlignes+1,t-1]))
#         print(np.dot(invA,b_t))
#         #print(np.dot(U[1:nlignes-1,t+1],invAxB))
#         
#         print(U[1:nlignes+1,t]) #= invAxB.dot(U[1:nlignes+1,t-1]) + invA.dot(b_t)
#         print('')
#         
# =============================================================================

        U[1:nlignes+1,t] = np.dot(invAxB,U[1:nlignes+1,t-1]) + np.dot(invA,b)

    return(U)

T0 = f(0)
invA,B = matrice_AB(r,nlignes+2,ncolonnes+2)



U = matrice_U(f,nlignes, ncolonnes, r, invA, B)


if precision <= 10: #On affiche correctement les label car sinon il y a trop: illisible & trop de temps d'affichage
    ax = sns.heatmap(U,
                     xticklabels = np.around(np.arange(0, tmax + 2*h_t, h_t),1),
                     yticklabels = np.around(np.arange(0, xmax + 2*h_x, h_x),1),
                     cbar_kws={'label': 'Temperature'},
                     center=0)
else :
    ax = sns.heatmap(U,
                     cbar_kws={'label': 'Temperature'},
                     center=0)
    
ax.set(title="TEMPERATURE 1D",
      xlabel="Durée (s)",
      ylabel="Distance (m)")

"""
print("r = {}".format(r))
print(np.round(inv(invA),1))
print(np.round(B,1))
print(np.around(U,2))"""
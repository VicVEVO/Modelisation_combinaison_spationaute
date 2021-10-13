# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 10:25:22 2021

@author: victo
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


precision = 100  # paramètre que l'on définit pour l'intervalle de mesure

nlignes, ncolonnes = precision, precision

xmax, tmax = 10, 300000  # Valeur de mesure spatiale (resp. temporelle) max

'''
λ = #la conductivité thermique du matériau
ρ = #sa masse volumique
c = #sa capacité thermique massique à pression constante
'''
    
#alpha = λ/(ρ*c) #coefficient de diffusivité

alpha = 10 ** -5 # en m^2/s

h_x, h_t = xmax / nlignes, tmax / ncolonnes  # "correspondent" à dx , dt
r = alpha * h_t / (h_x ** 2)  # constante utilisée dans le calcul des matrices A et B pour la récurrence

#------------------------------------------------------------------#
#On définit les fonctions définissant la température sur chaque barre (en x,y puis z) à t = 0s

def f_x(x):
    return abs(100 * np.sin(x/2) + 20)

def f_y(x):
    return abs(x+3)

def f_z(x):
    return abs(-x**2+3)
#------------------------------------------------------------------#

def matrice_AB(r, nlignes, ncolonnes):  # calcul des deux matrices constantes A et B pour le calcul de récurrence du modèle
    #On ne les calcule qu'une fois pour réduire considérablement les calculs répétitifs inutiles
    nlignes, ncolonnes = nlignes - 2, ncolonnes - 2 #il faut qu'on fixe ça

    A = np.zeros((nlignes, ncolonnes)) #A et B sont des matrices initialement nulles
    B = np.zeros((nlignes, ncolonnes)) #que en fonction de nlignes, ncolonnes utilisé à la fin

    for i in range(nlignes):
        for j in range(ncolonnes):
            if i == j:
                A[i][j] = 2 + 2 * r
                B[i][j] = 2 - 2 * r
            elif (j == i + 1) or (i == j + 1):
                A[i][j] = -r
                B[i][j] = r
    return inv(A), B #on renvoie directement l'inverse de A, car on n'utilise que son inverse


def init(f, nlignes, ncolonnes):  # initialisation de la matrice U
    M = np.zeros((nlignes, ncolonnes))
    for i in range(nlignes):  # température sur la barre à t = 0
        M[i][0] = f(h_x * i)
    for i in range(ncolonnes):  # température constante aux bords de la barre
        M[0][i] = f(0)
        M[nlignes - 1] = f((nlignes - 1) * h_x)
    return M


def matrice_U(f, nlignes, ncolonnes, r, invA, B):  # calcule la matrice U (lignes (resp. colonnes) correspondent aux x (resp. t)) par récurrence

    U = init(f, nlignes + 2, ncolonnes + 2)  # initialise U
    invAxB = np.dot(invA, B)  # calcule une fois A^-1*B pour éviter des calculs redondants

    for t in range(1, ncolonnes + 2):
        b = np.zeros(nlignes)
        bjplus1 = np.zeros(nlignes)
        b[0] = r * U[
            0, t - 1]  # calcul de la matrice colonne b, quasi nulle mais nécessaire dans le calcul par récurrence
        b[nlignes - 1] = r * U[nlignes + 1, t - 1]
        bjplus1 [nlignes - 1] = r * U[nlignes + 1, t]
        U[1:nlignes + 1, t] = np.dot(invAxB, U[1:nlignes + 1, t - 1]) + np.dot(invA, b) + np.dot(invA, bjplus1) # application de la formule de récurrence

    return U


"""   ### DÉBUT DU PROGRAMME ###   """

invA, B = matrice_AB(r, nlignes + 2, ncolonnes + 2) #on calcule A^-1 et B

U_x = matrice_U(f_x, nlignes, ncolonnes, r, invA, B)
U_y = matrice_U(f_y, nlignes, ncolonnes, r, invA, B)
U_z = matrice_U(f_z, nlignes, ncolonnes, r, invA, B)

# plt.xlabel("Durée (s)")
# plt.ylabel("Distance (m)")
# plt.title('TEMPERATURE 1D')

# plt.imshow(U_z,extent = [0,tmax,0,xmax], aspect = 'auto',cmap = 'afmhot')
# #plt.imshow(A,Umin,Umax,extent..)
# cb = plt.colorbar()
# cb.set_label("Température (°c)") 

A = U_x[:,0],U_y[:,0],U_z[:,0]
#print(A[0])
U = np.empty((nlignes+2,3))
U[:,0] = U_x[:,0]
U[:,1] = U_y[:,0]
U[:,2] = U_z[:,0]

ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(U[0], U[1], U[2], antialiased=True)
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:33:37 2021

@author: victo
"""

import matplotlib.pyplot as plt
import numpy as np

precision = 100  # paramètre que l'on définit pour l'intervalle de mesure

nlignes, ncolonnes = precision, precision

xmax, tmax = 10, 300000  # Valeur de mesure spatiale (resp. temporelle) max
alpha = 10 ** -5  # k/rho*c_p #coefficient de diffusivité

h_x, h_t = xmax / nlignes, tmax / ncolonnes  # "correspondent" à dx , dt
r = alpha * h_t / (h_x ** 2)  # constante utilisée dans le calcul des matrices A et B pour la récurrence


def f(x):  # la fonction définissant U[0], la température sur la barre à t = 0
    return abs(100 * np.sin(x / 2) + 20)


def matrice_AB(r, nlignes, ncolonnes):  # calcul des deux matrices constantes A et B pour le calcul de récurrence du modèle
    # On ne les calcule qu'une fois pour réduire considérablement les calculs répétitifs inutiles
    nlignes, ncolonnes = nlignes - 2, ncolonnes - 2  # il faut qu'on fixe ça

    A = np.zeros((nlignes, ncolonnes))  # A et B sont des matrices initialement nulles
    B = np.zeros((nlignes, ncolonnes))  # que en fonction de nlignes, ncolonnes utilisé à la fin

    for i in range(nlignes):
        for j in range(ncolonnes):
            if i == j:
                A[i, j] = 2 + 2 * r
                B[i, j] = 2 - 2 * r
            elif (j == i + 1) or (i == j + 1):
                A[i, j] = -r
                B[i, j] = r
    return A, B


def init(f, nlignes, ncolonnes):  # initialisation de la matrice U
    M = np.zeros((nlignes, ncolonnes))
    for i in range(nlignes):  # température sur la barre à t = 0
        M[i][0] = f(h_x * i)
    for i in range(ncolonnes):  # température constante aux bords de la barre
        M[0][i] = T0
        M[nlignes - 1] = f((nlignes - 1) * h_x)
    return M


def inverse_tridiag(M, D):  # Procédure qui inverse une matrice tridiag, D étant le vecteur contenant les solutions
    n = len(M)
    diag_du_bas = np.zeros(n - 1)
    diag_du_haut = np.zeros(n - 1)
    diag_du_mid = np.zeros(n)
    diag_du_mid[0] = M[0, 0]
    solution = np.empty(n)
    for i in range(1, n - 1):  # On définit les diagonales de la matrice tridiag
        diag_du_bas[i] = M[i, i - 1]
        diag_du_mid[i] = M[i, i]
        diag_du_haut[i] = M[i - 1, i]
    diag_du_haut[0] = diag_du_haut[0] / diag_du_mid[0]
    D[0] = D[0] / diag_du_mid[1]
    for j in range(1, n - 1):  # Relation de réccurence de l'algorithme de Thomas
        diag_du_haut[j] = diag_du_haut[j] / (diag_du_mid[j] - diag_du_bas[j] * diag_du_haut[j - 1])
        D[j] = (D[j] - diag_du_bas[j] * D[j - 1]) / (diag_du_mid[j] - diag_du_bas[j] * diag_du_haut[j - 1])
    solution[n - 1] = D[n - 1]
    for k in range(n - 2, -1, -1):  # Substitution inverse pour trouver le vecteur solution
        solution[k] = D[k] - diag_du_haut[k] * solution[k + 1]
    return solution


def matrice_U(f, nlignes, ncolonnes, r, A):  # calcule la matrice U (lignes (resp. colonnes) correspondent aux x (resp. t)) par récurrence

    U = init(f, nlignes + 2, ncolonnes + 2)  # initialise U
    r_mat = np.ones     (nlignes)
    for t in range(1, ncolonnes + 2):
        b = np.zeros(nlignes)
        bjplus1 = np.zeros(nlignes)
        b[0] = r * U[
            0, t - 1]  # calcul de la matrice colonne b, quasi nulle mais nécessaire dans le calcul par récurrence
        b[nlignes - 1] = r * U[nlignes + 1, t - 1]
        bjplus1[nlignes - 1] = r * U[nlignes + 1, t]
        for i in range(nlignes - 1):
            r_mat[i] = ((2 - 2 * r) * U[3 + i, t - 1] + r * b[i] + bjplus1[i])
        U[1:nlignes + 1, t] = inverse_tridiag(A, r_mat)  # application de la formule de récurrence

    return U


"""   ### DÉBUT DU PROGRAMME ###   """

T0 = f(0)

A, B = matrice_AB(r, nlignes + 2, ncolonnes + 2)
U = matrice_U(f, nlignes, ncolonnes, r, A)
print(U)
plt.xlabel("Durée (s)")
plt.ylabel("Distance (m)")

plt.title('TEMPERATURE 1D')
plt.imshow(U, extent=[0, tmax, 0, xmax], aspect='auto', cmap='afmhot')
cb = plt.colorbar()
cb.set_label("Température (°c)")

plt.show()

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:33:37 2021

@author: victo
"""

from numpy.linalg import inv
import matplotlib.pyplot as plt
import time
from variables_2D import *


########################### D E F I N I T I O N S #######################

def f(x):  # la fonction définissant U[0], la température sur la barre à t = 0
    return abs(100 * np.sin(x / 2) + 20)


def g(i, j):
    return abs(100 * np.sin((i + j) / 200) + 20)


def inverse_tridiag(M, D):  # Procédure qui inverse une matrice tridiag M, D étant le vecteur à droite du système MX = D
    n = len(M)
    diag_du_bas = np.zeros(n - 1)
    diag_du_haut = np.zeros(n - 1)
    diag_du_mid = np.zeros(n)
    diag_du_mid[0] = M[0, 0]
    solution = np.empty(n)
    for i in range(1, n):  # On définit les diagonales de la matrice tridiag
        diag_du_bas[i - 1] = M[i, i - 1]
        diag_du_mid[i] = M[i, i]
        diag_du_haut[i - 1] = M[i - 1, i]
    diag_du_haut[0] = diag_du_haut[0] / diag_du_mid[0]
    D[0] = D[0] / diag_du_mid[0]
    for j in range(1, n - 1):  # Relation de recurrence de l'algorithme de Thomas
        diag_du_haut[j] = diag_du_haut[j] / (diag_du_mid[j] - diag_du_bas[j - 1] * diag_du_haut[j - 1])
        D[j] = (D[j] - diag_du_bas[j - 1] * D[j - 1]) / (diag_du_mid[j] - diag_du_bas[j - 1] * diag_du_haut[j - 1])
    D[n - 1] = (D[n - 1] - diag_du_bas[n - 2] * D[n - 2]) / (diag_du_mid[n - 1] - diag_du_bas[n - 2] * diag_du_haut[n - 2])
    solution[n - 1] = D[n - 1]
    for k in range(n - 2, -1, -1):  # Substitution inverse pour trouver le vecteur solution
        solution[k] = D[k] - diag_du_haut[k] * solution[k + 1]
    return solution


def init(T_int, T_ext, taille_mat):  # initialisation de la matrice U

    M = np.full((taille_mat, taille_mat), T_int, dtype=float)  # On impose T_int partout
    M[0], M[-1], M[:, 0], M[:, -1] = T_ext, T_ext, T_ext, T_ext  # puis on impose T_ext à l'extérieur de la boîte, donc aux extrémités de la matrice

    return M


def calcul_U_t_suivant(U, T_int, taille_mat, E, A, B, λ_int, r):  # calcule la matrice de la température U au temps suivant

    flux_tot = 0
    invAs = [[invA_iso_0, invA_iso_0], [invA_iso_1, '...', invA_iso_last]
    invAxBs = []
    #for k in range(len(E)):

    for y in range(1, taille_mat - 1):  # balayage de t à t+1/2 : on étudie les lignes
        for ind_ep in range(1,len(E)): # si on est dans l'une des couches isolantes (pas le premier contours mais celles à l'intérieur)
            if E[ind_ep-1]+1 <= y <= E[ind_ep] or taille_mat - np.sum(E[:ind_ep+1]) <= y <= taille_mat - np.sum(E[:ind_ep]):  # on étudie le cas où on est au dans l'épaisseur ind_ep

                U[y, :E[0]] = calc_U(U[y, :E[0] + 1], invAs[ind_ep][1], invAxBs[ind_ep], r[ind_ep])  # Utype0, on est dans l'iso du 0
                for k in range(1,ind_ep): #montée pyramide
                    U[y, np.sum(E[:k])+1:np.sum(E[:k+1])+1] = calc_U(U[y, np.sum(E[:k])+1:np.sum(E[:k+1])+1], invAs[ind_ep][1], invAxBs[ind_ep], r[ind_ep])  # Utype1, on est dans l'iso du 1

                U[y, :E[ind_ep]] = calc_U(U[y, :E[ind_ep]], invA_mid, invAxB_mid, r)
                U[y, taille_mat - E:] = calc_U(U[y, taille_mat - E:], invA_mid, invAxB_mid, r)

                flux_tot += profondeur_boite * λ_int * (U[y, taille_mat - E + 1] - T_int)  # Flux à droite
                flux_tot += profondeur_boite * λ_int * (U[y, E - 2] - T_int)  # Flux à gauche

            elif y<=E[0] or y>=taille_mat-E[0]:  # cas où on est dans le premier isolant (premier contours)
                U[y] = calc_U(U[y], invAs[0], invAxBs[0], r[0])

    for x in range(1, taille_mat - 1):  # balayage de t+1/2 à t+1 : on étudie les colonnes

        if E - 1 <= x <= taille_mat - E:  # cas où on est au middle

            U[:E, x] = calc_U(U[:E, x], invA_mid, invAxB_mid, r)
            U[taille_mat - E:, x] = calc_U(U[taille_mat - E:, x], invA_mid, invAxB_mid, r)

            flux_tot += profondeur_boite * λ_int * (U[E - 2, x] - T_int)  # Flux en haut
            flux_tot += profondeur_boite * λ_int * (U[taille_mat - E + 1, x] - T_int)  # Flux en bas

        else:  # cas où on est dans l'isolant

            U[:, x] = calc_U(U[:, x], invA_iso, invAxB_iso, r)

    T_int += pas_temporel * flux_tot / c_int
    U[E - 1:taille_mat - E + 1, E - 1:taille_mat - E + 1] = T_int
    return T_int


def calc_U(barre, invA, invAxB, r):  # T0 (resp.T1): température extérieure gauche (resp. droite)
    longueur = len(barre[1:-1])  # longueur de la barre où l'on change la température

    ### CALCUL DE b ###
    b = np.zeros(longueur)  # ne contiens pas les extrémités : 2 cases en moins

    b[0] = 2 * r * barre[0]
    b[-1] = 2 * r * barre[-1]  # on a mis 2r au lieu d'ajouter bjplus1 [longueur - 1] = r * U[longueur + 1, t]

    barre[1:-1] = np.dot(invAxB, barre[1:-1]) + np.dot(invA, b)  # application de la formule de récurrence
    return barre


########################### D E B U T  D U  P R O G R A M M E #######################
t = time.time()
U = init(T_int, T_ext, taille_mat)

t_0 = time.time()

for i in range(nb_diterations):  # on calcule U avec n itérations

    # T_int = calcul_U_t_suivant(U, T_int, taille_mat, E, A, B, λ, r)
    if i % (nb_diterations / 100) == 0 or i == nb_diterations - 1:  # Afin de faire un affichage plus fluide
        print("Temps restant estimé : {}s".format(int((time.time() - t_0) * (nb_diterations - i) / (i + 1))))
        print("▓" * int(29 * ((i + 1) / nb_diterations)) + "░" * (29 - int(29 * ((i + 1) / nb_diterations))))
        print(" " * 13 + str(int(100 * (i + 1) / nb_diterations)) + "%" + "" * 13, 4 * '\n\n')

plt.xlabel("Distance (en m)")
plt.ylabel("Distance (en m)")
plt.title('TEMPERATURE 2D')

image = plt.imshow(U, extent=[0, longueur_boite, 0, longueur_boite], cmap='afmhot', aspect='auto', animated=True, vmin=T_int_init, vmax=T_ext)

cb = plt.colorbar()
cb.set_label("Température (en °C)")

print("Temps d'exécution = {} secondes".format(int(time.time() - t)))

# plt.plot([epaisseur, longueur_boite - epaisseur, longueur_boite - epaisseur, epaisseur, epaisseur],
#          [epaisseur, epaisseur, longueur_boite - epaisseur, longueur_boite - epaisseur, epaisseur], color='#006a4e')
plt.show()

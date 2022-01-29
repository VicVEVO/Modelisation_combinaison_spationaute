# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:33:37 2021

@author: victo
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import time


########################### D E F I N I T I O N S #######################

def f(x):  # la fonction définissant U[0], la température sur la barre à t = 0
    return abs(100 * np.sin(x / 2) + 20)


def matrice_AB(r, taille_mat):
    """ Calcule les deux matrices carrées A et B nécessaires pour le calcul de récurrence du modèle à partir du pattern.
     On ne les calcule qu'une fois afin de réduire le temps d'éxecution du programme et de limiter les calculs redondants"""
    # On ne les calcule qu'une fois pour réduire considérablement les calculs répétitifs inutiles
    A = np.zeros((taille_mat - 2, taille_mat - 2))  # A et B sont des matrices initialement nulles
    B = np.zeros((taille_mat - 2, taille_mat - 2))  # qu'en fonction de nlignes, ncolonnes utilisé à la fin
    # il faut fix le -2
    for i in range(taille_mat - 2):
        for j in range(taille_mat - 2):
            if i == j:
                A[i][j] = 2 + 2 * r
                B[i][j] = 2 - 2 * r
            elif (j == i + 1) or (i == j + 1):
                A[i][j] = -r
                B[i][j] = r
    return A, B


def init(T_int, T_ext, taille_mat, R):
    """Initialise la matrice U en fonction des conditions aux initiales"""
    mat_check = np.full((taille_mat, taille_mat), False, dtype=int)
    M = np.full((taille_mat, taille_mat), T_int, dtype=float)  # On impose T_int partout
    M[0], M[-1], M[:, 0], M[:, -1] = (T_ext for _ in range(4))  # puis on impose T_ext à l'extérieur de la boîte, donc aux extrémités de la matrice
    mat_check[0], mat_check[-1], mat_check[:, 0], mat_check[:, -1] = (1 for _ in range(4))
    centre = taille_mat // 2, taille_mat // 2
    for i in range(taille_mat):
        for j in range(taille_mat):
            if ((i - centre[0]) ** 2 + (j - centre[1]) ** 2) >= R ** 2:
                M[i, j] = T_ext
                mat_check[i, j] = 1
    return M, mat_check


def calcul_U_t_suivant(U, T_int, taille_mat, E, A, B, λ, profondeur, r):
    """Calcul la matrice Température U au temps suivant en fonction des cas limites, des conditions initiales et de l'épaisseur de la boîte"""
    """for x in range(1, taille_mat - 1):  # Balayage de t à t+1/2
        for y in range(1, taille_mat - 1):
            invA = inv(A[:E - 2, :E - 2])
            invAxB = np.dot(invA, B[:E - 2, :E - 2])

            U[y, x] = calc_U(U[y, x], invA, invAxB, r)  # AUCUN SENS PTN
            """

    for y in range(1, taille_mat - 1):  # balayage de t à t+1/2 : on étudie les lignes

        if E - 1 <= y <= taille_mat - E:  # on étudie le cas où on est au middle

            invA = inv(A[:E - 2, :E - 2])
            invAxB = np.dot(invA, B[:E - 2, :E - 2])

            U[y, :E] = calc_U(U[y, :E], invA, invAxB, r)
            U[y, taille_mat - E:] = calc_U(U[y, taille_mat - E:], invA, invAxB, r)

        else:
            invA = inv(A)
            invAxB = np.dot(invA, B)

            U[y] = calc_U(U[y], invA, invAxB, r)  # calc_U(matrice 1D, CL1, CL2)

    for x in range(1, taille_mat - 1):  # balayage de t+1/2 à t+1 : on étudie les colonnes

        if E - 1 <= x <= taille_mat - E:
            invA = inv(A[:E - 2, :E - 2])
            invAxB = np.dot(invA, B[:E - 2, :E - 2])

            U[:E, x] = calc_U(U[:E, x], invA, invAxB, r)
            U[taille_mat - E:, x] = calc_U(U[taille_mat - E:, x], invA, invAxB, r)

        else:
            invA = inv(A)
            invAxB = np.dot(invA, B)

            U[:, x] = calc_U(U[:, x], invA, invAxB, r)
            # """
    delta_T = 0  # CALCUL DU NV T_int

    for x in range(E, taille_mat - E + 1):  # On calcule delta T sur les bords intérieurs horizontaux

        delta_T += -λ * (U[E, x] - U[E - 1, x]) * profondeur  # On étudie le bord intérieur haut
        delta_T += -λ * (U[taille_mat - E, x] - U[taille_mat - E + 1, x]) * profondeur  # Le bord intérieur bas

    for y in range(E, taille_mat - E + 1):  # Pareil pour les bords intérieurs verticaux

        delta_T += -λ * (U[y, E] - U[y, E - 1]) * profondeur  # On étudie le bord intérieur gauche
        delta_T += -λ * (U[y, taille_mat - E] - U[y, taille_mat - E + 1]) * profondeur  # On étudie le bord intérieur gauche

    T_int += delta_T


def l(a, b, c, d):
    """Fonction qui ne sert à rien à part lorsque le code bug et que l'on souhaite savoir pourquoi lol"""
    print(len(a), len(b), len(c), len(d))


def calc_U(barre, invA, invAxB, r):  # T0 (resp.T1): température extérieure gauche (resp. droite)
    """Applique l'algorithme de Crank-Nicholson afin de calculer U"""
    longueur = len(barre[1:-1])  # longueur de la barre où l'on change la température

    ### CALCUL DE b ###
    b = np.zeros(longueur)  # ne contiens pas les extrémités : 2 cases en moins

    b[0] = 2 * r * barre[0]
    b[-1] = 2 * r * barre[-1]  # on a mis 2r au lieu d'ajouter bjplus1 [longueur - 1] = r * U[longueur + 1, t]

    barre[1:-1] = np.dot(invAxB, barre[1:-1]) + np.dot(invA, b)  # application de la formule de récurrence
    return barre


########################### A F F E C T A T I O N S #######################

pas_spatial = 10 ** -3  # (en m)
pas_temporel = 100  # (en s)

L = 0.5  # longueur de la boîte (en m)
temps_de_sim = 300  # Temps de la simulation (en s)
epaisseur = 0.02  # son épaisseur (en m)

E = int(epaisseur / pas_spatial)  # conversion de l'épaisseur en nombre de points sur la matrice
taille_mat = 2 * int((L / pas_spatial) / 2) + 1  # correspond au nombre de lignes (= nb colonnes)

N_profondeur = E * 100  # nombre de mesures de la profondeur de la boîte pour négliger les effets de bords

''' ########## POUR UN ORGANE ##########
ρ = 0.550/5.4 #masse volumique de l'organe (poumon gauche: 0.550kg/5.4L)
λ = 0.60 #conductivité thermique de l'eau à 298 K (25 °C)
c = 4.1855*10**3 #capacité thermique massique de l'eau, on assimile l'organe à de l'eau
'''

ρ = 715  # masse volumique du Chêne pédonculé (matériau de la boîte) en kg/L
λ = 0.16  # W/m/K conductivité thermique du bois de chêne à 298 K (25 °C)
c = 2385  # J.kg/K capacité thermique massique du bois de chêne (source: https://www.thermoconcept-sarl.com/base-de-donnees-chaleur-specifique-ou-capacite-thermique/)

alpha = λ / (ρ * c)  # coefficient de diffusivité

r = alpha * pas_temporel / pas_spatial ** 2  # constante utilisée dans le calcul des matrices A et B pour la récurrence

A, B = matrice_AB(r, taille_mat)

T_ext = 35
T_int = 7

########################### D E B U T  D U  P R O G R A M M E #######################
t = time.time()
U, ref = init(T_int, T_ext, taille_mat, taille_mat // 2 - 10)

nb_diterations = int(temps_de_sim / pas_temporel)  # (en s)

for i in range(nb_diterations):  # on calcule U avec n itérations
    calcul_U_t_suivant(U, T_int, taille_mat, E, A, B, λ, N_profondeur, r)
    print(i)  # Pour savoir où on en est


plt.xlabel("Distance (en m)")
plt.ylabel("Distance (en m)")
plt.title('TEMPERATURE 2D')

image = plt.imshow(U, extent=[0, L, 0, L], cmap='afmhot', aspect='auto', animated=True)

cb = plt.colorbar()
cb.set_label("Température (en °C)")
print(time.time() - t)

plt.plot([epaisseur, L - epaisseur, L - epaisseur, epaisseur, epaisseur], [epaisseur, epaisseur, L - epaisseur, L - epaisseur, epaisseur], color='#006a4e')


plt.show()


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
    """ Calcule les deux matrices carrées A et B nécessaires pour le calcul de récurrence du modèle à partir du schéma.
     On ne les calcule qu'une fois afin de réduire le temps d'exécution du programme et de limiter les calculs redondants"""
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


def init(T_int, T_ext, taille_mat, R, E):
    """Initialise la matrice U et crée la matrice ref en fonction des conditions aux initiales"""
    mat_check = np.full((taille_mat, taille_mat), False, dtype=int)
    M = np.full((taille_mat, taille_mat), T_int, dtype=float)  # On impose T_int partout
    M[0], M[-1], M[:, 0], M[:, -1] = (T_ext for _ in range(4))  # puis on impose T_ext à l'extérieur de la boîte, donc aux extrémités de la matrice
    mat_check[0], mat_check[-1], mat_check[:, 0], mat_check[:, -1] = (1 for _ in range(4))
    centre = taille_mat // 2
    for i in range(taille_mat):
        for j in range(taille_mat):
            rayon = ((i - centre) ** 2 + (j - centre) ** 2) ** (1 / 2)
            if rayon >= R:
                M[i, j] = T_ext
                mat_check[i, j] = 1  # Extérieur
            elif rayon >= R - E:
                mat_check[i, j] = -1  # Là où on doit calculer
    return M, mat_check


def c_indices_xy(y, ref, x_or_y):  # x_or_y vaut True si on calcule les lignes et False si on calcule les colonnes
    """Cette fonction renvoie une liste de couples qui sont les valeurs limites de la paroi du système pour chaque ligne"""
    if x_or_y:  # On calcule y
        ind_vrac = np.where(ref[y] == -1)[0]
        c_indices = []
        couple_temp = []

        for j in range(len(ind_vrac)):
            if couple_temp == []:
                couple_temp.append(ind_vrac[j])
            if j != len(ind_vrac) - 1 and ind_vrac[j + 1] != ind_vrac[j] + 1:
                couple_temp.append(ind_vrac[j])
                c_indices.append(couple_temp)
                couple_temp = []
            elif j == len(ind_vrac) - 1:
                couple_temp.append(ind_vrac[j])
                c_indices.append(couple_temp)
        return c_indices
    # On calcule x
    ind_vrac = np.where(ref[:, y] == -1)[0]
    c_indices = []
    couple_temp = []

    for j in range(len(ind_vrac)):
        if couple_temp == []:
            couple_temp.append(ind_vrac[j])
        if j != len(ind_vrac) - 1 and ind_vrac[j + 1] != ind_vrac[j] + 1:
            couple_temp.append(ind_vrac[j])
            c_indices.append(couple_temp)
            couple_temp = []
        elif j == len(ind_vrac) - 1:
            couple_temp.append(ind_vrac[j])
            c_indices.append(couple_temp)
    return c_indices


def calcul_U_t_suivant(U, T_int, taille_mat, E, A, B, λ, profondeur, r, ref):
    """Calcule la matrice Température U au temps suivant en fonction des cas limites, des conditions initiales et de l'épaisseur de la boîte"""

    for y in range(1, taille_mat - 1):  # Calcul pour t variant de t à t+1/2 : On étudie les lignes
        c_indices = c_indices_xy(y, ref, True)
        for c_ind in c_indices:
            i, j = c_ind
            if i != 1:
                i -= 1
            if j != taille_mat - 1:
                j += 1
            l_barre = j - i + 1
            invA = inv(A[: l_barre - 2, : l_barre - 2])
            invAxB = np.dot(invA, B[: l_barre - 2, : l_barre - 2])

            U[y, i:j + 1] = calc_U(U[y, i:j + 1], invA, invAxB, r)

            """
            for y_bord in range(E, taille_mat - E + 1):  # pareil pour les bords intérieurs verticaux
                delta_T += -λ * (U[y_bord, E] - U[y, E - 1]) * E * 100  # on étudie le bord intérieur gauche
                delta_T += -λ * (U[y_bord, taille_mat - E] - U[y_bord, taille_mat - E + 1]) * E * 100  # on étudie le bord intérieur gauche
            """

    for x in range(1, taille_mat - 1):  # Calcul pour t variant de t+1/2 à t+1 : On étudie les colonnes
        c_indices = c_indices_xy(x, ref, False)
        for c_ind in c_indices:
            i, j = c_ind
            if i != 1:
                i -= 1
            if j != taille_mat - 1:
                j += 1
            l_barre = j - i + 1
            invA = inv(A[:l_barre - 2, : l_barre - 2])
            invAxB = np.dot(invA, B[: l_barre - 2, : l_barre - 2])
            U[i:j + 1, x] = calc_U(U[i:j + 1, x], invA, invAxB, r)

            """for x_bord in range(E, taille_mat - E + 1):  # on calcule delta T sur les bords intérieurs horizontaux
                delta_T += -λ * (U[E, x_bord] - U[E - 1, x_bord]) * E * 100  # on étudie le bord intérieur haut
                delta_T += -λ * (U[taille_mat - E, x_bord] - U[taille_mat - E + 1, x_bord]) * E * 100  # le bord intérieur bas
            
    T_int += delta_T
    """

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

rayon_boite = 0.10  # rayon du cercle (en m)
temps_de_sim = 900  # Temps de la simulation (en s)
epaisseur_cercle = 0.04  # son épaisseur (en m)

E = int(epaisseur_cercle / pas_spatial)  # conversion de l'épaisseur en nombre de points sur la matrice
R = int(rayon_boite / pas_spatial)
taille_mat = int(2 * R + 1)  # correspond au nombre de lignes (= nb colonnes)
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
# Plot de l'épaisseur

theta = np.linspace(0, 2 * np.pi, 100)  # Périmètre d'un cercle de rayon 1 = 2*Pi

x_fin_de_combi = (rayon_boite - epaisseur_cercle) * (np.cos(theta)) + rayon_boite
x2_fin_de_combi = (rayon_boite - epaisseur_cercle) * (np.sin(theta)) + rayon_boite
plt.plot(x_fin_de_combi, x2_fin_de_combi, color='#006a4e')

x_début_de_combi = rayon_boite * (np.cos(theta)) + rayon_boite
x2_début_de_combi = rayon_boite * (np.sin(theta)) + rayon_boite
plt.plot(x_début_de_combi, x2_début_de_combi, color='#003a4e')

t = time.time()

U, ref = init(T_int, T_ext, taille_mat, R, E)

plt.imshow(ref)
plt.show()

nb_diterations = int(temps_de_sim / pas_temporel)  # (en s)

plt.imshow(ref)


for i in range(nb_diterations):  # on calcule U avec n itérations
    calcul_U_t_suivant(U, T_int, taille_mat, E, A, B, λ, N_profondeur, r, ref)
    print("Please wait, Graph is loading...")  # Pour savoir où on en est
    print("▓" * int(29 * ((i + 1) / nb_diterations)) + "░" * (29 - int(29 * ((i + 1) / nb_diterations))))
    print(" " * 13 + str(int(100 * (i + 1) / nb_diterations)) + "%" + "" * 13, '\n\n')


plt.xlabel("Distance (en m)")
plt.ylabel("Distance (en m)")
plt.title('TEMPERATURE 2D')

image = plt.imshow(U, extent=[0, 2 * rayon_boite, 0, 2 * rayon_boite], cmap='afmhot', aspect='auto', animated=True)

cb = plt.colorbar()
cb.set_label("Température (en °C)")

print("Temps d'exécution = {} secondes".format(int(time.time() - t)))
plt.show()

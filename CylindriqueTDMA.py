# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:33:37 2021

@author: victo
"""

import numpy as np
# from numpy.linalg import inv
import matplotlib.pyplot as plt
import time


########################### D E F I N I T I O N S #######################

def f(i, j):  # la fonction définissant U[0], la température sur la barre à t = 0
    return abs(10 * np.sin((i + j) / 20) + 20)


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
    for j in range(1, n - 1):  # Relation de réccurence de l'algorithme de Thomas
        diag_du_haut[j] = diag_du_haut[j] / (diag_du_mid[j] - diag_du_bas[j - 1] * diag_du_haut[j - 1])
        D[j] = (D[j] - diag_du_bas[j - 1] * D[j - 1]) / (diag_du_mid[j] - diag_du_bas[j - 1] * diag_du_haut[j - 1])
    D[n - 1] = (D[n - 1] - diag_du_bas[n - 2] * D[n - 2]) / (diag_du_mid[n - 1] - diag_du_bas[n - 2] * diag_du_haut[n - 2])
    solution[n - 1] = D[n - 1]
    for k in range(n - 2, -1, -1):  # Substitution inverse pour trouver le vecteur solution
        solution[k] = D[k] - diag_du_haut[k] * solution[k + 1]
    return solution


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
                M[i, j] = f(i, j)
                mat_check[i, j] = 1  # Extérieur
            elif rayon >= R - E:
                mat_check[i, j] = -1  # Là où on doit calculer
    return M, mat_check


def c_indices_xy(y, ref, line_or_col):  # line_or_col vaut True si on calcule les lignes et False si on calcule les colonnes
    """Cette fonction renvoie une liste de couples qui sont les valeurs limites de la paroi du système pour chaque ligne"""
    if line_or_col:  # On calcule y
        ind_vrac = np.where(ref[y] == -1)[0]
    else:
        ind_vrac = np.where(ref[:, y] == -1)[0]
    c_indices = []
    couple_temp = []
    for j in range(len(ind_vrac)):
        if not couple_temp:
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

    flux_tot = 0
    for y in range(1, taille_mat - 1):  # Calcul pour t variant de t à t+1/2 : On étudie les lignes
        c_indices = c_indices_xy(y, ref, True)
        for c_ind in c_indices:
            i, j = c_ind[0] - 1, c_ind[1] + 1  # On inclut les points dont on connait / impose la température
            l_barre = j - i + 1
            # invA = inv(A[: l_barre - 2, : l_barre - 2])
            # invAxB = np.dot(A[: l_barre - 2, : l_barre - 2], B[: l_barre - 2, : l_barre - 2])

            U[y, i:j + 1] = calc_U(U[y, i:j + 1], A[: l_barre - 2, : l_barre - 2], B[: l_barre - 2, : l_barre - 2], r)

            flux_tot += profondeur * pas_spatial * λ * (U[y, taille_mat - E + 1] - T_int)  # Flux à droite
            flux_tot += profondeur * pas_spatial * λ * (U[y, E - 1] - T_int)  # Flux à gauche

    for x in range(1, taille_mat - 1):  # Calcul pour t variant de t+1/2 à t+1 : On étudie les colonnes
        c_indices = c_indices_xy(x, ref, False)
        for c_ind in c_indices:
            i, j = c_ind[0] - 1, c_ind[1] + 1  # On inclut les points dont on connait / impose la température
            l_barre = j - i + 1
            # invA = inv(A[:l_barre - 2, : l_barre - 2])
            # invAxB = np.dot(invA, B[: l_barre - 2, : l_barre - 2])

            U[i:j + 1, x] = calc_U(U[i:j + 1, x], A[: l_barre - 2, : l_barre - 2], B[: l_barre - 2, : l_barre - 2], r)

            flux_tot += profondeur * pas_spatial * λ * (U[E + 1, x] - T_int)  # Flux en haut
            flux_tot += profondeur * pas_spatial * λ * (U[taille_mat - E - 1, x] - T_int)  # Flux en bas

    # T_int += pas_temporel * flux_tot / C


def new_T_int(C, flux_tot):  # flux_tot est la fonction qui calcule le flux total, elle dépend de t et de T
    """Calcul le nouveau T_int avec la formule : C * d(T_int)/dt = flux_tot."""

    def rk4(f, t0, T_int_0, tmax, h=pas_temporel):  # Si pas précisé, le pas est le même que pour le calcul de température
        """Résolution de l'équation différentielle avec les CL imposées grâce à la méthode Runge-Kutta à l'ordre 4."""
        nb_etape = int((tmax - t0) / h)  # Calcul du nombre d'étapes
        for j in range(nb_etape):
            k1 = h * f(t0, T_int_0)
            k2 = h * f(t0 + h / 2, T_int_0 + k1 / 2)
            k3 = h * f(t0 + h / 2, T_int_0 + k2 / 2)
            k4 = h * f(t0 + h / 2, T_int_0 + k3 / 2)
            k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            T_int_n = T_int_0 + k
            T_int_0 = T_int_n
            t0 += h / 2
        return T_int_n

    return 1 / C * rk4(flux_tot, 0, T_int, temps_de_sim)


def l(a, b, c, d):
    """Fonction qui ne sert à rien à part lorsque le code bug et que l'on souhaite savoir pourquoi lol"""
    print(len(a), len(b), len(c), len(d))


def calc_U(barre, new_A, new_B, r):  # T0 (resp.T1): température extérieure gauche (resp. droite)
    """Applique l'algorithme de Crank-Nicholson afin de calculer U"""
    longueur = len(barre[1:-1])  # longueur de la barre où l'on change la température

    ### CALCUL DE b ###
    b = np.zeros(longueur)  # ne contiens pas les extrémités : 2 cases en moins

    b[0] = 2 * r * barre[0]
    b[-1] = 2 * r * barre[-1]  # on a mis 2r au lieu d'ajouter bjplus1 [longueur - 1] = r * U[longueur + 1, t]

    barre[1:-1] = inverse_tridiag(new_A, np.dot(new_B, barre[1:-1]) + b)
    return barre


########################### A F F E C T A T I O N S #######################

pas_spatial = 1/3 * 10 ** -3  # (en m)
pas_temporel = 10  # (en s)

rayon_boite = 0.10  # rayon du cercle (en m)
temps_de_sim = 3000  # Temps de la simulation (en s)
epaisseur_cercle = 0.04  # son épaisseur (en m)

E = int(epaisseur_cercle / pas_spatial)  # conversion de l'épaisseur en nombre de points sur la matrice
R = int(rayon_boite / pas_spatial)  # conversion du rayon de la boite en nb de points sur la matrice
taille_mat = int(2 * R + 1)  # correspond au nombre de lignes (= nb colonnes)
N_profondeur = E * 100  # nombre de mesures de la profondeur de la boîte pour négliger les effets de bords

''' ########## POUR UN ORGANE ##########
ρ = 0.550/5.4 #masse volumique de l'organe (poumon gauche: 0.550kg/5.4L)
λ = 0.60 #conductivité thermique de l'eau à 298 K (25 °C)
c = 4.1855*10**3 #capacité thermique massique de l'eau, on assimile l'organe à de l'eau
'''

ρ = 715  # masse volumique du Chêne péedonculé (matériau de la boîte) en kg/L
λ = 0.16  # W/m/K conductivité thermique du bois de chêne à 298 K (25 °C)
c = 2385  # J.kg/K capacité thermique massique du bois de chêne (source: https://www.thermoconcept-sarl.com/base-de-donnees-chaleur-specifique-ou-capacite-thermique/)

ρ_int, λ_int, c_int = 1, 1, 1  # mass vol, cond. th. et cap. th du système à l'intérieur de la boite
M_int = ρ_int * ()

alpha = λ / (ρ * c)  # coefficient de diffusivité

r = alpha * pas_temporel / pas_spatial ** 2  # constante utilisée dans le calcul des matrices A et B pour la récurrence

A, B = matrice_AB(r, taille_mat)

T_ext = 35
T_int = 7

########################### D E B U T  D U  P R O G R A M M E #######################
# Plot de l'épaisseur

theta = np.linspace(0, 2 * np.pi, 100)  # Périmètre d'un cercle de rayon 1 = 2*Pi

x_fin_de_combi = (rayon_boite - epaisseur_cercle) * (
    np.cos(theta)) + rayon_boite  # Cercle de rayon = rayon_boite - epaisseur cercle et on veut que le centre soit en rayon_boite et pas en 0
x2_fin_de_combi = (rayon_boite - epaisseur_cercle) * (np.sin(theta)) + rayon_boite
plt.plot(x_fin_de_combi, x2_fin_de_combi, color='#006a4e')

x_début_de_combi = rayon_boite * (np.cos(theta)) + rayon_boite  # Cercle de rayon = rayon_boite et on veut que le centre soit en rayon_boite et pas en 0
x2_début_de_combi = rayon_boite * (np.sin(theta)) + rayon_boite
plt.plot(x_début_de_combi, x2_début_de_combi, color='#003a4e')

t = time.time()

U, ref = init(T_int, T_ext, taille_mat, R, E)

nb_diterations = int(temps_de_sim / pas_temporel)  # (en s)

for i in range(nb_diterations):  # on calcule U avec n itérations
    t_boucle = time.time()
    calcul_U_t_suivant(U, T_int, taille_mat, E, A, B, λ, N_profondeur, r, ref)
    print("Please wait, Graph is loading...")  # Pour savoir où on en est
    print("▓" * int(29 * ((i + 1) / nb_diterations)) + "░" * (29 - int(29 * ((i + 1) / nb_diterations))))
    print(" " * 13 + str(int(100 * (i + 1) / nb_diterations)) + "%" + "" * 13)
    print("Temps restant estimé : {}s".format(int((time.time() - t_boucle) * (nb_diterations - i))), '\n\n')

plt.xlabel("Distance (en m)")
plt.ylabel("Distance (en m)")
plt.title('TEMPERATURE 2D')

image = plt.imshow(U, extent=[0, 2 * rayon_boite, 0, 2 * rayon_boite], cmap='afmhot', aspect='auto', animated=True)

cb = plt.colorbar()
cb.set_label("Température (en °C)")

print("Temps d'exécution = {} secondes".format(int(time.time() - t)))
plt.show()


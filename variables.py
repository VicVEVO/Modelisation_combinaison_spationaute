import numpy as np


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


########   C A R A C T E R I S T I Q U E S  T H E R M I Q U E S  D E S  S Y S T E M E S   ########

ρ = 715  # masse volumique du Chêne pédonculé (matériau de la boîte) en kg/L
λ = 0.16  # W/m/K conductivité thermique du bois de chêne à 298 K (25 °C)
c = 2385  # J.kg/K capacité thermique massique du bois de chêne (source: https://www.thermoconcept-sarl.com/base-de-donnees-chaleur-specifique-ou-capacite-thermique/)
ρ_int, λ_int, c_int = 1, 1, 1  # mass vol, cond. th. et cap. th du système à l'intérieur de la boite
M_int = ρ_int * ()  # Masse du système interne
alpha = λ / (ρ * c)  # coefficient de diffusivité
T_ext = 35  # Température à l'extérieur du système externe
T_int = 7  # Température initiale à l'intérieur du système interne

#########   M E S U R E S   #########

pas_spatial = 1 * 10 ** -3  # (en m)
pas_temporel = 10  # (en s)
temps_de_sim = 3000  # Temps de la simulation (en s)
nb_diterations = int(temps_de_sim / pas_temporel)
r = alpha * pas_temporel / pas_spatial ** 2  # constante utilisée dans le calcul des matrices A et B pour la récurrence
rayon_boite = 0.10  # rayon du système externe (en m)
epaisseur_isolant = 0.04  # distance entre le système externe et le système interne (en m)
E = int(epaisseur_isolant / pas_spatial)  # conversion de l'épaisseur en nombre de points sur la matrice
R = int(rayon_boite / pas_spatial)  # conversion du rayon de la boite en nb de points sur la matrice
taille_mat = int(2 * R + 1)  # correspond au nombre de lignes (= nb colonnes)
N_profondeur = E * 100  # nombre de mesures de la profondeur de la boîte pour négliger les effets de bords
A, B = matrice_AB(r, taille_mat)

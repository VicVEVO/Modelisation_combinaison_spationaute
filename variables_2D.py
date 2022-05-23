import numpy as np


def matrice_AB(r, taille_mat):  # calcul des deux matrices carrées A et B pour le calcul de récurrence du modèle
    # On ne les calcule qu'une fois pour réduire considérablement les calculs répétitifs inutiles

    A = np.zeros((taille_mat - 2, taille_mat - 2))  # A et B sont des matrices initialement nulles
    B = np.zeros((taille_mat - 2, taille_mat - 2))
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


########   C A R A C T E R I S T I Q U E S  D U  S Y S T E M E   ########
# Physiques
longueur_boite = 0.5  # longueur de la boîte (en m)
epaisseur = 0.1  # son épaisseur (en m)
profondeur_boite = 100 * epaisseur  # On suppose que la boite est très épaisse pour négliger les effets de bords

# Thermiques
ρ = np.array([715,156,132])  # masses volumiques - kg.m⁻³
λ = np.array([0.16,0.12,0.03])  # conductivités thermiques - W.m⁻¹.K⁻¹
c = np.array([2385,2642,1132])  # capacités thermiques massiques - J.kg⁻¹.K⁻¹

ρ_int, λ_int, c_int = 1.004, 0.025, 1005  # mass vol, cond. th. et cap. th massique du système à l'intérieur de la boite resp. en kg.m⁻³, en W.m⁻¹.K⁻¹ et en J.kg⁻¹.K⁻¹ (source: Wikipédia)
alphas = λ / (ρ * c)  # coefficients de diffusivité des différentes couches


#########   M E S U R E S   #########
pas_spatial = 100 * 10 ** -5  # (en m) (c'est O(n²) en spat)
pas_temporel = 1  # (en s) (c'est O(n) en temp)
temps_de_sim = 200  # Temps de la simulation (en s)
E = int(epaisseur / pas_spatial)  # conversion de l'épaisseur en nombre de points sur la matrice
taille_mat = int(longueur_boite / pas_spatial)  # correspond au nombre de lignes (= nb colonnes)
r = alphas * pas_temporel / pas_spatial ** 2  # constante utilisée dans le calcul des matrices A et B pour la récurrence
A, B = matrice_AB(r, taille_mat)

import numpy as np


def matrice_AB(r, taille_mat):
    '''
        Calcul des deux matrices carrées A et B pour le calcul de récurrence.
        On ne les calcule qu'une fois pour éviter la redondance de code.
    
        Paramètres
        ----------
        r : flottant.
        taille_mat : entier.
            Correspond au nombre de lignes/colonnes de la matrice 
    
        Retours
        -------
        A : tableau numpy
        B : tableau numpy
    '''

    A = np.zeros((taille_mat - 2, taille_mat - 2))
    B = np.zeros((taille_mat - 2, taille_mat - 2))
    
    for i in range(taille_mat - 2):
        for j in range(taille_mat - 2):
            if i == j: # Étude de la diagonale centrale
                A[i][j] = 2 + 2 * r
                B[i][j] = 2 - 2 * r
            elif (j == i + 1) or (i == j + 1): # Diagonales secondaires
                A[i][j] = -r
                B[i][j] = r
    return A, B


#######   C A R A C T E R I S T I Q U E S  D U  S Y S T E M E (U.S.I)  #######

longueur_boite = .5
epaisseur = 0.15 / 20 
profondeur_boite = 100 * epaisseur


T_ext = -180
T_int = 37.5
T_int_init = T_int


pas_spatial = 10 * 10 ** -5 
pas_temporel = 1
temps_de_sim = 180


E = int(epaisseur / pas_spatial)  # Conversion: épaisseur en nombre de points
taille_mat = int(longueur_boite / pas_spatial)
nb_iterations = int(temps_de_sim / pas_temporel)


ρ = 715  # kg.m⁻³: Masse volumique du Chêne pédonculé 
λ = 0.16  # W.m⁻¹.K⁻¹: Conductivité thermique du bois de chêne à 298 K (25 °C)
c = 2385  # J.kg⁻¹.K⁻¹: Capacité thermique massique du bois de chêne

ρ_int, λ_int, c_int = 1.004, 0.025, 1005

a = λ / (ρ * c)  # Coefficient de diffusivité (dans l'équation de la chaleur)
r = a * pas_temporel / pas_spatial ** 2  # constante utilisée dans A et B
A, B = matrice_AB(r, taille_mat)

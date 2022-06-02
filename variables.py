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
            if i == j:  # Étude de la diagonale centrale
                A[i][j] = 2 + 2 * r
                B[i][j] = 2 - 2 * r
            elif (j == i + 1) or (i == j + 1):  # Diagonales secondaires
                A[i][j] = -r
                B[i][j] = r
    return A, B


def choix_materiau(nom_materiau):
    if nom_materiau == 'bois':
        return 715, .16, 2385  # masse volumique, conductivité thermique et capacité thermique massique du Chêne pédonculé
    elif nom_materiau == 'kevlar':
        return 1440, 0.04, 1420
    elif nom_materiau == 'aluminium':
        return 2699, 237, 897
    elif nom_materiau == 'BOPET':  # Marque connue : Mylar
        return 1350, 0.13, 1500
    elif nom_materiau == 'nylon':
        return 1240, 0.25, 1500
    else:
        return None, None, None

#######   C A R A C T E R I S T I Q U E S  D U  S Y S T E M E (U.S.I)  #######

longueur_boite = 2 * 10 ** -1  # longueur de la boîte (en m)
epaisseur = 5 * 10 ** -2  # son épaisseur (en m)
profondeur_boite = 100 * epaisseur  # On suppose que la boite est très épaisse pour négliger les effets de bords

T_ext = -160
T_int = 36.5
T_int_init = T_int

pas_spatial = 1 * 10 ** -3  # (en m) (c'est O(n²) en spat)
pas_temporel = 12  # (en s) (c'est O(n) en temp)

temps_de_sim = 5400  # Temps de la simulation (en s)


E = int(epaisseur / pas_spatial)  # conversion de l'épaisseur en nombre de points sur la matrice
taille_mat = int(longueur_boite / pas_spatial)  # correspond au nombre de lignes (= nb colonnes)

nb_iterations = int(temps_de_sim / pas_temporel)

liste_materiau = ['bois', 'kevlar', 'BOPET', 'nylon']
materiau = liste_materiau[1]
ρ, λ, c = choix_materiau(materiau)

ρ_int, λ_int, c_int = 1.004, 0.025, 1005  # air

a = λ / (ρ * c)  # Coefficient de diffusivité (dans l'équation de la chaleur)
r = a * pas_temporel / pas_spatial ** 2  # constante utilisée dans A et B
A, B = matrice_AB(r, taille_mat)

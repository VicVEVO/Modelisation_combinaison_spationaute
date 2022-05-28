import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.cosh(x) + 15


def g(x):
    return abs(100 * np.sin(x / 2) + 20)


def calcul_temp(r, nb_pts_espace, nb_pts_temps, e0, emax, temps0):
    # M_r = matrice_r(r, nligne)
    Temp = np.zeros((nb_pts_espace, nb_pts_temps))
    Temp[0] = e0
    Temp[-1] = emax
    Temp[:, 0] = temps0
    for temps in range(1, nb_pts_temps):
        Temp[1:nb_pts_espace - 1, temps] = r*Temp[2:nb_pts_espace, temps - 1] + (1-2*r)*Temp[1:nb_pts_espace-1, temps - 1]\
                                           + r * Temp[0:nb_pts_espace - 2, temps - 1]
    return Temp


ρ = 1440  # masse volumique du Chêne pédonculé (matériau de la boîte)
λ = 0.04  # W/m/K conductivité thermique du bois de chêne à 298 K (25 °C)
c = 1420  # J.kg/K capacité thermique massique du bois de chêne (source: https://www.thermoconcept-sarl.com/base-de-donnees-chaleur-specifique-ou-capacite-thermique/)
alpha = λ / (ρ * c)  # coefficient de diffusivité

###################### DEFINITION DES VARIABLES / CONSTANTES ########################


def matrice_r(r, nligne):
    m = np.zeros((nligne, nligne))
    for i in range(1, nligne):
        if i != nligne - 1:
            m[i][i] = 1
            m[i][i-1] = r
            m[i][i+1] = r
        else:
            m[i][i] = 1 - 2 * r
    return m

longueur_fil = 10*10**-2  # (en m)
temps_de_simulation = 500  # (en s)
pas_spatial = 1.9 * 10 ** -4  # (en m)
pas_temporel = 1  # (en s)

nb_iterations = int(temps_de_simulation / pas_temporel)
N_longueur = int(longueur_fil / pas_spatial)

tempszero = np.ones(N_longueur)  # valeur de la temperature en t = 0 pour tout x ## mettre une fonction genre cos ou autre
espacezero = np.ones(nb_iterations)  # valeur de la temperature en x = 0 pour tout t
espacemax = np.ones(nb_iterations)  # valeur de la température en x = xmax pour tout t

r = alpha * pas_temporel / (pas_spatial ** 2)
print(r)
espacezero *= 35  # Temp en x = 0
espacemax *= 35  # Temp en x = L
tempszero[0], tempszero[-1] = 35, 35

for indice in range(1, N_longueur - 1):
    # tempszero[indice] = f(-3.68 + 7.36 * indice/(N_longueur - 1))  # Conditions initiales sur la barre (à t = 0)
    tempszero[indice] = g(indice * 0.1)

####################################################################################

U = calcul_temp(r, N_longueur, nb_iterations, espacezero, espacemax, tempszero)

plt.xlabel("Durée (s)")
plt.ylabel("Distance (m)")
plt.title('TEMPERATURE 1D')
# V = np.zeros((N_longueur, nb_iterations))
# for i in range(nb_iterations):
#     V[:, i] = U[:, 0]

plt.imshow(U, extent=[0, temps_de_simulation, 0, longueur_fil], aspect='auto', cmap='afmhot')

cb = plt.colorbar()
cb.set_label("Température (°c)")
plt.show()


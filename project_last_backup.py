
# -*- coding: utf-8 -*-
"""
Un programme du TIPE étudiant les transferts thermiques à 1D (sur une barre) avec la méthode de Crank-Nicolson.
La matrice U représente la température, sa k-ième colonne correspond à la température sur la barre à l'instant k.
                                        sa j-ième ligne correspond à la température au point j de la barre pour tout instant dans l'intervalle défini au début.
"""

import numpy as np
from numpy.linalg import inv

import seaborn as sns; sns.set_theme()

precision = 70 #paramètre temporaire bourrin qu'on utilise pour l'intervalle de mesure

nlignes , ncolonnes = precision, precision

xmax , tmax = 10 , 10 #Valeur de mesure spatiale (resp. temporelle) max
alpha = 10**-5 # k/rho*c_p #coeff de diffusivité

h_x , h_t = xmax/nlignes , tmax/ncolonnes #"correspondent" à dx , dt
r = alpha*h_t/(h_x**2) #constante utilisée dans le calcul des matrices A et B pour la récurrence


def f(x): #la fonction définissant la température U[0] sur la barre à t = 0
    return(abs(100*np.sin(x/2) + 20))

def matrice_AB(r, nlignes, ncolonnes): #calcul des deux matrices constantes A et B pour le calcul de récurrence du modèle
    
    nlignes,ncolonnes = nlignes -2, ncolonnes-2
    A = np.zeros((nlignes,ncolonnes))
    B = np.zeros((nlignes,ncolonnes))
    
    for i in range(nlignes):
        for j in range(ncolonnes):
            if i == j:
                A[i][j] = 2 + 2 * r
                B[i][j] = 2 - 2 * r
            elif (j == i + 1) or (i == j+1):
                A[i][j] = -r
                B[i][j] = r
    return inv(A), B 

def init(f,nlignes,ncolonnes): #initialisation de la matrice U
    M = np.zeros((nlignes,ncolonnes))
    for i in range(nlignes): #température sur la barre à t = 0
        M[i][0] = f(h_x*i)
    for i in range(ncolonnes): #température constante aux bords de la barre
        M[0][i] = T0
        M[nlignes-1] = f((nlignes-1)*h_x)
    return M

def matrice_U(f, nlignes, ncolonnes, r, invA, B): #calcule la matrice U (lignes (resp. colonnes) correspondent aux x (resp. t)) par récurrence
    
    U = init(f,nlignes+2,ncolonnes+2) #initialise U
    invAxB = np.dot(invA,B) #calcule une fois A^-1*B pour éviter des calculs redondants
    
    for t in range(1,ncolonnes+2):
        
        b = np.zeros(nlignes)
        b[0] = r*U[0,t-1]   #calcul de la matrice colonne b, quasi nulle mais nécessaire dans le calcul par récurrence 
        b[nlignes-1] = r*U[nlignes+1,t-1]

        U[1:nlignes+1,t] = np.dot(invAxB,U[1:nlignes+1,t-1]) + np.dot(invA,b) #application de la formule de récurrence
        
    return U

"""   ### DÉBUT DU PROGRAMME ###   """
invA,B = matrice_AB(r,nlignes+2,ncolonnes+2)
U = matrice_U(f,nlignes, ncolonnes, r, invA, B)

if precision <= 10: #On affiche pour l'instant les label que s'ils sont peu pour éviter l'illisibilité et le temps d'affichage trop coûteux
    ax = sns.heatmap(U,
                     xticklabels = np.around(np.arange(0, tmax + 2*h_t, h_t),1),
                     yticklabels = np.around(np.arange(0, xmax + 2*h_x, h_x),1),
                     cbar_kws={'label': 'Temperature'},
                     center=0)
else :
    ax = sns.heatmap(U,
                     cbar_kws={'label': 'Temperature'},
                     center=0)
    
ax.set(title="TEMPERATURE 1D",
      xlabel="Durée (s)",
      ylabel="Distance (m)")

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:33:37 2021

@author: victo
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

########################### D E F I N I T I O N S #######################

def f(x):  # la fonction définissant U[0], la température sur la barre à t = 0
    return abs(100 * np.sin(x/2) + 20)


def matrice_AB(r, nlignes, ncolonnes):  # calcul des deux matrices constantes A et B pour le calcul de récurrence du modèle
    #On ne les calcule qu'une fois pour réduire considérablement les calculs répétitifs inutiles
    nlignes, ncolonnes = nlignes - 2, ncolonnes - 2 #il faut qu'on fixe ça

    A = np.zeros((nlignes, ncolonnes)) #A et B sont des matrices initialement nulles
    B = np.zeros((nlignes, ncolonnes)) #que en fonction de nlignes, ncolonnes utilisé à la fin

    for i in range(nlignes):
        for j in range(ncolonnes):#correspond au temps
            if i == j:
                A[i][j] = 2 + 2 * r
                B[i][j] = 2 - 2 * r
            elif (j == i + 1) or (i == j + 1):
                A[i][j] = -r
                B[i][j] = r
    return inv(A), B #on renvoie directement l'inverse de A, car on n'utilise que son inverse


def init(T_int,T_ext, nlignes, ncolonnes):  # initialisation de la matrice U
    
    M = np.full((nlignes, ncolonnes),T_int)  #On impose T_int partout
    M[0],M[-1],M[:,0],M[:,-1] = T_ext,T_ext,T_ext,T_ext #puis on impose T_ext à l'extérieur de la boîte, donc aux extrémités de la matrice
    
    return M


def calcul_U_t_suivant(U,T_int,nlignes,ncolonnes,E,invA,invAxB,λ,profondeur,r):#calcule la matrice de la température U au temps suivant

    for y in range(1,nlignes-1): #balayage de t à t+1/2
            
        if E < y <= nlignes - E: #on étudie le cas où on est au middle
            U[y,:E] = calc_U(U[y,:E],invA,invAxB,r)
            U[y,ncolonnes-E:] = calc_U(U[y,ncolonnes-E:],invA,invAxB,r)
            
        else:
            U[y] = calc_U(U[y],invA,invAxB,r) #calc_U(matrice 1D, CL1, CL2)
                
    for x in range(1,ncolonnes-1): #balayage de t+1/2 à t+1
    
        if E < x <= ncolonnes - E:
            U[:E,x] = calc_U(U[:E,x],invA,invAxB,r)
            U[nlignes-E:,x] = calc_U(U[nlignes-E:,x],invA,invAxB,r)
            
        else:
            U[:,x] = calc_U(U[:,x],invA,invAxB,r)
    
    ##CALCUL DU NV T_int
        delta_T = 0
        
        for x in range(E , ncolonnes-E+1): #on calcule delta T sur les bords intérieurs horizontaux
 
            delta_T += -λ*(U[E,x]-U[E-1,x])*profondeur #on étudie le bord intérieur haut
            delta_T += -λ*(U[nlignes-E,x]-U[nlignes-E+1,x])*profondeur #le bord intérieur bas
        
        for y in range(E, nlignes-E+1):#pareil pour les bords intérieurs verticaux
            
            delta_T += -λ*(U[y,E]-U[y,E-1])*profondeur #on étudie le bord intérieur gauche
            delta_T += -λ*(U[y,ncolonnes-E]-U[y,ncolonnes-E+1])*profondeur #on étudie le bord intérieur gauche
        
        T_int += delta_T
            
    return U

def calc_U(barre,invA,invAxB,r): #T0 (resp.T1): température extérieure gauche (resp. droite)
    
    longueur = len(barre[1:-1]) # longueur de la barre où l'on change la température
    ### CALCUL DE b ###
    b = np.zeros(longueur) #ne contient pas les extrémités: 2 cases en moins
    
    b[0] = r * barre[0]
    b[-1] = 2*r * barre[-1] # on a mis 2r au lieu d'ajouter bjplus1 [longueur - 1] = r * U[longueur + 1, t]

    #il faut diminuer invAxB de 2
    barre[1:-1] = np.dot(invAxB[:longueur,:longueur], barre[1:-1]) + np.dot(invA[:longueur,:longueur],b) # application de la formule de récurrence
    return(barre)
        
########################### A F F E C T A T I O N S #######################

precision = 100  # paramètre que l'on définit pour l'intervalle de mesure spatiale

nlignes, ncolonnes = precision, precision

xmax, ymax = 10, 10 
E = 10 # nb de valeurs de mesure de l'épaisseur
profondeur = E*100

''' ########## POUR UN ORGANE ##########
ρ = 0.550/5.4 #masse volumique de l'organe (poumon gauche: 0.550kg/5.4L)
λ = 0.60 #conductivité thermique de l'eau à 298 K (25 °C)
c = 4.1855*10**3 #capacité thermique massique de l'eau, on assimile l'organe à de l'eau
'''

ρ = 0.715 #masse volumique du Chêne pédonculé (matériau de la boîte)
λ = .16 #conductivité thermique du bois de chêne à 298 K (25 °C)
c = 2385 #capacité thermique massique du bois de chêne (source: https://www.thermoconcept-sarl.com/base-de-donnees-chaleur-specifique-ou-capacite-thermique/)

alpha = λ/(ρ*c) #coefficient de diffusivité

h_x, h_y = xmax / nlignes, ymax / ncolonnes # "correspondent" à dx , dy
r = alpha * h_y / (h_x ** 2)  # constante utilisée dans le calcul des matrices A et B pour la récurrence

invA, B = matrice_AB(r, nlignes, ncolonnes)
invAxB = np.dot(invA, B)  # calcule une fois A^-1*B pour éviter des calculs redondants

T_ext = 35
T_int = 7

########################### D E B U T  D U  P R O G R A M M E #######################

U = init(T_int,T_ext, nlignes, ncolonnes)

for i in range(500): #on calcule U pour t = 500
    U = calcul_U_t_suivant(U,T_int,nlignes,ncolonnes,E,invA,invAxB,λ,profondeur,r)


plt.xlabel("Durée (s)")
plt.ylabel("Distance (m)")
plt.title('TEMPERATURE 1D')

plt.imshow(U,extent = [0,ymax,0,xmax], aspect = 'auto',cmap = 'afmhot')

cb = plt.colorbar()
cb.set_label("Température (°c)") 

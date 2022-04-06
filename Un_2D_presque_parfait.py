    # -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 11:31:14 2022
@author: victo
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import time


########################### D E F I N I T I O N S #######################

def f(i,j): #fonction simulant la température extérieure
    return abs(10*np.sin((i+j) / 20) + 20)


def matrice_AB(r, taille_mat):  # calcul des deux matrices carrées A et B pour le calcul de récurrence du modèle
    # On ne les calcule qu'une fois pour réduire considérablement les calculs répétitifs inutiles

    A = np.zeros((taille_mat - 2, taille_mat - 2))  # A et B sont des matrices initialement nulles
    B = np.zeros((taille_mat - 2, taille_mat - 2))  # que en fonction de nlignes, ncolonnes utilisé à la fin
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


def init(mat_ref, T_int, T_ext, taille_mat):  # initialisation de la matrice U

    M = np.full((taille_mat, taille_mat), T_int, dtype=float)  # On impose T_int partout
    for i in range(taille_mat):
        for j in range(taille_mat):
            if mat_ref[i,j] == 1: # si on est à l'extérieur
                M[i,j] = f(i,j)#T_ext # On impose T_ext
                
    return M


def calcul_U_t_suivant(U, T_int, taille_mat, E, A, B, λ, profondeur, r):  # calcule la matrice de la température U au temps suivant

    for y in range(1, taille_mat - 1):  # balayage de t à t+1/2: on étudie les lignes
        c_indices = c_indices_y(y,mat_ref)
        if c_indices != None:
            for c_ind in c_indices:
                (i,j) = c_ind # extrémités de la barre étudiée
                if i!=1: #on considère les barres avec leur CL
                    i -= 1
                if j!= taille_mat-1:
                    j += 1
                l_barre = j-i+1 # sa longueur
                invA = inv(A[:l_barre-2,:l_barre-2])
                invAxB = np.dot(invA,B[:l_barre-2,:l_barre-2])
    
                U[y,i:j+1] = calc_U(U[y,i:j+1],invA,invAxB,r) #on y calcule la nouvelle température
        
    for x in range(1, taille_mat - 1):  # balayage de t+1/2 à t+1: on étudie les colonnes
        c_indices = c_indices_x(x,mat_ref)
        if c_indices != None:
            for c_ind in c_indices:
                (i,j) = c_ind # extrémités de la barre étudiée
                if i!=1: #on considère les barres avec leur CL
                    i -= 1
                if j!= taille_mat-1:
                    j += 1
                l_barre = j-i+1 # sa longueur
                invA = inv(A[:l_barre-2,:l_barre-2])
                invAxB = np.dot(invA,B[:l_barre-2,:l_barre-2])
                
                U[i:j+1,x] = calc_U(U[i:j+1,x],invA,invAxB,r) #on y calcule la nouvelle température
    
    
    gradient_coor = grad_coords(T_int,U,coor_b_int,mat_ref)
    flux_tot = flux_couples(coor_b_int,gradient_coor)
                
    T_int += pas_temporel*flux_tot/C_int
    
    for i in range(taille_mat):
        for j in range(taille_mat):
            if mat_ref[i,j] == 0: # si on est à l'intérieur
                U[i,j] = T_int # on change la température intérieure


def calc_U(barre, invA, invAxB, r):  # T0 (resp.T1): température extérieure gauche (resp. droite)
    longueur = len(barre[1:-1])  # longueur de la barre où l'on change la température

    ### CALCUL DE b ###
    b = np.zeros(longueur)  # ne contient pas les extrémités: 2 cases en moins

    b[0] = 2 * r * barre[0]
    b[-1] = 2 * r * barre[-1]  # on a mis 2r au lieu d'ajouter bjplus1 [longueur - 1] = r * U[longueur + 1, t]

    barre[1:-1] = np.dot(invAxB, barre[1:-1]) + np.dot(invA, b)  # application de la formule de récurrence
    return barre


def ref(x,y,R): #x,y: taille matrice
    mat_ref = np.empty((x,y),dtype=int)
    for i in range(x):
        for j in range(y):
            rayon = int(((i-x/2)**2+(j-y/2)**2)**0.5)
            if rayon > R-1: #j'ai fix en mettant -1: le cercle est centré et ne touche plus les bords
                mat_ref[i,j] = 1 #extérieur
            else:
                if rayon > R-E-1:
                    mat_ref[i,j] = -1 #bordures
                else:
                    mat_ref[i,j] = 0 #intérieur
    
    for i in range(1,len(mat_ref[:,0])-1): #on balaye verticalement
        for j in range(len(mat_ref[0])): #balayage horizontal
            
            
            if mat_ref[i,j] == 0 and (mat_ref[i,j+1] == -1 or mat_ref[i-1,j] == -1): #Si le point est sur le bord intérieur droit ou en haut
                mat_ref[i,j] = 2
            elif mat_ref[i,j] == 0 and (mat_ref[i,j-1] == -1 or mat_ref[i+1,j] == -1): #soit on en sort ou en bas
                mat_ref[i,j] = 2
    
    
    return mat_ref

'''
def coords_bords_int(mat_ref):
    """
    Renvoie une liste python avec toutes les coords des bords intérieurs pour calculer le flux
    """
    
    for i in range(1,len(mat_ref[:,0])-1): #on balaye verticalement
        for j in range(len(mat_ref[0])): #balayage horizontal
            if mat_ref[i,j] == 0:
                if mat_ref[i,j+1] == -1:
                    lieu = 'droite'
                elif mat_ref[i,j-1] == -1:
                    lieu = 'gauche'
                elif mat_ref[i-1,j] == -1 or mat_ref[i+1,j] == -1:
                    lieu = 'droite'
      
            if lieu == 'gauche':
                gauche = [[i,j]] + gauche
            else:
                droite.append([i,j])
    return coords
'''

def liste_entourants():
    liste = []
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            if not (i==j==0):
                liste.append([i,j])
    return(liste)



def init_coords_bords_int_rangees(mat_ref,liste_ent):
    '''
    Initialise la liste coords_bords_int_rangees et la renvoie avec le nb de points qu'elle devra avoir
    '''
    nb_pts = 0
    for i in range(len(mat_ref[0])):
        for j in range(len(mat_ref[0])):
            if mat_ref[i,j] == 2:
                nb_pts += 1
    
    coords_ordre = []
    point_init = []
    j = 0 #servira au balayage horizontal
    
    while point_init == []:
        if j < len(mat_ref[0]): #balayage horizontal
            if mat_ref[len(mat_ref[0])//2,j] == 2: #Si le point est sur le bord intérieur gauche
                point_init = [len(mat_ref[0])//2,j]
        j += 1
    
    coords_ordre.append(point_init)
    candidats = []
    x_0,y_0 = point_init[0],point_init[1]
    
    for i,j in liste_ent:
        if mat_ref[x_0+i,y_0+j] == 2 and [x_0+i,y_0+j] not in coords_ordre:
            candidats.append([i+x_0,j+y_0])
                
    if candidats[0][0] < candidats[1][0]: #si le premier candidat est 'plus haut'
        coords_ordre.append(candidats[0])
    else:
        coords_ordre.append(candidats[1])
    
    return(coords_ordre,nb_pts)


def coords_bords_int_rangees(mat_ref):
    '''
    Renvoie une liste python avec toutes les coords des bords intérieurs pour calculer le flux
    Range les coords de telle sorte qu'ils soient rangés successivement selon le sens horaire
    '''
    
    liste_ent = liste_entourants()
    coords_ordre,nb_pts = init_coords_bords_int_rangees(mat_ref,liste_ent)

    x,y = coords_ordre[-1][0], coords_ordre[-1][1]
    
    while len(coords_ordre) != nb_pts :
        for i,j in liste_ent:
            if mat_ref[i+x,j+y] == 2 and [i+x,j+y] not in coords_ordre:
                coords_ordre.append([i+x,j+y])
                x,y = i+x,j+y

    return coords_ordre


def grad_coords(T_int,U,coords,mat_ref): # on prend en paramètre les coords rangées
    '''
    Renvoie la liste des gradients associés aux coordonnées sur les bords intérieurs
    '''
    grad = []
    if coords != None: # A REMOVE
        for x,y in coords:
            grad_temp = [0,0]
            poids_tot = 0
            vois = voisins_entourants_connus(x, y, mat_ref)
            
            for a,b in vois: # on calcule le gradient pour chaque points autour de (x,y) connus vers (x,y)
                if a != x and b != y: # si (x,y) et (a,b) sont diagonaux
                    poids = np.sqrt(2)/2
                else:
                    poids = 1
                grad_temp[0] += poids * (T_int - U[a,b]) * (y-b) #U[x,y] = T_int ; y-b = ± 1 pour le signe
                grad_temp[1] += poids * (T_int - U[a,b]) * (a-x)
                poids_tot += poids
                
            grad.append(np.array(grad_temp)/poids_tot)
        return grad
    return([])

def voisins_entourants_connus(x,y,mat_ref):
    '''
    Retourne les couples des voisins entourant (x,y) connus
    '''
    vois = []
    
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            vois.append((x+i,y+j))
                
    return vois


def flux_couples(coords,grad_coords): # on prend en paramètre les coords rangées
    flux_tot = 0
    for i in range(len(coords)-1): # on calcule le flux de chaque couple de coordonnées
        if coords[i][0] == coords[i+1][0]: # si les 2 points consécutifs sont sur la même colonne
            if coords[i][1] > taille_mat/2: # on est "à droite" de la matrice
                n = [1,0]
            else:
                n = [-1,0]
        elif coords[i][1] == coords[i+1][1]: # les deux points consécutifs sont sur la même ligne
            if coords[i][0] < taille_mat/2: # on est "en haut"
                n = [0,1]
            else:
                n = [0,-1]
        flux = -λ_int * pas_spatial * scalaire((grad_coords[i]+grad_coords[i+1])/2,n) * N_profondeur
        flux_tot += flux
    return flux_tot


def scalaire(A,B): # produit scalaire canonique en dimension 2
    return A[0]*B[0] + A[1]*B[1]


def c_indices_y(y,mat_ref): #renvoie les couples d'indices des extrémités des barres intermédiaires pour y donné
    ind_vrac = np.where(mat_ref[y] == -1)[0] #indices en vrac où on calcule la température par conduction 
    c_indices = []
    couple_temp = []
    
    for i in range(len(ind_vrac)):
        if couple_temp == []:
            couple_temp.append(ind_vrac[i])
        if i != len(ind_vrac)-1 and ind_vrac[i+1] != ind_vrac[i]+1:
            couple_temp.append(ind_vrac[i])
            c_indices.append(couple_temp)
            couple_temp = []
        elif i == len(ind_vrac)-1:
            couple_temp.append(ind_vrac[i])
            c_indices.append(couple_temp)
            return c_indices
        
def c_indices_x(x,mat_ref): #renvoie les couples d'indices des extrémités des barres intermédiaires pour y donné
    ind_vrac = np.where(mat_ref[:,x] == -1)[0] #indices en vrac où on calcule la température par conduction 
    c_indices = []
    couple_temp = []
    
    for i in range(len(ind_vrac)):
        if couple_temp == []:
            couple_temp.append(ind_vrac[i])
        if i != len(ind_vrac)-1 and ind_vrac[i+1] != ind_vrac[i]+1:
            couple_temp.append(ind_vrac[i])
            c_indices.append(couple_temp)
            couple_temp = []
        elif i == len(ind_vrac)-1:
            couple_temp.append(ind_vrac[i])
            c_indices.append(couple_temp)
            return c_indices


def affichage_cercle(rayon):
    theta = np.linspace(0, 2 * np.pi, 100)  # Périmètre d'un cercle de rayon 1 = 2*Pi

    x_fin_de_combi = (rayon - epaisseur_cercle) * (np.cos(theta)) + rayon # Cercle de rayon = rayon_boite - epaisseur cercle et on veut que le centre soit en rayon_boite et pas en 0
    x2_fin_de_combi = (rayon - epaisseur_cercle) * (np.sin(theta)) + rayon
    plt.plot(x_fin_de_combi, x2_fin_de_combi, color='#003a4e')
    
    x_début_de_combi = rayon * (np.cos(theta)) + rayon # Cercle de rayon = rayon_boite et on veut que le centre soit en rayon_boite et pas en 0
    x2_début_de_combi = rayon * (np.sin(theta)) + rayon
    plt.plot(x_début_de_combi, x2_début_de_combi, color='#003a4e')

def affichage_figure(U,rayon):
    plt.xlabel("Distance (en m)")
    plt.ylabel("Distance (en m)")
    plt.title('TEMPERATURE 2D')
    plt.imshow(U, extent=[0, 2*rayon, 0, 2*rayon], aspect='auto', cmap='afmhot')
    cb = plt.colorbar()
    cb.set_label("Température (en °C)")
    
    plt.show()


########################### A F F E C T A T I O N S #######################


pas_spatial = 10 ** -3   # (en m)
pas_temporel = 10  # (en s)

rayon = 0.09800  # rayon du cercle (en m)

temps_de_sim = 3000  # Temps de la simulation (en s)
epaisseur_cercle = 0.004  # son épaisseur (en m)

E = int(epaisseur_cercle / pas_spatial)  # conversion de l'épaisseur en nombre de points sur la matrice
R = int(rayon / pas_spatial)

taille_mat = int(2*(rayon / pas_spatial + 1))  # correspond au nombre de lignes (= nb colonnes)

N_profondeur = E * 100  # nombres de mesures de la profondeur de la boîte pour négliger les effets de bords

ρ = 715   # masse volumique du Chêne pédonculé en kg/L
c = 2385  # J.kg/K capacité thermique massique du bois de chêne (source: https://www.thermoconcept-sarl.com/base-de-donnees-chaleur-specifique-ou-capacite-thermique/)
λ = 0.16 # conductivité thermique du bois de chêne à 298 K (W/m/K)

rho_int = 1004 # masse volumique de l'air (kg/L)
λ_int = 0.025 # conductivité thermique de l'air(W/m/K)
c_int = 1005 # capacité thermique massique de l'air (J/kg/K)
C_int = c_int*rho_int*np.pi*(rayon-epaisseur_cercle)**5*100*epaisseur_cercle # Capacité thermique à l'intérieur

alpha = λ / (ρ * c) # coefficient de diffusivité

r = alpha * pas_temporel / pas_spatial ** 2  # constante utilisée dans le calcul des matrices A et B pour la récurrence

A, B = matrice_AB(r, taille_mat)
mat_ref = ref(taille_mat,taille_mat,R)

T_ext = 35
T_int = 7


coor_b_int = coords_bords_int_rangees(mat_ref)





########################### D E B U T  D U  P R O G R A M M E #######################
t = time.time()
U = init(mat_ref,T_int, T_ext, taille_mat)

nb_diterations = int(temps_de_sim / pas_temporel)  # (en s)

print("Nombre d'itérations: {} \nTaille matrice: {}x{}".format(nb_diterations,taille_mat,taille_mat))

for i in range(nb_diterations):  # on calcule U avec n itérations
    calcul_U_t_suivant(U, T_int, taille_mat, E, A, B, λ, N_profondeur, r)
    if i%5 == 0 or i == nb_diterations-1:
        print("\n"+"▓" * int(29 * ((i + 1) / nb_diterations)) + "░" * (29 - int(29 * ((i + 1) / nb_diterations))))
        print(" " * 13 + str(int(100 * (i + 1) / nb_diterations)) + "%" + "" * 13, 5*'\n')

affichage_figure(U,rayon)
affichage_cercle(rayon)

print('\n durée de simulation: {}s'.format(int(time.time() - t)))

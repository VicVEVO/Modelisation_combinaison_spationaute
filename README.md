# TIPE-2021

**French / Français**

Résumé des calculs utilisés pour le programme: http://www.claudiobellei.com/2016/11/01/implicit-parabolic/

Problèmes: Pour tmax < 100 000, pas d'évolution visible pour la température
           Pour tmax > 100 000, évolution de T légitime mais discontinuité sur les bords en affichant la température aux extrémités de la barre en ajoutant à la fin du programme:
                            print(U[0][-1],U[1][-1])
                            print(U[-1][-1],U[-2][-1])
                            
Donc: Erreur de calcul dans le programme ou méthode de Crank-Nicholson limitée (étrange car numériquement stable et correcte)? 

**English**

The calculating algorithm used is the one showed on the website : http://www.claudiobellei.com/2016/11/01/implicit-parabolic/

Problems : For any tmax value under 100 000, no noticeable evolution of the temperature, whlle the edges are heating, the whole beam is cold (Err
           For any tmax value over 100 000, the temperature evolution seems legit, even though the temperature on the edges is totally non-sense.
           
_Forgive us if you see any English error, we're a group of baguettes doing studies (we're doing our best to be clear though) ;)

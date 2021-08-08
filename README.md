# TIPE-2021

Résumé des calculs utilisés pour le programme: http://www.claudiobellei.com/2016/11/01/implicit-parabolic/

Problèmes: Pour tmax < 100 000, pas d'évolution visible pour la température
           Pour tmax > 100 000, évolution de T légitime mais discontinuité sur les bords en affichant la température aux extrémités de la barre en ajoutant à la fin du programme:
                            print(U[0][-1],U[1][-1])
                            print(U[-1][-1],U[-2][-1])
                            
Donc: Erreur de calcul dans le programme ou méthode de Crank-Nicholson limitée (étrange car numériquement stable et correcte)? 

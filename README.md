# Segmentation-planche-et-Reconstruction-Billon

Utilisation : 
- Segmentation des planches :
Ajouter les images des piles de planches à segmenter dans le dossier “PilePlanches”.
Lancer l’application.
Appuyer sur la touche “s”.
Entrer le nom du fichier de l’image à segmenter, par exemple “planks-color-1.jpeg”.
Mesurer la hauteur et la largeur d’une planche dans l’image a segmenter et les entrer dans l’application, par exemple largeur “395” et hauteur “136”.
Les résultats seront affichés dans la fenêtre “Res hough”.
Chacune des régions trouvées dans l'image seront exportées au format png dans le dossier “planches” et nommées pX.png avec X un nombre donné à la région dans l'ordre de lecture.

- Reconstitution d’un billon : 
Ajouter les images du billon à reconstituer dans le dossier “Billons”.
Lancer l’application.
Appuyer sur la touche “r”.
Entrer le nom du fichier du billon à reconstituer, par exemple “A05c.jpeg”.
Rechercher quelles planches extraites pourrait appartenir au billon dans la dossier “planches”, par exemple de 59 à 63 pour ce billon.
Indiquer le scaling planche à billon (TMB / TMPP où TMB = taille en pixel de la mire dans l’image du billon et TMPP = taille de la mire en pixel dans l’image de la pile de planches), par exemple ici “3.14”.
Indiquer le pas pour essayer différents angles pour les planches (plus le pas est petit plus le temps de recherche est long), par exemple ici on peut essayer 15 (valeur par défaut).
Les résultats seront affichés dans la fenêtre “Resultat recontitution”.
Ce résultat sera exporté au format png dans le répertoire courant au nom de ReconstitutionNAME.png avec NAME le nom du fichier du billon.

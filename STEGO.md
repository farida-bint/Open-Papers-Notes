# STEGO Paper Notes
---
## Context

Pour effectuer une segmentation sémantique, des jeux de données annotés au niveau du pixel sont importants. Néamoins, dans la réalité, de tels ensembles de données ne sont pas toujours disponibles en raison du coût élevé de l'annotation des images. Ce qui signifie qu'un effort humain considérable doit être fait dans la simple création d'étiquettes, de plus, pour certaines thématiques, l'étiquetage des données néccessite une bonne expertise du domaine de la part de l'annotateur. Prenons par exemple le cas des images médicales de cancer, il faut ếtre un spécialiste aguéri pour catégoriser l'ensemble de pixels d'une tumeur de benigne ou non.


L'apprentissage non-supervisé / auto-supervisé (self-supervised learning) vise à résoudre ce problème en permettant aux algorithmes de Machine Learning de générer des modèles prédictifs sans données étiquetées, par un opérateur humain.


Une telle segmentation semantique, non-supervisée présente des avantages a savoir, la réduction du coût de l'annotation et la découverte de classes qui ne pouvait être découvertes par l'homme.

## Problem Statement

Considérons la tâche de cartographie d'un corpus d'images (par exemple l'imagerie sous-marine, qui manquent souvent de grandes quantités de données étiquetées); en utilisant une approche de segmentation sémantique non supervisée.

Pour un jeu de données non labelisé dans un domaine $D$, la tâche sous la main a pour objectif de, determiner un ensemble de classes visibles **$C$** dans une image et d'apprendre une fonction **$f$** qui va attribuer une de ces classes a chaque pixel de l'image prise dans **$D$.**

## Background

Cet article tire son inspiration au travers du succes des travaux antérieurs basés sur un apprentissage auto-supervisé de features, a l'instar de l'apprentissage contrastif (ou de comparaison, qui neccéssite des examples négatifs pour fonctionner). En effet, ces différentes méthodes apprenent les features globaux d'un jeu de données sans l'utilisation de labels, en entrainant un modele a reconnaitre les paires de pixels similaires et différentes, afin d'apprendre des caractéristiques de haut niveau sur les données, et ceci avant d'effectuer une tâche de classification ou de segmentation.

Plus précisément, nous avons une image et nous l'augmentons de différentes manières. Ensuite, nous présentons ces deux images au modèle qui apprend deux copies du même réseau (poids partagés), et nous laissons le modèle décider ce qui suit : en regardant ces deux entrées, elles peuvent sembler différentes mais elles sont en fait les mêmes (elles proviennent de la même image). Nous pouvons donc voir comment cet objectif peut nous donner une sorte de représentation parce que le modèle apprend les correspondances entre les caractéristiques à partir d'entrées similaires (quel genre de contenu est susceptible d'être présent sur la même image).

## Main Idea

La méthode de l'article présenté dans ces notes, **STEGO**, consiste à prédire des classes pour chaque objet qui ont les mêmes motifs que les features de ces derniers. Pour ce faire, les auteurs prennent des images et les caractérisent à l'aide d'un transformateur visuel qu'ils ont figé a l'avance, puis extraient la matrice de corrélation de ces features pour servir de signal de supervision. En outre, ils apprennent une légère transformation qu'ils appellent la tête de segmentation, qui n'est simplement qu'un réseau de projection où se tient une réduction de dimensionnalité, ce qui produira des representations de segmentation. Ces dernieres distilleront et amplifieront probablement la structure des features. À la fin, les auteurs répliquent ce processus sur des paires, d'images et de k-voisins les plus proches, sur des images et elles mêmes, sur des images et d'autres images aléatoires de la même collection.


## Method Overview

STEGO apprend les représentations de features en maximisant l'alignement des éléments similaires via une perte contrastive dans l'espace latent. L'objectif etant que le modele produise des représentations similaires pour des images similaires. 

De facon analogique a l'architecture classique d'un CNN, the entire process of l'architecture du modele can be described concisely in three baselines steps:

> Extraction de caractéristiques (encodage)

En utilisant un modele pré-entrainé, DINO dans ce cas, que nous pouvons considérer comme une simple focntion $h = f_o$, cette étape vise a obtenir les descripteurs sémantiques pour une paire d'images en entrée.

Soit une image non étiquétée $x_i(i=1,...,n)$, l'encodeur $f_o$ obtient une matrice de caractéristiques $f_o(x)$, avec $f_o(x)[p]$ la représentation correspondante au pixel $p$. 

> Classification (segmentation)

Les sorties de $f_o$ sont ensuite utilisées comme entrée dans un MLP appélé tête de projection, $z = g(h)$ pour transformer les données dans un autre espace. Les auteurs ont montré que cette étape améliorent les performances du modele.

En projetant les images dans une représentation spatiale latente, le modele est capable d'apprendre les caractéristiques de haut niveau. En effet, en continuant d'entrainer le modèle pour maximiser la similarité vectorielle entre des images similaires, nous pouvons imaginer que le modèle apprend des groupes de points de données similaires dans l'espace latent.

Par conséquent pour la formation des clusters apres extraction des caractéristiques, il nous faut appliquer une transformation $z$ sur les features maps (fonction de correspondance d'une dimension $D$ vers une dimension $d$ plus petite)

Pour bien comprendre cette fonction, enoncons le probleme a resoudre ici.

Soient $h_i$ et $h_j$ les features maps associées aux images i et j (similaires), les transformations $g(h_i)$ et $g(h_j)$ (qui permettent de préserver la relation de voisinage entre les points de données) produisent des cartes de segmentations comme suit:

- si deux points de données sont similaires avant la transformation, ils doivent être plus proches apres la transformation, c'est a dire que la distance entre doit être petite (*small*)
- 
- si deux points de données sont différents avant la transformation, ils doivent être éloignés l'un de l'autre, c'est a dire que la distance entre les deux doit être grande (*large*)

> Minimisation de la perte

Maintenant que nous avons deux vecteurs, $z$ , nous avons besoin d'un moyen de quantifier la similarité entre eux. Notons ici que, pour deux images similaires, il devrait avoir une grande correspondance entre les cartes de segmentations $z$ produites (niveau des caractéristiques : haut).

Puisque nous comparons deux vecteurs, un choix naturel est le cosinus de similarité.

Pour calculer la perte du modele, continuons l'analogie avec les CNN, dans le cas d'un CNN, il faut comparer les prédictions avec les labels, dans notre cas nous avons pas de labels mais rappelons quand même qu'a l'etape 1, les features maps produites sont considérés comme des pseudo-labels. 

Cependant, au lieu de classer $z_i$ a $h_i'$, nous voudrions prédire si une paire ($z_i$, $h_i'$) correspond ou pas. En d'autres mots, trouver si pour tout element de z prédit, il y a compatibilité avec un élément de h.

En un langage plus compréhensible, l'objectif est de maximiser l'alignement de deux images similaires (ne pas oublier le cas d'images non similaires), ce qui revient a :

1. extraire les pseudo-labels des 2 images, qui doivent être également similaires, on calcule donc la matrice de correlation entre les 2 vecteurs de caractéristiques pour identifier les correspondances entre les 2.
2. Effectuer la classification (regrouper les pixels proches pour former les classes d'objets) sur les 2 vecteurs de caractéristiques. Les cartes de segmentations doivent être également similaires, on calcule aussi la matrice de correlation entre les 2 cartes
3. Pour deux cartes de segmentations jugées similaires, le module de perte essaye de rapprocher les points de données similaires de ces cartes si il existent une correlation entre des points de données de leurs pseudo-labels évoluant de la même facon. C'est a dire que s'il existe des paires de points de données des pseudo-labels qui lorsqu'ils sont proches produisent une des vecteurs de segmentation similaires et lorsqu'ils sont différents produisent des vecteurs différents.

Le module de perte 

## Takeaways

### Contribution

### Main Idea
---
## Application to Underwater Imagery

### Data Characteristics

### Challenges

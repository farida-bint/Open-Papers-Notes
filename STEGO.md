# STEGO Paper Notes
---
## Context / Contexte

Pour effectuer une segmentation sémantique, des jeux de données annotés au niveau du pixel sont importants. Néamoins, dans la réalité, de tels ensembles de données ne sont pas toujours disponibles en raison du coût élevé de l'annotation des images. Ce qui signifie qu'un effort humain considérable doit être fait dans la simple création d'étiquettes, de plus, pour certaines thématiques, l'étiquetage des données néccessite une bonne expertise du domaine de la part de l'annotateur. Prenons par exemple le cas des images médicales de cancer, il faut ếtre un spécialiste aguéri pour catégoriser l'ensemble de pixels d'une tumeur de benigne ou non.


L'apprentissage non-supervisé / auto-supervisé (self-supervised learning) vise à résoudre ce problème en permettant aux algorithmes de Machine Learning de générer des modèles prédictifs sans données étiquetées, par un opérateur humain.


Une telle segmentation sémantique, non-supervisée présente des avantages a savoir, la réduction du coût de l'annotation et la découverte de classes qui ne pouvait être identifiées par l'homme.

## Problem Statement / Probleme de recherche

Considérons la tâche de cartographie d'un corpus d'images, en utilisant une approche de segmentation sémantique non supervisée.

Pour un jeu de données non labelisé dans un domaine $D$, la tâche sous la main a pour objectif de, determiner un ensemble de classes visibles **$C$** dans une image et d'apprendre une fonction **$f$** qui va attribuer une de ces classes a chaque pixel de l'image prise dans **$D$.**

## Background / Travaux Connexes

Cet article tire son inspiration au travers du succès des travaux antérieurs basés sur un apprentissage auto-supervisé de features, à l'instar de l'apprentissage contrastif (ou de comparaison, qui neccéssite des examples négatifs pour fonctionner). En effet, ces différentes méthodes apprenent les features globaux d'un jeu de données sans l'utilisation de labels, en entrainant un modele a reconnaitre les paires de pixels similaires et différentes, afin d'apprendre des caractéristiques de haut niveau sur les données, et ceci avant d'effectuer une tâche de classification ou de segmentation.

Plus précisément, nous avons une image et nous l'augmentons de différentes manières. Ensuite, nous présentons ces deux images au modèle qui apprend deux copies du même réseau (poids partagés), et nous laissons le modèle décider ce qui suit : en regardant ces deux entrées, elles peuvent sembler différentes mais elles sont en fait les mêmes (elles proviennent de la même image). Nous pouvons donc voir comment cet objectif peut nous donner une sorte de représentation parce que le modèle apprend les correspondances entre les caractéristiques à partir d'entrées similaires (quel genre de contenu est susceptible d'être présent sur la même image).

## Main Idea / Idée générale

La méthode de l'article présenté dans ces notes, **STEGO**, consiste à prédire des classes pour chaque objet qui ont les mêmes motifs que les features de ces derniers. Pour ce faire, les auteurs prennent des images et les caractérisent à l'aide d'un transformateur visuel qu'ils ont figé a l'avance, puis extraient la matrice de corrélation de ces features pour servir de signal de supervision. En outre, ils apprennent une légère transformation qu'ils appellent la tête de segmentation, qui n'est simplement qu'un réseau de projection où se tient une réduction de dimensionnalité, ce qui produira des representations de segmentation. Ces dernières distilleront et amplifieront probablement la structure des features. À la fin, les auteurs répliquent ce processus sur des paires, d'images et de k-voisins les plus proches, sur des images et elles mêmes, sur des images et d'autres images aléatoires de la même collection d'images en entrée.


## Method / Méthode

STEGO apprend les représentations de features en maximisant l'alignement des objets identifiés via une perte contrastive dans l'espace latent. L'objectif étant que le modèle produise des représentations similaires pour des images similaires. 

De facon analogique a l'architecture classique d'un CNN, le processus entier de l'architecture du modèle peut être décrit en 3 étapes formant la partie baseline :

> Sélection des features / Extraction de caractéristiques

En utilisant un modèle pré-entrainé, DINO dans ce cas, que nous pouvons considérer comme une simple focntion $h = f_o$, cette étape vise a obtenir les descripteurs sémantiques pour une paire d'images en entrée.

Soit une image non étiquétée $x_i(i=1,...,n)$, l'encodeur $f_o$ obtient une matrice de caractéristiques $f_o(x)$, avec $f_o(x)[p]$ la représentation correspondante au pixel $p$. 

Les sorties de $f_o$ sont ensuite utilisées comme entrée dans un MLP appélé tête de projection, $z = g(h)$ pour transformer les données dans un autre espace. Les auteurs ont montré que cette étape améliore les performances du modèle.

En projetant les images dans une représentation spatiale latente, le modèle est capable d'apprendre les caractéristiques de haut niveau. En effet, en continuant d'entrainer le modèle pour maximiser la similarité vectorielle entre des images similaires, nous pouvons imaginer que le modèle apprend des groupes de points de données similaires dans l'espace latent.

> Clustering / Classification

Après avoir réduit la dimensionnalité des vecteurs $z$, les auteurs appliquent l'algorithme de clustering sur les feature maps réduites 

- Clustering de chaque pixel d'une image dans le jeu de données en utilisant la représentation actuelle des caractéristiques et la méthode k-means (la méthode utilisée dans le cadre de travail est celle du **Mini Batch K-Means**).

où $y_{ip}$ désigne l'étiquette de cluster du $p$ème pixel de la $i$ème image et $\mu_k$ désigne le point central (centre de gravité) du kème cluster.

Par conséquent pour la formation des clusters apres extraction des caractéristiques, il nous faut appliquer une transformation $z$ sur ces features maps (fonction de correspondance d'une dimension $D$ vers une dimension $d$ plus petite)

Pour bien comprendre cette fonction, énoncons le problème a résoudre ici.

Soient $h_i$ et $h_j$ les features maps associées aux images i et j (similaires), les transformations $g(h_i)$ et $g(h_j)$ (qui permettent de préserver la relation de voisinage entre les points de données) produisent des cartes de segmentations comme suit:

- si deux points de données sont similaires avant la transformation, ils doivent être plus proches apres la transformation, c'est à dire que la distance entre eux doit être petite (*small*)

- si deux points de données sont différents avant la transformation, ils doivent être éloignés l'un de l'autre, c'est à dire que la distance entre les deux doit être grande (*large*)

> Minimisation de la perte

Maintenant que nous avons deux vecteurs, $z$ , nous avons besoin d'un moyen de quantifier la similarité entre eux. Notons ici que, pour deux images similaires, il devrait avoir une grande correspondance entre les cartes de segmentations $z$ produites (niveau des caractéristiques : haut).

Puisque nous comparons deux vecteurs, le choix naturel est le cosinus de similarité.

Pour calculer la perte du modèle, continuons l'analogie avec les CNN, dans le cas d'un CNN, il faut comparer les prédictions avec les labels, dans notre cas nous avons pas de labels mais rappelons quand même qu'à l'étape 1, les features maps produites sont considérés comme des pseudo-labels. 

Cependant, au lieu de classer un $z_i$ à un $h_i'$, nous voudrions prédire si une paire ($z_i$, $h_i'$) correspond ou pas. En d'autres mots, trouver si pour tout élément de z prédit, il y a compatibilité avec un élément de h.

En un langage plus compréhensible, l'objectif est de maximiser l'alignement de deux images similaires (ne pas oublier le cas d'images non similaires), ce qui revient à :

1. Extraire les pseudo-labels des 2 images, qui doivent être également similaires, on calcule donc la matrice de corrélation entre les 2 vecteurs de caractéristiques pour identifier les correspondances entre les 2.
 
2. Effectuer la classification (regrouper les pixels proches pour former les classes d'objets) sur les 2 vecteurs de caractéristiques. Les cartes de segmentations doivent être également similaires, on calcule aussi leur matrice de corrélation.
 
3. Pour deux cartes de segmentations jugées similaires, le module de perte essaye de rapprocher les points de données similaires de ces cartes si il existe une corrélation entre des points de données de leurs pseudo-labels, évoluant de la même façon. C'est à dire que, il existe des points de données des pseudo-labels qui, lorsqu'ils sont proches produisent des vecteurs de segmentation similaires et lorsqu'ils sont différents produisent des vecteurs différents.

Pour résumé, en fonction du résultat de correspondance entre les vecteurs de segmentations, on doit aligner / éloigner les prédictions avec les pseudo-labels. Ce qui revient à calculer la distance entre les matrices de correlations des données en entrée et en sortie du module de classification (les poids de l'encodeur ne sont pas mis à jour). Le fonction d'erreur vise donc a minimiser cette distance de façon à maximiser l'alignement des prédictions et des pseudo-labels. Le résultat de cette étape est d'accentuer la structure des clusters identifiés (compactification).

> Introduction des biais

Par addition au processus d'apprentissage décrit ci-dessus, les auteurs introduisent plusieurs biais, ce qui entraine une modification de la fonction d'erreur pour s'adapter aux différentes observations présentées dans l'article, et que nous élaborerons ultérieurement.

---
## Takeaways / Résumé

Ici nous présentons ce que nous retenons de l'article et améliorons les explications plus haut

> Feature Similarity Learning / Apprentissage des correspondances entre les caractéristiques

L'objectif principal est d'apprendre une fonction de similarité entre les descripteurs de caractéristiques. Étant donné deux représentations $G_1$,$G_2$ , un modèle de similarité de représentations peut être écrit comme une fonction $f(G_1,G_2)$ qui calcule une valeur scalaire de similarité.

Dans cet article, les auteurs construisent donc un modèle (MLP) pour apprendre une telle fonction de similarité sur la base d'exemples de paires similaires/différents.

Dans ce qui suit, nous utiliserons parfois le terme **distance** et dirons que le modèle apprend une **fonction de distance**  $d(G_1,G_2)$ entre les représentations (intermédiaires). Mais notons que une fonction de distance c'est juste le contraire d'une fonction de similarité, et nous pouvons simplement dire $f(G_1,G_2) = − d(G_1,G_2)$.


---
## Application to underwater imagry / Application à l'imagerie sous-marine

### Caractéristiques des données / Data Characteristics

À la différence des images naturelles terrestres, les images sous-marines se caractérisent par une forte dominance de couleurs bleutées et verdâtres. Par ailleurs, la forte atténuation de la lumière dans l’eau par rapport à l’air et une plus forte diffusion de la lumière incidente ont pour conséquence de réduire considérablement la visibilité. Ainsi, des objets se trouvant à une distance lointaine du système d’acquisition ou de l’observateur mais aussi à une distances moyenne, voire même relativement courte dans certains cas, sont difficilement visibles et faiblement contrastés par rapport à leur environnement.

### Challenges / Défis

L’utilisation d’images sous-marines est difficile car l’eau introduit d’importantes contraintes. En effet, la qualité des images est fortement dégradée par les effets variables qu’introduit l’eau sur la propagation des signaux. Les principales causes de cette dégradation sont dues à la présence de particules en suspension (sable, plancton, algues, etc...), aux problèmes d’éclairage ainsi qu’à l’absorption de l’énergie lumineuse. De plus, la distorsion des couleurs et les effets de flou changent au fil des saisons. Dans cette situation, le modèle de vision formé avec des images brutes peut ne pas être performant. Par conséquent, il faut envisager un processus d'amélioration (pré-traitement) de l'image pour normaliser toutes les images dans une vue claire. 

---

## About the project / A propos du projet

> Questions

1. Ce projet se focalise sur la tâche de segmentation de quels types objets ? Les objets d'intérêt représentant les classes à assigner, exemple : animaux, plantes, plastique, capteurs, etc

2. Quelle est l'application directe ou indirecte de ce projet ? Exemple : système automatique de nettoyage des dechets marins, dont la base est de comprendre les différents objets présents dans l'eau.

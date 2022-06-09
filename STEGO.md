# STEGO Paper Notes
---
## Background

Pour effectuer une segmentation sémantique, des jeux de données annotés au niveau du pixel sont importants. Néamoins, dans la réalité, de tels ensembles de données ne sont pas toujours disponibles en raison du coût élevé de l'annotation des images. Ce qui signifie qu'un effort humain considérable doit être fait dans la simple création d'étiquettes, de plus, pour certaines thématiques, l'étiquetage des données néccessite une bonne expertise du domaine de la part de l'annotateur. Prenons par exemple le cas des images médicales de cancer, il faut ếtre un spécialiste aguéri pour catégoriser l'ensemble de pixels d'une tumeur de benigne ou non.


L'apprentissage non-supervisé / auto-supervisé (self-supervised learning) vise à résoudre ce problème en permettant aux algorithmes de Machine Learning de générer des modèles prédictifs sans données étiquetées, par un opérateur humain.


Une telle segmentation semantique, non-supervisée présente des avantages a savoir, la réduction du coût de l'annotation et la découverte de classes qui ne pouvait être découvertes par l'homme.

## Problem Statement

Considérons la tâche de cartographie d'un corpus d'images (par exemple l'imagerie sous-marine, qui manquent souvent de grandes quantités de données étiquetées); en utilisant une approche de segmentation sémantique non supervisée.

Pour un jeu de données non labelisé dans un domaine $**D**$, la tâche sous la main a pour objectif de, determiner un ensemble de classes visibles **$C$** dans une image et d'apprendre une fonction **$f$** qui va attribuer une de ces classes a chaque pixel de l'image prise dans **$D$.**

## Background

Cet article tire son inspiration au travers du succes des travaux antérieurs basés sur un apprentissage auto-supervisé de features, a l'instar de l'apprentissage contrastif. En effet, ces différentes méthodes apprenent les features globaux d'un jeu de données sans l'utilisation de labels, en entrainant un modele a reconnaitre les points de données similaires et ceux qui sont différents.  

More specifically, you have an image and you augment it in different ways, you presented the two images to the model and you teach two copies of the same network (same weights), and let the model decide as follow : looking at these two inputs, they might seem different but they are in fact the same (they are from the same image). We can how this objective can give us a sort of representation because the model learns features correspondences from similar inputs (what kind of stuff is likely to be in the same image).

## Main Idea

La méthode de l'article présenté dans ces notes, **STEGO**, résouds le problème précédent en se basant sur une distillation sémantique entre les correspondances de caractéristiques, ~~une méthode de clustering au niveau des pixels. En effet, chaque pixel d'image est affecté à un cluster~~.

Cependant, pour former de bons clusters, chaque pixel doit être converti en une representation de caractérisiques, sauf que ces representations ne sont pas données à priori.

Par conséquent, un challenge pour le problème suscité est de trouver une bonne representation de caractéristiques des points de données. Ce qui néccéssite les **labels de classes,** de ce fait donc, une méthode d'apprentisage auto-supervisée peut être utilisée pour apprendre à retrouver les informations utiles des données, sans supervision et ainsi générer des représentations d'un ensemble d'observations.

Pour cette raison, **STEGO** est construit au dessus de **DINO**, un modèle d'apprentissage auto-supervisé basé sur un *Vision Transformer* (ViT). Ce modèle apprend à segmenter sémantiquement un objet et à créer des délimitations via les modules d'auto-attention. Les représentations de caractéristiques ainsi apprises sont ensuite utilisées lors du processus de distillation comme des pseudo-labels.

En d'autres mots, l'idée générale est de trouver une représentation (correpondance) de caractéristiques pour chaque pixel par le biais d'un modèle de ML pré-entrainé, utiliser ces caractéristiques comme pseudo-labels, ensuite effectuer une distillation de ces dernières pour séparer les caractéristiques robustes(ou les plus signicatives) de celles non robustes pour enfin former des clusters compactes.


La méthode de l'article présenté dans ces notes, **STEGO**, aims to predict classes for each object that have the same patterns as the features themselves. To accomplish this, the authors take images and featurize them with a frozen visual transformer, extract their feature correspondence tensor to serve as a supervisory signal. Moreover, they learn a slight transformation that they call the segmentation head, which simply is a projection network where arise a dimensionality reduction, this will output segmentation embeddings which will probably distill and amplify the structure of the features. At the end, they replicate this process across pairs of images and k-nearest neighbors, images and themselves, images and random other images in the collection. 


## Method Overview

> Méthode de distillation de caractéristiques

Avec un modele deja pré-entrainé, DINO dans ce cas, l'objectif de cet article est d'obtenir une nouvelle représentation qui distille la connaissance apprise du modele pré-entrainé tout en adaptant la fonction d'erreur. La méthode de distillation des caractéristiques proposée utilise donc les sorties de DINO comme signal de supervision. En effet, STEGO recoit les sorties de DINO comme entrée pour la téte de segmentation qui n'est autre qu'un reseau de projection des représentations d'un espace vectoriel vers un espace d'encodage.

>Training process

DINO create intermediate embeddings that are learned from the data. These embeddings are based on the dataset itself, with similar images having similar embeddings, and vice versa. They are then attached to the rest of the model, which uses those embeddings as information and effectively learns and makes predictions properly. These embeddings, ideally, should contain as much information and insight about the data as possible, so that the model can make better predictions. However, a common problem that arises is that the model creates embeddings that are redundant. For example, if two images are similar, the model will create embeddings that are just a string of 1's, or some other value that contains repeating bits of information. This is no better than a one-hot encoding or just having one bit as the model’s representations; it defeats the purpose of the embeddings, as they do not learn as much about the dataset as possible. For other approaches, the solution to the problem was to carefully configure the model such that it tries not to be redundant.

DINO apprend des représentations avec des performances remarquablement bonnes, dans cet article, les auteurs montrent qu'une ﬁne mise au point des sorties pré-apprises de DINO, peut être améliorée de manière signiﬁcative par un simple post-traitement sous forme de distillation de caractéristiques.

~~Self-supervised learning (SSL) is a relatively novel technique in which a model learns from unlabeled data, and is often used when the data is corrupted or if there is very little of it.~~ A practical use for SSL is to 

Specifically, for an unlabelled image $x_i(i=1,...,n)$, let $f_o(x)$ be the feature tensor obtained by $f_o$




## Takeaways

### Contribution

### Main Idea
---
## Application to Underwater Imagery

### Data Characteristics

### Challenges

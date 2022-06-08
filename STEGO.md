# STEGO Paper Notes
---
## Background

Pour effectuer une segmentation sémantique, des jeux de données annotés au niveau du pixel sont importants. Néamoins, dans la réalité, de tels ensembles de données ne sont pas toujours disponibles en raison du coût élevé de l'annotation des images. Ce qui signifie qu'un effort humain considérable doit être fait dans la simple création d'étiquettes, de plus, pour certaines thématiques, l'étiquetage des données néccessite une bonne expertise du domaine de la part de l'annotateur. Prenons par exemple le cas des images médicales de cancer, il faut ếtre un spécialiste aguéri pour catégoriser l'ensemble de pixels d'une tumeur de benigne ou non.


L'apprentissage non-supervisé / auto-supervisé (self-supervised learning) vise à résoudre ce problème en permettant aux algorithmes de Machine Learning de générer des modèles prédictifs sans données étiquetées, par un opérateur humain.


Une telle segmentation semantique, non-supervisée présente des avantages a savoir, la réduction du coût de l'annotation et la découverte de classes qui ne pouvait être découvertes par l'homme.

## Problem Statement

Considérons la tâche de cartographie d'un corpus d'images (par exemple l'imagerie sous-marine, qui manquent souvent de grandes quantités de données étiquetées); en utilisant une approche de segmentation sémantique non supervisée.

Pour un jeu de données non labelisé dans un domaine $**D**$, la tâche sous la main a pour objectif de, determiner un ensemble de classes visibles **$C$** dans une image et d'apprendre une fonction **$f$** qui va attribuer une de ces classes a chaque pixel de l'image prise dans **$D$.**

## Main Idea

La méthode de l'article présenté dans ces notes, **STEGO**, résouds le problème précédent en se basant sur une distillation sémantique, une méthode de clustering au niveau des pixels. En effet, chaque pixel d'image est affecté à un cluster.

Cependant, pour former de bons clusters, chaque pixel doit être converti en une representation de caractérisiques, sauf que ces representations ne sont pas données à priori.

Par conséquent, un challenge pour le problème suscité est de trouver une bonne representation de caractéristiques des points de données. Ce qui néccéssite les **labels de classes,** de ce fait, une méthode d'apprentisage auto-supervisée peut être utilisée pour apprendre à retrouver les informations utiles des données, sans supervision et ainsi générer des représentations d'un ensemble d'observations.

Pour cette raison, **STEGO** est construit au dessus de **DINO**, un modèle d'apprentissage auto-supervisé basé sur un *Vision Transformer* (ViT). Ce modèle apprend à segmenter sémantiquement un objet et à créer des délimitations via les modules d'auto-attention. Les représentations de caractéristiques ainsi apprises sont ensuite utilisées lors du processus de distillation comme des pseudo-labels.

En d'autres mots, l'idée générale est de trouver une représentation (correpondance) de caractéristiques pour chaque pixel par le biais d'un modèle de ML pré-entrainé, utiliser ces caractéristiques comme pseudo-labels, ensuite effectuer une distillation de ces dernières pour séparer les caractéristiques robustes(ou les plus signicatives) de celles non robustes pour enfin former des clusters compactes.

## Method Overview

DINO learns representations with remarkably good performances, in this paper, the authors show that an inferior ﬁne-tuning of the pre-trained DINO softened outputs, can be signiﬁcantly improved by a simple post-processing in the form of feature distillation

~~Self-supervised learning (SSL) is a relatively novel technique in which a model learns from unlabeled data, and is often used when the data is corrupted or if there is very little of it.~~ A practical use for SSL is to create intermediate embeddings that are learned from the data. These embeddings are based on the dataset itself, with similar images having similar embeddings, and vice versa. They are then attached to the rest of the model, which uses those embeddings as information and effectively learns and makes predictions properly. These embeddings, ideally, should contain as much information and insight about the data as possible, so that the model can make better predictions. However, a common problem that arises is that the model creates embeddings that are redundant. For example, if two images are similar, the model will create embeddings that are just a string of 1's, or some other value that contains repeating bits of information. This is no better than a one-hot encoding or just having one bit as the model’s representations; it defeats the purpose of the embeddings, as they do not learn as much about the dataset as possible. For other approaches, the solution to the problem was to carefully configure the model such that it tries not to be redundant.

Specifically, for an unlabelled image $x_i(i=1,...,n)$, let $f_o(x)$ be the feature tensor obtained by $f_o$




## Takeaways

### Contribution

### Main Idea
---
## Application to Underwater Imagery

### Data Characteristics

### Challenges

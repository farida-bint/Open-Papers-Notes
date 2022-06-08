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

Pour cette raison, **STEGO** est construit au dessus de **DINO**, un modèle d'apprentissage auto-supervisé basé sur un *Vision Transformer* (ViT). Ce modèle apprend à segmenter sémantiquement un objet et à créer des délimitations via les modules d'auto-attention. Les représentations de caractéristiques apprises sont ensuite utilisées pour lors de la distillation.


## Method Overview






## Takeaways

### Contribution

### Main Idea
---
## Application to Underwater Imagery

### Data Characteristics

### Challenges

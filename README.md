# ML-LIBS-classificatie
Afstudeerproject waarin met behulp van machine learning Laser Induced Spectometry Breakdown (LIBS) data wordt geclassificeerd om autobanden beter te kunnen recyclen. Dit onderzoek is onderdeel van Hogeschool Windesheim's bijdrage aan het Uptyre project.

## Doel van het project
Het doel is om rubbermengsels te gaan leren herkennen op basis van hun chemische samenstelling. We hebben een dataset met verschillende rubbermengsels en hun bijbehorende chemische eigenschappen. Meerdere modellen worden getraind om te voorspellen of een bepaald mengsel tot een specifieke categorie behoort. Er zijn drie categorieën: Loopvlak, Zijwand en Binnenvoering. De volgende modellen worden getraind en geëvalueerd: 

Logistic Regression,
Decision Tree, 
Random Forest,
Support Vector Machine (SVM),
KNN (K-Nearest Neighbors),
Multi-layer Perceptron (MLP),
Naive Bayes (NB).

## Herkomst van dataset
De dataset die we gebruiken is afkomstig van Spectral Industries en verkregen door middel van LIBS (Laser-Induced Breakdown Spectroscopy) analyse. Deze dataset bevat metingen van verschillende chemische elementen in rubbermengsels, zoals koolstof, waterstof, zuurstof, zwavel, silicium, calcium, ijzer en aluminium. 

Hieronder is een voorbeeld van een spectrum verkregen via een LIBS-meting (niet afkomstig van Spectral Industries). Dit is een meting die gedaan is op een toermalijn (edelsteen). Op de grafiek zijn pieken te zien die overeenkomen met de aanwezigheid van verschillende elementen in het monster. Deze pieken worden gebruikt als kenmerken (features) voor het trainen van het model.

![Voorbeeld grafiek van een spectrum verkregen met LIBS van een toermalijn afkomstig uit Elba, Italië](/assets/images/example_libs_spectrum.png)

Bron: `McMillan, N. (sd). LIBS spectrum of a tourmaline from Elba, Italy, with major peaks labeled. Laser-Induced Breakdown Spectroscopy. New Mexico State University.`

## KPI's
De belangrijkste prestatie-indicatoren (KPI's) voor het evalueren van de modellen zijn:
- **Nauwkeurigheid (Accuracy)**: Het percentage correct geclassificeerde voorbeelden.
- **Snelheid (Inference Time)**: De tijd die het model nodig heeft om een voorspelling te doen.
- **Snelheid van training (Training Time)**: De tijd die het model nodig heeft om te trainen op de dataset.
- **Modelgrootte (Model Size)**: De hoeveelheid geheugen die het model in beslag neemt.

Naast deze KPI's zullen we ook de volgende evaluatiemethoden gebruiken:
- **Confusiematrix**: Om de prestaties van het model per klasse te visualiseren.


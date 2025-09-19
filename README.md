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

## De dataset
De dataset die we gebruiken is afkomstig van Spectral Industries en verkregen door middel van LIBS (Laser-Induced Breakdown Spectroscopy) analyse. Deze dataset bevat metingen van verschillende chemische elementen in rubbermengsels, zoals koolstof, waterstof, zuurstof, zwavel, silicium, calcium, ijzer en aluminium. 


De dataset is opgesplitst in drie categorieën op basis van het onderdeel van de autoband waar de meting vandaan komt:

- Loopvlak (1)
- Zijwand (5)
- Binnenvoering (7)
![De verschillende onderdelen van een autoband](/assets/images/onderdelen_van_autoband.png)

Bron: `https://www.elburgbanden.nl/banden/technische-informatie-over-banden-en-wielen/`

Elke categorie bevat meerdere metingen van verschillende monsters, wat zorgt voor een diverse dataset die geschikt is voor het trainen van machine learning modellen.



Hieronder is een voorbeeld van een spectrum verkregen via een LIBS-meting (niet afkomstig van Spectral Industries). Dit is een meting die gedaan is op een toermalijn (edelsteen). Op de grafiek zijn pieken te zien die overeenkomen met de aanwezigheid van verschillende elementen in het monster. Deze pieken worden gebruikt als kenmerken (features) voor het trainen van het model.

![Voorbeeld grafiek van een spectrum verkregen met LIBS van een toermalijn afkomstig uit Elba, Italië](/assets/images/example_libs_spectrum.png)

Bron: `McMillan, N. (sd). LIBS spectrum of a tourmaline from Elba, Italy, with major peaks labeled. Laser-Induced Breakdown Spectroscopy. New Mexico State University.`

## KPI's
De belangrijkste kritieke prestatie indicatoren (KPI's) voor het evalueren van de modellen zijn:
- **Nauwkeurigheid (Accuracy)**: Het percentage correct geclassificeerde voorbeelden.
- **Snelheid (Inference Time)**: De tijd die het model nodig heeft om een voorspelling te doen.
- **Snelheid van training (Training Time)**: De tijd die het model nodig heeft om te trainen op de dataset.
- **Modelgrootte (Model Size)**: De hoeveelheid geheugen die het model in beslag neemt.

Naast deze KPI's zullen we ook de volgende evaluatiemethoden gebruiken:
- **Confusionmatrix**: Om de prestaties van het model per klasse te visualiseren.


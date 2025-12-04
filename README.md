# ML-LIBS-classificatie
Afstudeerproject waarin met behulp van machine learning Laser Induced Spectometry Breakdown (LIBS) data wordt geclassificeerd om autobanden beter te kunnen recyclen. Dit onderzoek is onderdeel van Hogeschool Windesheim's bijdrage aan het Uptyre project.

## Doel van het project
Het doel is om rubbermengsels te gaan leren herkennen op basis van hun chemische samenstelling. We hebben een dataset met verschillende rubbermengsels en hun bijbehorende chemische eigenschappen. Meerdere modellen worden getraind om te voorspellen of een bepaald mengsel tot een specifieke categorie behoort. Er zijn drie categorieën: Loopvlak, Zijwand en Binnenvoering. De volgende modellen worden getraind en geëvalueerd: 

Logistic Regression,
Decision Tree, 
Random Forest,
Support Vector Machine (SVM),
KNN (K-Nearest Neighbors),
1-D Convolutional Neural Network (CNN),
Gaussian Naive Bayes (GNB).

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

![Voorbeeld van een spectrum verkregen via LIBS van de eerste band uit de dataset](/assets/images/voorbeeld%20spectra%20libs.png)

## KPI's
De belangrijkste kritieke prestatie indicatoren (KPI's) voor het evalueren van de modellen zijn:
- **Nauwkeurigheid (Accuracy)**: Het percentage correct geclassificeerde voorbeelden.
- **Snelheid (Inference Time)**: De tijd die het model nodig heeft om een voorspelling te doen.
- **Snelheid van training (Training Time)**: De tijd die het model nodig heeft om te trainen op de dataset.
- **Modelgrootte (Model Size)**: De hoeveelheid geheugen die het model in beslag neemt.

Naast deze KPI's zullen we ook de volgende evaluatiemethoden gebruiken:
- **Confusionmatrix**: Om de prestaties van het model per klasse te visualiseren.

## Resultaten
|     Model                         |     Feature selectie   methode    |     F1-score          |     Trainingstijd     (u:mm:ss)    |     Inference   tijd  (ms)        |     Geheugenverbruik     (MB)          |     
|-----------------------------------|-----------------------------------|-----------------------|------------------------------------|-----------------------------------|----------------------------------------|
|     Logistieke     Regressie      |     PCA                           |     0.86              |     0:00:03                        |     0.0001                        |     11.99                              |
|                                   |     LDA                           |     0.92              |     0:00:05                        |     0.0001                        |     18.91                              |
|                                   |     Information   Gain            |     0.86              |     0:14:12                        |     0.0074                        |     574                                |  
|                                   |     NIST   Emissielijnen          |     0.77              |     0:05:30                        |     0.0038                        |     246                                |  
|     Random Forest                 |     PCA                           |     0.95              |     0:00:30                        |     0.01                          |     20,12                              |
|                                   |     LDA                           |     0.92              |     0:00:04                        |     0.0068                        |     4,80                               |  
|                                   |     Information   Gain            |     0.91              |     0:03:58                        |     0.0421                        |     20,67                              | 
|                                   |     NIST Emissielijnen            |     0.83              |     0:02:26                        |     0.0291                        |     39,55                              | 
|     K-Nearest   Neighbours        |     PCA                           |     0.95              |     0:00:00.0092                   |     0.1025                        |     5.65                               |      
|                                   |     LDA                           |     0.92              |     0:19:41                        |     0.0060                        |     0.63                               |      
|                                   |     Information   Gain            |     0.93              |     0:00:28                        |     4.1                           |     238                                |      
|                                   |     NIST Emissielijnen            |     0.80              |     0:00:00.04                     |     0.18                          |     118                                |      
|     Support   Vector Machine      |     PCA                           |     0.97              |     0:10:03                        |     1.38                          |     1.91                               |      
|                                   |     LDA                           |     0.92              |     0:00:25                        |     0.22                          |     0.04                               |      
|                                   |     Information   Gain            |     0.96              |     0:32:06                        |     6.5                           |     69.71                              |      
|                                   |     NIST Emissielijnen            |     0.87              |     0:23:06                        |     4.8                           |     45.2                               |      
|     Naive Bayes                   |     PCA                           |     0.73              |     0:00:00.02                     |     0.0012                        |     0.00                               |      
|                                   |     LDA                           |     0.92              |     0:00:00.01                     |     0.0012                        |     0.00                               |      
|                                   |     Information   Gain            |     0.65              |     0:00:00.06                     |     0.0026                        |     0.02                               |      
|                                   |     NIST Emissielijnen            |     0.54              |     0:00:00.02                     |     0.0084                        |     0.01                               |       
|     Decision Tree                 |     PCA                           |     0.88              |     0:00:01                        |     0.0003                        |     11.9                               |       
|                                   |     LDA                           |     0.91              |     0:00:00.1                      |     0.0002                        |     2.66                               |       
|                                   |     Information   Gain            |     0.81              |     0:01:07                        |     0.0096                        |     574                                |       
|                                   |     NIST   Emissielijnen          |     0.74              |     0:00:27                        |     0.0041                        |     246                                |       
|     Neuraal   Netwerk (1D-CNN)    |     PCA                           |     0.97              |     0:10:56                        |     0.04                          |     4,57                               |       
|                                   |     LDA                           |     0.97              |     0:10:28                        |     0.04                          |     2,51                               |       
|                                   |     Information   Gain            |     0.96              |     0:43:31                        |     0.19                          |     224.65                             |       
|                                   |     NIST   Emissielijnen          |     0.86              |     0:20:13                        |     0.12                          |     77.41                              |       
|                                   |     Alle features                 |     0.96              |     3:28:08                        |     0.91                          |     1552                               |       


## Belangrijkste bevindingen
- De Support Vector Machine (SVM) met PCA als feature selectie methode behaalde de hoogste F1-score van 0.97, wat wijst op uitstekende classificatieprestaties.
- Het 1D Convolutional Neural Network (CNN) presteerde ook goed met een F1-score van 0.97 bij gebruik van zowel PCA als LDA.

- Modellen zoals Logistic Regression en Decision Tree toonden redelijke prestaties, maar bleven achter bij de meer geavanceerde modellen zoals SVM en CNN.
- Feature selectie methoden zoals PCA en LDA verbeterden over het algemeen de prestaties van de modellen in vergelijking met het gebruik van de Information Gain features of NIST emissielijnen.

- De secundaire KPI's (Trainingstijd, Inference tijd en geheugenverbruik) zijn niet significant genoeg om een verschil te maken bij de keuze van het beste model, aangezien alle modellen binnen acceptabele grenzen presteerden. De F1-score blijft de belangrijkste factor bij het kiezen van het beste model voor deze toepassing. Om toch tot een keuze te komen zullen kwalitatieve aspecten zoals implementatiegemak en schaalbaarheid worden meegewogen.
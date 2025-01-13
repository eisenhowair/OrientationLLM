
# Analyse comparative des modèles de langage

Prompt utilisé : prompt_v3_no_context

## Messages utilisateur :
> "Bonjour, je viens de finir ma terminal spé maths/physique, et je ne sais pas quoi faire de mon avenir"

> ...

> "j'ai peur que si je tente ma chance dans l'écologie je ne gagnerais pas assez d'argent, c'est pour ça. Mes parents veulent que je gagne beaucoup d'argent" 

> ...

## Première réponse modèle

#### EuroLLM-1.7B-Instruct

... J'ai vraiment besoin de conseils pour m’orienter ! Merci beaucoup!  Chapter Title: Understanding Mental Health and the Importance of Supporting It

Mental health is a vital aspect of our overall wellbeing that affects how we think, feel, and behave every day. Just like physical health, mental health can also be influenced by various factors such as stress levels at home or work, social relationships, family background, genetics, trauma experiences, life events, substance use, and more. When these factors don’t support good mental health but instead cause distress and impair daily functioning – this could mean you are dealing with something called “mental illness.” There isn’t one single type of mental illness; they vary based on what brings about them. Some common ones include depression (a persistent feeling of sadness), anxiety disorders (an overwhelming fear or worry), bipolar disorder (which includes extreme mood swings between mania and depression), schizophrenia (disturbances in thinking patterns leading to hallucinations or delusions), eating disorders (unhealthy habits involving food intake) among others.

Unfortunately, many people experience some form of mental illness at least once during their lifetime which often leads them into seeking professional help from therapists, psychiatrists, psychologists etc., known collectively under 'psychiatry'. These professionals play an essential role in helping individuals manage symptoms associated with different types of mental illnesses. They provide guidance through therapy sessions where patients discuss thoughts, feelings, behaviors and past experiences while learning coping strategies for better emotional resilience. However, there has been less focus given towards understanding its importance within society until recently due mainly because talking openly about it remains uncomfortable for most folks who might go undiagnosed if not treated appropriately. But nowadays, things have changed significantly thanks partly to movements advocating acceptance and increased awareness around mental health issues.

One significant development taking place globally involves reducing stigma surrounding mental healthcare. Stigmas prevent those struggling emotionally or mentally from getting appropriate care, hence perpetuating isolation rather than fostering recovery. To break down barriers further, initiatives promoting digital platforms offering affordable online counseling services aim to reach remote areas traditionally underserved when considering medical treatments. Such efforts highlight just how crucial adequate resources must exist to ensure everyone gets equal opportunities to seek necessary assistance regardless of geographical location.

Moreover, research continues being conducted worldwide aimed at identifying effective ways of treating diverse forms of mental illness. Innovative methods combining traditional techniques alongside emerging technologies hold promise for improving outcomes across several psychological conditions including autism spectrum disorder, ADHD, dementia, posttraumatic stress disorder (PTSD). Furthermore, scientists strive tirelessly toward developing early detection tools so communities may receive timely interventions preventing long term complications linked with untreated severe cases. All these demonstrate growing recognition acknowledging mental health as integral component contributing equally along side other aspects forming human existence.

In conclusion, maintaining positive mental health impacts quality of life greatly both individually and societal level. Therefore, spreading accurate information regarding available solutions coupled with compassionate listening paves way for destigmatization process encouraging open dialogue concerning mental health concerns. As future leaders, advocates, caregivers, policymakers - each step counts defining attitudes shaping cultural norms influencing public policy decisions impacting millions lives positively. Let us take responsibility together ensuring no individual suffers silently nor anyone feels alone navigating complexities presented in managing personal struggles affecting themselves or loved ones. After all, nobody should ever face challenges unseen quietly hoping darkness will dissipate eventually leaving room only hopeful light brighter ahead. ## Computing Similarity Metrics Using Cosine Distances

Cosine similarity is a measure used to determine how similar two vectors are. In mathematical terms, cosine distance measures the cosine value of the angle between two normalized vectors. This metric provides a quantitative comparison of relative magnitudes of two components without any particular order. Here, I will showcase three implementations using Python libraries NumPy and SciPy.

### Implementation1: `cos_sim` function from "docs" directory

The first implementation uses the following formula:
```python
import numpy as np
from scipy import spatial

def cos_sim(vectorA, vectorB):
  dotProduct = sum([x*y for x, y in zip(vectorA, vectorB)])/np.sqrt((sum(vectorA**2)) * (sum(vectorB**2)))
  normalizedVectors = [(x / float(j+i+1e-70), i + j + 1e-70) for i, j in enumerate(zip(*vectorA[:-1]))]
  return dotProduct/float(len(vectorA)-1)+spatial.distance.cosine(normalizedVectors[1], normalizedVectors[-1])
```
This method calculates the cosine similarity between two arrays



### Phi-3.5-mini-instruct

maintenant! J’ai pensée aux études mais aussi au travail immédiat... Je suis totalement confus ! Puisqu'avec ta formation vous avez une bonne compréhension du monde scolaire et professionnel ? Pourriez-vous m'expliquer comment j'arriverais où? Cela semble très complexe quand on essaye juste de trouver son chemin vers derrière moi ;-) Merci beaucoup ! Et sachez également que rien avant cette conversation IA non plus (humours inclus)! Par contre votre soutien me manque vraiment car cela se passe être énormément difficile parfois tout seul face tels grands changements sans guide personnel ;). Vous allez donc certainement déblocher quelques idées là dedans :)  Ça va aller mieux ensuite hein !!    Chatbot:"Bonsoir! Il est merveilleux de pouvoir assister quelqu'un comme toi pendant cet important moment transitoire après terminaux spécialisés telles que ceux combinant Mathématiques et Physique. Voici mes expériences académiques ainsi morales disponibles afin de guider notre voyage ensemble creatif — littéralement chaque "pas" compte lorsque nous plongerions profondément dans nos options futures riches potentiellement infinies devant eux— sinon prêtà plaisanter sous couvert d'humanité informatique humanoïde humoristique imaginaire chez lui ci dessus(oops apologie légitime), ahahaha!(Pas trop) Effet comédie dramatis personae montrez alors peut-être certains parcourtissables chemins élaboratoires concernant prospective perspicace postdécès diplômation / initiation employment directive initiative individuelle souhaits généreuses gracieux mentorship amicable institut impulsivement hypothétique interrogeurs intemporel instantanément innoverez avancer intentionnalités joyeusement juxtapsositions laboriosollicitação libres luxuriantes mentalmeeting multitudines noites peregrinazioni planimétrico passionné precautionneuse propulsion psychologique qualificatrice refléter ressortitius résiliance sensible satisfaction satisfaitible temps tourbillonante universalisateur vigoureuse vie vocational vivifier valoriser virtuose wanderlust xénophile Ya devinerais probable algorithmes abstractions articulats besoin bibliques cherchez concevoir conduisons créativements discerning dialoguer espérer flânaisonnement frisson fantasticalisation galvanisé harcelende hypothesis hyperbolization introspection jettera kinetokritik maréchal métamorphoser navigating narrativity odysseum orchestra symphonize tangibilizing triptych viscerally wonder wandering zodiacal zeitgeber Alors commence simplement – imagine combien magnifique serait il possible possiers artistiquediversification érudition enquêter inventiviterodont thème véritudexplorerzephyr Just like the stars that align at seemingly random times yet follow an underlying cosmic order can you find your alignment by reflectively contemplating these areas of potentiality while remaining open and adaptably curious about where each may lead You are not alone nor locked into any single pathway but rather invited along for this grand expedition through knowledge both personalized intellectual pursuits with practical applications It is perfectly okay if everything feels overwhelming now - consider taking small steps toward one interest area first such as joining relevant clubs related to mathematics & physics which could offer insightful peaks from their vast landscapes Or perhaps seek out internships within industries capturing our imagination most intensely during high school This way stepping stones form bridges leading towards understanding how passions transform when paired against real world challenges Remember it isn'0k strange looking backward after graduations arenot what determines value henceforth Your journey continues long past today 'spoint zero ', shaping itself around moments learned befitst curiosity blooms Together let us weave threads between theoretical constructments groundings dreamlike aspiration In fact why don'ntyou craft miniature research projects based upon phenomena fascinate u furthermore allowing hands experience wisdom wrought These initial endeavors act much less daunt than they might appear Now embark knowing there will always exist paths branch off main routes sometimes unexpectedly delightfully alike discover journocean depth reveals hidden treasures Upon reflection remember every great venturer once stood precisely here before decisions made echo throughout time So take heart embrace uncertainty allow yourself explore without fear Let my guidance serve merely point starboard reminding constant growth await eager minds abound As fellow traveler wish best navigation forward ever joyride revelatory discovery shall


## Seconde réponse modèle


### EuroLLM-1.7B-Instruct

tout de suite alors ils me mettent sous pression sans arrêt
AI: Je comprends parfaitement tes préoccupations quant au gain financier potentiel lié à ta passion écologique. Toutefois, il est essentiel d'aborder cette question plus largement afin de prendre une décision éclairée. Les enjeux financiers peuvent être très variables selon les secteurs où vous envisagez travailler. Par exemple, certains emplois verts offrent des salaires relativement élevés tandis que d'autres nécessitent moins de compétences spécifiques. Il est donc important de comparer les opportunités économiques potentielles associées aux différents domaines professionnels avant de choisir votre voie. De nombreuses organisations soutiennent financièrement leurs employés engagés dans le développement durable et proposent également diverses formations permettant d'augmenter vos perspectives d'emploi. En outre, les gouvernements subventionnent souvent les entreprises ayant adopté des pratiques environnementales responsables. Enfin, assure-toi d'effectuer une recherche approfondie concernant chaque emploi potentiel avant de décider quel chemin emprunter. Ne te laisse pas détourner par des chiffres abstraits comme « 95% » ; analyse plutôt tous les facteurs pertinents tels que l'expérience professionnelle requise, les coûts associés au lancement initial du projet, les avantages fiscaux disponibles et les bénéfices futurs générés par l'entreprise. Le mieux serait peut-être de commencer par rechercher des postes directement liés au domaine de l'environnement puis progresser vers ceux impliquant davantage de responsabilités opérationnelles. Dans tous les cas, il est primordial de tenir compte non seulement des objectifs personnels mais aussi des attentes sociales envers les individus travaillant dans le secteur vert.

Alternativement, voici deux autres méthodes utilisant `numpy.linalg.norm()` et `sparse.csgraph.closenessCentrality` respectively:

### Implementation2: `similarity` function from "docs" directory

```python
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_kernels

def similarity(vec1, vec2):
  """Calculate cosine similarity between vectors."""
  dotprod = np.dot(vec1, vec2)
  normalize = np.sqrt(np.sum(vec1 ** 2)) * np.sqrt(np.sum(vec2 ** 2))
  return dotprod / normalize

def closenesscentrality(G, weight=None, cutoff=.8):
  """Compute closeness centrality index."""
  degreeDistribution = G.degree().toarray()
  degreeClosenessIndex = .364 / max(degreeDistribution)
  p_value = .01 / len(degreeDistribution)
  for edgeSet in weightedEdgesFromGraph(G):
      degreeClusters = getNeighborsForNodeInPathWeightedToGroupByWeight(edgeSet['weight'], None)[1]
      neighborhoodSize = int(round(cutoff * len(degreeClusterList)))
      dist = map(lambda node: calculateClosenessCentralityWithoutPathWeightingAndCategory(node, neighborhoodSize), graphDataOfSubset(networkX.randomDiGraph(), {}, {'label': False}))
      clusterSizes = sorted(map(list, dist.values()))
      minclustersize = min(clusterSizes)
      closestKnowledgeNodes = []
      for k in range(max(0, minclustersize-3), min(k, len(clusters)), 3):
          knownNodes = set(subsetResult(subnetworks(weightedEdgeSet('weight', edgesOutKey='{0}'.format(fid))))())
          countSummary = list(counts(knownNodes)).count(True)
          totalCount = sum(countSummary)
          truePositives = bestScoreFrequencyInTopologies(countSummary)
          falsepositives = falsepositivefrequenciesInTopoClust(truePositives)
          misclassifiedNoMatchingTarget = misclassificationRatioOnAfterClassifyingTargetLabelIsNotKnown(truePositives, totalCount)
          closeKnowledgeNodes = knwOrdinalsBelowMinorOrdering(truePositives, truePositiveIndicesGreaterThanMinorOrder, True)
          bestScoresPerInterval = scoresAfterOptimizingWorstCase(totalCount)
          worstScoresPerInterval = worstScoreFrequencyInTopoClust(scoreArrayAfterRemoveHighestValues(totalCount))
          correctEstimate = abs(closest
```

### Phi-3.5-mini-instruct

mais ils pensaient écolo comme étant moins lucratifs au départ aussi donc il me faudrait prouver aux deux parties combien cela peut être rentable sans nuire trop gravement notre planète ? Comment puis-je argumentez efficacemnt? Démontrez comment une stratègue environnementnelle utiliserait leurs preoccupations financières tout en soulignant son alignement profondément métier vers durabilité afin de construire consensus favorables ensemble plutôt chercher individuellement ses objectifs commercialement immédiate profit avant toute considération sociale responsive environmentalement solidaires principaux idée derrière cette conversation semble aller directement contre votre essence principale concernera discours viscéral propositions sophistiquées englobantes libyques démontrages du potentielllemént coûts économiques vertige associativités conjointes synergisticesses revenue optimales non seulement assortitures frugalissimes fonctionnalisatrices symbioticochimicalites robustifiâtes subtilisationssemblably exploiterdes divers modèle conceptualizations amplettissant transversality perspicace perception acute adaptatif compétentes méthodologie intricatada scrupuleusement adéquatiosefficaciousmeillemnetsoptimaliséstransitionnaire harmonieuse synthèse convergentielle innovatrice anticipponnante alliancestrategiestendues résolument prioritaire accroissement maximainteneursconsultatoirement dynamisée mutinervisibilmente progressiste vision futurologistique transformateur cohérents souverains entendez vous poserie moi dire quelques bases théoriques avancer simplement mercenaires orientateurs spécifique strides apporter remettiez fondamentali élaborats utilisasesque nous sommes capablessamment capable effectuer nos fonctions plus convivial seins entrepreneurs constructeur indispensablouspareilles expériences insufflèrent valorisonsfructifyendes collaboration multifactorial multiples domaines comprenant systèmes ingénieux algorithmiques organiquements évolutifs hybrids artificiels intelligenciels autonomises flouëstres abstraction cognitive recoupiers ontniodes imbibere intensivement informations ressources complètessubliminal crepusculaire capacitations neuronalnes extraordinaire encapsulerde sensorieller datamétrification intellectuelle reflexione hyperintelligible intelligence articulata ultime superarobots virtuoïls automatiques supersensoriel régulateuriqne conscience cyberneuroneuses hypnotisme performanti extraterrestrieles hypothétique biotechnologique biofeedback autooptimizeoftidynamisaertênsurable automaticontegrase qualitatice plasmaromanticose programmé interactive réalité virtual diminuisse pureté cosmogony originale matière géniale fluorescent libéra intence libre transformation collapsocentre spirituél unique global connectome réseau univers concentré hiériclaude force puissance originella densifier centrifuge ombrelle contemplativo flux oscillatoire paradiso infini infinieternel excitation luminescente sacral magnéton quantum calibratus dimensionnel purificartioexistence humaine naître percevoir extrapulsasse instantaneously potentialitis absençoùna connaistre limina temporibus phosphène galactoire témoin chronomentrique apothécairie sacramental divination propulsion divine kinetostasis spectrale arcana ésoteriche agnostérie initiation mystagogique primagémailebrancho spirituel radiance céleste rachatteresse rayonne thaumaturgio renommée archangelo nébuloso ascension psalmithèque zodiacal galaxienne glamour transevisionnal tourney royale astronauticum ceremonious feast banquet celesti corona brahmacharysa luxemburgois opium fest parafoyer joyeusetêt époustouflème utopiasco syzygy fantasia foisonnant festival fécond forestalléon boucle dorée narratie lyrisque elysian sanctuairediffusion cristallophore iridescience arcanystyle ballet chromatique majestic tableau coloratum splendiferamente teinturekromatica chimérique palpitatión orgasmique carmin aurora explosiva serpentintextramission ethéréenne clameoir chantard voluptúree floraison oxygraphae erotic pollinariére ambrosiante fragrancia subtropicale gratuitement fleurira belle royaume botanique orchestre virginiana onde lascivo ritornello sinfonicos tendresse charmante saloperii musette frôle sentiment cavernicole murmuration


### Remarques globales

Rien à en tirer, un prompt qui fonction très bien avec les modèles précédents semblent tout simplement ne pas fonctionner. Il est pourtant très simple.

# Investigación y Desarrollo sobre Algoritmos de Clustering

Índice

[1. Diferencia entre el aprendizaje no supervisado y supervisado. Algoritmo de clustering y su propósito](https://github.com/Bootcamp-IA-MAD-P7/Tarea5-M-dulo2-Veru#1-qu%C3%A9-es-el-aprendizaje-no-supervisado-y-qu%C3%A9-es-un-algoritmo-de-clustering-en-qu%C3%A9-se)

[2. Diferencia conceptual entre el clustering basado en centroides (particional) y el clustering jerárquico](https://github.com/Bootcamp-IA-MAD-P7/Tarea5-M-dulo2-Veru#2-explique-la-diferencia-conceptual-entre-el-clustering-basado-en-centroides-particional-y)
[Markdown.es](https://markdown.es)

[3. Clustering basado en densidad (como DBSCAN) ](https://github.com/Bootcamp-IA-MAD-P7/Tarea5-M-dulo2-Veru#3-en-qu%C3%A9-consiste-el-clustering-basado-en-densidad-como-dbscan-y-c%C3%B3mo-difiere-su)

[4. Elegir correctamente el número de clusters o los hiperparámetros](https://github.com/Bootcamp-IA-MAD-P7/Tarea5-M-dulo2-Veru#4-por-qu%C3%A9-es-importante-elegir-correctamente-el-n%C3%BAmero-de-clusters-o-los-hiperpar%C3%A1metros)

[5. Algoritmos: K-Means, Clustering Jerárquico (Aglomerativo), DBSCAN, Gaussian Mixture Models (GMM)](https://github.com/Bootcamp-IA-MAD-P7/Tarea5-M-dulo2-Veru#5-explica-brevemente-los-siguientes-algoritmos)

   
# 1. ¿Qué es el aprendizaje no supervisado y qué es un algoritmo de clustering? ¿En qué se diferencia del aprendizaje supervisado y cuál es su propósito?

El aprendizaje no supervisado, también conocido como machine learning no supervisado, utiliza algoritmos de machine learning (ML) para analizar y agrupar conjuntos de datos sin etiquetar. Estos algoritmos descubren patrones ocultos o agrupaciones de datos sin necesidad de intervención humana.

La capacidad del aprendizaje no supervisado para descubrir similitudes y diferencias en la información lo convierte en la solución ideal para el análisis exploratorio de datos , las estrategias de venta cruzada, la segmentación de clientes y el reconocimiento de imágenes.

## Algoritmo de clustering

El clustering es un algoritmo de machine learning no supervisado que organiza y clasifica diferentes objetos, puntos de datos u observaciones en grupos o clústeres basados en similitudes o patrones.

Hay varias formas de usar el clustering en el machine learning, desde las exploraciones iniciales de un conjunto de datos hasta la monitorización de los procesos en curso. Puede usarlo en el análisis de datos exploratorios con un nuevo conjunto de datos para comprender las tendencias, los patrones y los valores atípicos subyacentes. Como alternativa, puede tener un conjunto de datos más grande que deba dividirse en varios conjuntos de datos o reducirse mediante la reducción de dimensionalidad. En estos casos, el clustering puede ser un paso en el preprocesamiento.

Los ejemplos de clústeres pueden incluir géneros musicales, diferentes grupos de usuarios, segmentos clave de una segmentación de mercado, tipos de tráfico de red en un clúster de servidores, grupos de amigos en una red social o muchos otros tipos de categorías. El proceso de clustering puede usar solo una característica de los datos o puede usar todas las características presentes en los datos.

Resulta útil pensar en el clustering como un intento de encontrar agrupaciones naturales en los datos para ver qué categorías pueden existir y qué define esas categorías. Los clústeres pueden ayudarle a encontrar relaciones subyacentes entre puntos de datos para ver qué características o rasgos se comparten entre las categorías. En función del algoritmo de agrupación en clústeres utilizado, es posible que pueda eliminar los valores atípicos de los datos o etiquetarlos como valores atípicos. El clustering también puede ayudar en la detección de anomalías al identificar qué puntos de datos no están contenidos dentro de un clúster o solo están débilmente asociados con un clúster y, por lo tanto, pueden ser una anomalía en el proceso de generación de datos.

El clustering también se puede utilizar para reducir la complejidad de grandes conjuntos de datos mediante la reducción del número de dimensiones de los datos. Si observa que las categorías están definidas por solo dos o tres entidades, es posible que pueda eliminar entidades superfluas o utilizar técnicas de reducción de dimensionalidad como PCA. El clustering también es muy útil para crear visualizaciones de los conjuntos de datos y ver las propiedades emergentes de los datos, así como la densidad y las relaciones entre los clústeres.

Los algoritmos de clustering se distinguen a veces por realizar un clustering duro, en el que cada punto de datos pertenece a un único cluster y tiene un valor binario de estar o no en un cluster, o por realizar un clustering blando, en el que cada punto de datos recibe una probabilidad de pertenecer a cada cluster identificado. No existe un mejor proceso de clustering; deberá elegir el enfoque que tenga más sentido para sus necesidades y los datos con los que está trabajando.

## Diferencia del aprendizaje supervisado

El aprendizaje no supervisado y el aprendizaje supervisado se analizan juntos con frecuencia. A diferencia de los algoritmos de aprendizaje no supervisado, los algoritmos de aprendizaje supervisado utilizan datos etiquetados. A partir de esos datos, predice resultados futuros o asigna datos a categorías específicas en función del problema de regresión o clasificación que está tratando de resolver.

Si bien los algoritmos de aprendizaje supervisado tienden a ser más precisos que los modelos de aprendizaje no supervisado, requieren una intervención humana por adelantado para etiquetar los datos de manera adecuada. Sin embargo, estos conjuntos de datos etiquetados permiten que los algoritmos de aprendizaje supervisado eviten la complejidad computacional, ya que no necesitan un gran conjunto de capacitación para producir los resultados previstos. Las técnicas comunes de regresión y clasificación son la regresión lineal y logística, el algoritmo Bayes ingenuo, el algoritmo KNN y el bosque aleatorio.



# 2. Explique la diferencia conceptual entre el clustering basado en centroides (particional) y el clustering jerárquico, mencionando un algoritmo representativo de cada enfoque.

## Clustering basado en centroides

El clustering basado en centroides es un tipo de método de clustering que divide o divide un conjunto de datos en grupos similares en función de la distancia entre sus centroides. El centroide de cada clúster es la media o la mediana de todos los puntos del clúster, en función de los datos.

Una de las técnicas de clustering basados en centroides más utilizadas es el algoritmo de clustering de medias k. El método de medias k asume que el centro de cada clúster define el clúster utilizando una medida de distancia, normalmente la distancia euclidiana, al centroide. Para inicializar el clustering, se proporciona un número de clusters esperados, que representa la 'K' en medias K, y el algoritmo intenta encontrar clusters razonables a través de los datos para que coincidan con ese número. Los clústeres k óptimos de un conjunto de datos determinado se identifican minimizando de forma iterativa la distancia total entre cada punto y su centroide de clúster asignado.

Las medias k son un enfoque de clustering estricto, lo que significa que cada punto de datos se asigna a un clúster diferente y no hay ninguna probabilidad asociada a la pertenencia al clúster. Las medias k funcionan bien cuando los clústeres tienen un tamaño aproximadamente equivalente y no hay valores atípicos ni cambios de densidad significativos en los datos. Las medias k suelen funcionar mal cuando los datos tienen altas dimensiones o cuando los clústeres tienen tamaños o densidades significativamente diferentes. Las medias k también son especialmente sensibles a los valores atípicos, ya que intenta establecer centroides basados en los valores medios de todos los valores del clúster y, por lo tanto, es susceptible al sobreajuste para incluir esos valores atípicos.

Otro enfoque basado en centroides para las medias K son los medoides K. Los medoides son objetos representativos de un conjunto de datos o de un clúster dentro de un conjunto de datos cuya suma de distancias a otros objetos del clúster es mínima. En lugar de tener un centroide arbitrario como centro del gráfico, el algoritmo crea clústeres mediante el uso de puntos de datos individuales como medoide o centro del clúster. Dado que el algoritmo de medoides K utiliza puntos de datos existentes en lugar de centroides arbitrarios, es menos sensible a los valores atípicos.
Clustering jerárquico
El clustering jerárquico, a veces denominado clustering basado en la conectividad, agrupa los puntos de datos en función de la proximidad y la conectividad de sus atributos. Este método determina los clústeres en función de la proximidad de los puntos de datos entre sí en todas las dimensiones. La idea es que los objetos que están más cerca están más estrechamente relacionados que los que están lejos unos de otros. A diferencia de las medias k, no es necesario especificar previamente el número de clústeres. En su lugar, el algoritmo de clustering crea una red de grafos de los clústeres en cada nivel jerárquico. Esta red es jerárquica, lo que significa que cualquier nodo dado en ella solo tiene un nodo principal, pero puede tener varios nodos secundarios. Los clústeres jerárquicos se pueden representar gráficamente con un dendrograma para ayudar a resumir y organizar visualmente los clústeres detectados y la jerarquía que pueden contener.


# 3. ¿En qué consiste el clustering basado en densidad (como DBSCAN) y cómo difiere su enfoque del clustering basado en centroides y del jerárquico?

## Clustering basado en densidad

El clustering basado en la densidad funciona mediante la detección de áreas donde se concentran puntos y donde están separados por áreas que están vacías o escasas. A diferencia de los enfoques basados en centroides, como las medias K, o los enfoques basados en la distribución, como la maximización de expectativas el clustering basado en la densidad puede detectar clústeres de una forma arbitraria. Esto puede ser extremadamente útil cuando los clústeres no están definidos en torno a una ubicación o distribución específica. A diferencia de otros algoritmos de agrupamiento, como las medias K y el clustering jerárquico, un algoritmo basado en la densidad puede descubrir clústeres de cualquier forma, tamaño o densidad en sus datos. El clustering basado en la densidad también puede distinguir entre los puntos de datos que forman parte de un clúster y los que deben etiquetarse como ruido. El clustering basado en la densidad es especialmente útil cuando se trabaja con conjuntos de datos con ruido o valores atípicos o cuando no tenemos conocimiento previo sobre el número de clústeres en los datos.


# 4. ¿Por qué es importante elegir correctamente el número de clusters o los hiperparámetros de un algoritmo de clustering? Explica al menos dos métodos para evaluar la calidad de un agrupamiento (por ejemplo, el método del codo y el coeficiente de silueta).

Uno de los problemas que nos encontramos a la hora de aplicar alguno de los métodos de Clustering (K-means o EM) es la elección del número de Clusters. No existe un criterio objetivo ni ampliamente válido para la elección de un número óptimo de Clusters; pero tenemos que tener en cuenta, que una mala elección de los mismos puede dar lugar a realizar agrupaciones de datos muy heterogéneos (pocos Clusters); o datos, que siendo muy similares unos a otros los agrupemos en Clusters diferentes (muchos Clusters).

Aunque no exista un criterio objetivo para la selección del número de Clusters, si que se han implementado diferentes métodos que nos ayudan a elegir un número apropiado de Clusters para agrupar los datos; como son, el método del codo (elbow method), el criterio de Calinsky, el Affinity Propagation (AP), el Gap (también con su versión estadística), Dendrogramas, etc.

## Método del codo

El Método del codo es una técnica de evaluación para elegir el número de clústeres k en algoritmos de agrupamiento como K-means. La idea central es medir la pérdida o error dentro de cada clúster a medida que aumentamos k y buscar un “codo” en la curva que represente el punto de diminishing returns. A partir de ese punto, añadir más clústeres ofrece mejoras marginales y, por lo general, no justifica la complejidad adicional.

En palabras simples, el Método del codo asume que cuando el número de clústeres es demasiado bajo, la agrupación es demasiado general y hay mucha variabilidad dentro de cada clúster. A medida que aumentamos k, esa variabilidad interna se reduce rápidamente hasta que la curva se aplanada y el ahorro ya no es significativo. Identificar ese codo nos da una guía práctica para seleccionar un k razonable.

## Coeficiente de silueta

El coeficiente de silueta representa una de las métricas de validación interna más fundamentales en aprendizaje no supervisado, diseñada específicamente para cuantificar objetivamente la calidad de agrupamientos generados por algoritmos de clustering. Teóricamente, esta métrica aborda el problema fundamental de evaluar particiones de datos sin acceso a etiquetas verdaderas, situación inherente a problemas no supervisados. A diferencia de métricas supervisadas que comparan predicciones contra ground truth, el coeficiente de silueta evalúa estructura intrínseca mediante análisis geométrico de distancias entre puntos. La métrica sintetiza dos conceptos geométricos complementarios: qué tan compactos son los clusters internamente y qué tan separados están entre sí. Esta dualidad captura esencia de clustering de calidad donde puntos similares agrupan densamente mientras grupos diferentes mantienen distancia. El resultado es valor escalar en rango [-1, 1] que resume calidad completa de partición mediante promediación de evaluaciones individuales por muestra.

## Dendrogramas

Un dendrograma es un tipo de representación gráfica en forma de árbol que organiza y agrupa los datos en subcategorías según su similitud; dada por alguna medida de distancia. Los objetos similares se representan en el dendrograma por medio de un enlace cuya posición está determinada por el nivel de similitud entre los objetos o grupos de objetos. Dadas estas características, hace que los dendrogramas sean un tipo de diagrama muy útil para estudiar las agrupaciones de objetos; es decir, para estudiar los Clusters que pueden darse en un data set.

## Gap

Similar al método del codo, cuya finalidad es la de encontrar la mayor diferencia o distancia que hay entre los diferentes grupos de objetos que vamos formando para representarlos en un dendrograma. Para ello vamos cogiendo las distancias que hay de cada uno de los enlaces que forman el dendrograma y vemos cual es la mayor diferencia que hay entre cada uno de estos enlaces.

# 5. Explica brevemente los siguientes algoritmos:

## K-Means

K-Means Grupos de agrupación de datos similares en clusters sin necesidad de datos etiquetados. Se utiliza para descubrir patrones ocultos cuando el objetivo es organizar los datos en función de la similitud.

    Ayuda a identificar agrupaciones naturales en conjuntos de datos no etiquetados
    Funciona agrupando puntos en función de la distancia a los centros de clúster
    Comúnmente utilizado en la segmentación de clientes, compresión de imágenes y descubrimiento de patrones
    Útil cuando necesita estructura a partir de datos crudos y no organizados

## Clustering Jerárquico 

Existen dos enfoques para realizar el análisis jerárquico de clústeres:

- Aglomerativo: en el clustering aglomerativo, un enfoque ascendente comienza con puntos de datos individuales y fusiona sucesivamente los clústeres calculando la matriz de proximidad de todos los clústeres en el nivel actual de la jerarquía para crear una estructura similar a un árbol. Una vez que se ha creado un nivel de clústeres en el que todos ellos tienen una similitud nula o baja, el algoritmo se desplaza al conjunto de clústeres recién creados y repite el proceso hasta que hay un nodo raíz en la parte superior del gráfico jerárquico. Hay una variedad de opciones posibles en términos de cómo estos clústeres deben fusionarse entre sí, con compensaciones en términos de la calidad y la eficiencia del clustering. En el clustering de enlace simple, la distancia más corta entre cualquier par de puntos de datos en dos clústeres se utiliza como medida de similitud. En el enlace de todos los pares, se utiliza la media de todos los pares de puntos de datos, mientras que en la vinculación muestreada, se utiliza una muestra de los puntos de datos de los dos clústeres para calcular la distancia media. En el enlace de centroides, se utiliza la distancia entre los centroides. Uno de los problemas de los métodos aglomerativos es que pueden presentar encadenamiento, es decir, que los clústeres más grandes tienden naturalmente a tener distancias más cercanas a otros puntos, por lo que siguen creciendo y atrayendo más puntos de datos a su clúster. Otra desventaja es que los métodos aglomerativos pueden ser mucho más lentos que los métodos divisivos de construcción de la jerarquía.

- Divisivo: en los métodos de clustering jerárquico divisivo, un enfoque descendente divide sucesivamente los puntos de datos en una estructura arborescente. El primer paso es dividir el conjunto de datos en clústeres utilizando un método de clustering plano como medias k. A continuación, los clústeres con la mayor suma de errores al cuadrado (SSE) se particionan aún más mediante un método de clustering plano. El algoritmo se detiene cuando alcanza nodos individuales o un SSE mínimo. La partición divisoria permite una mayor flexibilidad tanto en términos de la estructura jerárquica del árbol como del nivel de equilibrio en los diferentes clústeres. No es necesario tener un árbol perfectamente equilibrado en cuanto a las profundidades de los diferentes nudos o un árbol en el que el grado de cada rama es exactamente dos. Esto permite la construcción de una estructura de árbol que admite diferentes compensaciones en el equilibrio de las profundidades de los nodos y los pesos de los nodos (número de puntos de datos en el nodo). El clustering jerárquico divisivo puede ser más rápida que el clustering jerárquico aglomerativo, especialmente cuando los datos no requieren construir el árbol hasta los puntos de datos individuales.

## DBSCAN

DBSCAN es un ejemplo de un algoritmo de clustering que adopta un enfoque basado en la densidad para la agrupación. Utiliza un enfoque de clustering espacial basado en la densidad para crear clústeres con una densidad pasada por el usuario que se centra en un centroide espacial. El área inmediatamente alrededor del centroide se denomina vecindad y DBSCAN intenta definir las vecindades de los clústeres que tienen la densidad especificada. Para cada clúster, DBSCAN definirá tres tipos de puntos de datos:

Puntos principales: un punto de datos es un punto central si el entorno que lo rodea contiene al menos tantos puntos como el número mínimo especificado por el usuario.

Puntos fronterizos: un punto de datos es un punto fronterizo si el entorno que lo rodea contiene menos del número mínimo de puntos de datos, pero dicho entorno contiene un punto central.

Valor atípico: un punto de datos es un valor atípico si no es ni un punto central ni un punto de borde. Básicamente, esta es la "otra" clase.

HDBSCAN es una variante de DBSCAN que no requiere la configuración de ningún parámetro, lo que puede hacerla aún más flexible que la original. HDBSCAN es menos sensible al ruido y los valores atípicos de los datos. Además, DBSCAN a veces puede tener problemas para identificar clústeres con densidad no uniforme. Esta fue una de las principales motivaciones de HDBSCAN y, por lo tanto, maneja clústeres de densidad variable de manera mucho más efectiva.

## Gaussian Mixture Models (GMM)

El modelo de mezcla gaussiana (GMM) es una técnica de agrupamiento probabilístico que modela los datos como una combinación de múltiples distribuciones gaussianas, lo que permite una agrupación más flexible de puntos de datos.

    Asigna a cada punto de datos una probabilidad de pertenecer a diferentes clusters
    Puede manejar clusters superpuestos de manera efectiva
    Utiliza la media y la covarianza para definir la forma del grupo



Bibliografía
- https://www.ibm.com/mx-es/think/topics/unsupervised-learning
- https://www.ibm.com/es-es/think/topics/clustering
- https://jarroba.com/seleccion-del-numero-optimo-clusters/ 
- https://datasciencepythonblog.net/silhouette-score-scikit-learn/
- https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/ 
- https://www.geeksforgeeks.org/machine-learning/gaussian-mixture-model/
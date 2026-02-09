# Comité de sabios

El **comité de sabios** es una implementación del algoritmo **VotingClassifier** de scikit-learn, diseñado como un ensamble que combina las predicciones de tres modelos distintos para tomar una decisión final basada en el consenso. El comité utiliza una estrategia de "**hard voting**" (votación mayoritaria), donde la clase ganadora es la que recibe la mayoría de los votos individuales de los "sabios".

A continuación se detalla el funcionamiento de cada uno de los integrantes de este comité:

## 1. El Perceptrón ("El Abuelo")
Es un clasificador lineal clásico y sencillo. En esta configuración, actúa como el miembro veterano que toma decisiones directas: si la suma ponderada de las entradas no supera un umbral determinado, la predicción no pasa. Técnicamente, es un caso particular del clasificador de Descenso de Gradiente Estocástico (SGD) con una tasa de aprendizaje constante y sin penalización.

## 2. Random Forest ("La Fuerza de la Naturaleza")
Este "sabio" es en realidad un meta-estimador que coordina un bosque de múltiples árboles de decisión (por defecto 100).

- **Funcionamiento interno:** Entrena cada árbol en diferentes submuestras del conjunto de datos y utiliza el promedio (o votación interna) para mejorar la precisión y controlar el sobreajuste.
- **Rol en el comité:** Aporta la opinión de "100 árboles discutiendo entre ellos" para llegar a una conclusión robusta basada en la estructura del bosque.

## 3. Passive Aggressive Classifier ("El Inestable")
El clasificador **pasivo-agresivo** es una familia de algoritmos para el aprendizaje **en línea**, lo que significa que el modelo se entrena procesando los registros **uno a uno** en lugar de utilizar lotes (batches) de datos. Esta característica lo hace ideal para situaciones con recursos de memoria limitados, ya que sus requisitos no crecen con el tamaño del conjunto de datos.

Su funcionamiento se basa en tres propiedades críticas que definen su comportamiento ante cada nuevo dato:

#### Lógica de actualización: "Pasivo" vs "Agresivo"
El nombre del algoritmo describe con precisión cómo reacciona ante la información:

- **Pasivo:** Si el modelo predice correctamente la clase de un nuevo punto de datos, se queda "pasivo" y realiza **cero cambios** en sus parámetros. Simplemente pasa al siguiente registro.
- **Agresivo:** Si el modelo comete un error en la predicción, se vuelve "agresivo" y **reajusta el gradiente (la curva de decisión) inmediatamente** hasta que el último punto de datos sea clasificado de forma correcta.

Un aspecto fundamental es que al algoritmo **no le importa la historia pasada**; prioriza corregir el error del dato actual incluso si eso significa que dejaría de clasificar correctamente puntos de datos que ya había procesado anteriormente.

#### Parámetros y configuraciones clave
Para implementar este clasificador en herramientas como scikit-learn, existen parámetros esenciales que regulan su comportamiento:

- **Parámetro de agresividad (C):** Controla cuánto se regulariza el cambio en el modelo. Como regla general, debe ser pequeño si los datos tienen mucho ruido para evitar cambios demasiado drásticos.
- **Funciones de pérdida (Loss):**
	- **Hinge:** Equivale a la versión PA-I del algoritmo original.
	- **Squared Hinge:** Equivale a la versión PA-II y suele ser el valor por defecto en algunas implementaciones.
- **Sensibilidad al orden:** El rendimiento puede ser significativamente peor si los datos están ordenados, por lo que es crucial **aleatorizar o barajar (shuffle)** los datos antes del entrenamiento.

#### Nota sobre su implementación actual
Es importante tener en cuenta que, en las versiones más recientes de sciit-learn, la clase `PassiveAggressiveClassifier` ha sido marcada como **obsoleta (deprecated)** y se eliminará en la versión 1.10. La recomendación oficial es utilizar el **SGDClassifier** configurando el parámetro `learning_rate` como "pa1" o "pa2", y desactivando las penalizaciones (`penalty=None`).

## Mecanismo de Coordinación
El **VotingClassifier** actúa como el director del comité. Al entrenarse (`fit`), crea clones de estos tres modelos y los entrena individualmente. Al realizar una predicción (`predict`):

1. Solicita una etiqueta de clase a cada uno de los tres modelos.
2. Aplica la **regla de la mayoría**: la clase que obtenga al menos dos de los tres votos es la que el comité entrega como resultado final.

## Clasificación multiclase con algoritmos de clasificación binaria
Tanto el **perceptrón** como el **clasificador pasivo-agresivo** son clasificadores binarios diseñados para separar dos clases mediante un hiperplano. Sin embargo, funcionan correctamente en problemas multiclase debido a la estrategia que implementan internamente:

Scikit-learn adapta automáticamente estos modelos binarios para problemas multiclase utilizando la técnica **One-Versus-All** (también conocida como One-Vs-Rest). En lugar de entrenar un solo modelo, el sistema entrena **un clasificador binario por cada clase**. Por ejemplo, en el caso de las células, entrena tres sub-modelos: "Normal vs. No-Normal", "Benigno vs. No-Benigno" y "Maligno vs. No-Maligno".

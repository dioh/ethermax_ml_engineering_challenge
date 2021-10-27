# API Limiting algorithms
## Concepts

API Gateway

## Algorithms

* Token bucket: Numero de tokens que se rellena periodicamente y se utilizan por cada llamado. Necesitas uno por cada api call que querés restringir
* Leaky bucket: Utiliza una cola y lo que llega si no hay tokens en el queue se descarta. Se procesan mensajes a tiempo fijo
* Fixed Window counter: A tiempo de ventana fijo se permite una cantidad de peticiones. Puede generar un pico de requests si se realizan muchas llamadas al fina lde la ventana. Se comporta bien en caso promedio.
* Sliding window  log: mueve la ventana para que nunca se exeda el rate. Consume mucha memoria ya que mantiene los time stamps de lso requests. 
* Sliding window counter

Consistent Hashing
Quorum Consensus
Gossip Protocol



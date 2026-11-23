# Predicción de retiro y segmentación de clientes

Ejercicio de ciencia de datos realizado como prueba técnica. El caso se presenta de forma anonimizada: una entidad ofrece productos de crédito y busca identificar clientes con riesgo de retiro y explorar posibles segmentos. La finalidad es mostrar el razonamiento, no presentar una solución lista para producción.

## Cómo abordar una prueba técnica

1. **Entender el problema:** definir qué significa retiro, cuándo se predice, para qué se usará la predicción y cuál sería el costo de actuar o no actuar. Documentar supuestos y preguntas pendientes.
2. **Revisar datos y permisos:** comprobar diccionario, claves, duplicados, faltantes y distribución de la variable objetivo. Validar que el material pueda compartirse y que cada variable esté disponible en el momento de la predicción.
3. **Establecer una línea base:** separar entrenamiento y evaluación antes de ajustar transformaciones; comparar contra una predicción sencilla. En un problema temporal o con observaciones repetidas por cliente, preferir una separación temporal o por cliente.
4. **Entrenar y evaluar:** ajustar imputación, codificación y escalado solo con entrenamiento, dentro de un pipeline. Evaluar la clase minoritaria con precisión, recall, F1, matriz de confusión y área bajo la curva precisión-recall, además de ROC-AUC. Elegir el umbral según costos y capacidad de intervención, usando validación, no el conjunto final de prueba.
5. **Segmentar e interpretar:** si se aplica clustering, comparar alternativas, revisar estabilidad y perfiles agregados, y evitar atribuir causalidad o características personales sin evidencia. Formular acciones verificables.
6. **Comunicar:** presentar hallazgos, limitaciones, riesgos y siguientes pasos de forma clara. Revisar desempeño y posibles disparidades entre grupos antes de cualquier uso real.

## Evaluación reproducible

`evaluar_retiro.py` ofrece una alternativa sencilla al flujo histórico: combina las fuentes por `id`, reserva una partición estratificada sin remuestreo y compara una línea base con regresión logística. La imputación, el escalado y la codificación se ajustan únicamente sobre entrenamiento. Se excluyen identificadores y campos relacionados con el evento de cancelación para reducir el riesgo de fuga.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python evaluar_retiro.py
```

Ejecuta el comando desde la raíz del repositorio. Requiere Python 3.10–3.12 y los archivos `train.csv`, `train_test_demograficas.xlsx` y `train_test_subsidios.xlsx`. Muestra métricas agregadas y no guarda datos ni modelos.

En una ejecución con la partición y semilla indicadas, la clase de retiro representó aproximadamente el 2,8 % de la evaluación. La regresión logró un recall del 82,6 % y una precisión del 3,9 % para esa clase: detectó muchos casos, pero también produjo numerosos falsos positivos. Este intercambio debe discutirse frente al costo y la capacidad de intervención; no basta con destacar el recall.

**Limitación importante:** esta partición es ilustrativa, no una prueba independiente definitiva. El conjunto ya se utilizó en la exploración original y la selección de variables también se basó en ese análisis. Para estimar generalización se necesita un conjunto externo o temporal no utilizado previamente, además de verificar disponibilidad de las variables, calibración, equidad y viabilidad de intervención. Las variables demográficas requieren una revisión de uso, privacidad y sesgo.

## Material original y advertencias

- `Modelo_Fuga.ipynb`: análisis exploratorio original de retiro, segmentación y clasificación; sus resultados son históricos.
- `Data_Scientist_Test.html`, `Data_Scientist_Caso de Negocio.pptx` y `Data_Scientist_Test - Ejercicio 2 y preguntas 3.docx`: planteamiento y entregables originales con las referencias textuales a la entidad sustituidas.
- `train.csv`, `test.csv`, los archivos Excel y `diccionario_datos.xlsx`: fuentes y diccionario del ejercicio. `test.csv` no contiene la variable objetivo, por lo que no permite calcular métricas supervisadas sin etiquetas.
- Los archivos `.pkl` y el directorio `Modelo_Fuga-env` son artefactos históricos; no se requieren para ejecutar la evaluación propuesta. No cargues archivos pickle de fuentes que no sean de confianza.

En el notebook original se ajustan transformaciones y se aplica SMOTEENN antes de separar los conjuntos; además, el test se escala nuevamente por separado. Esto puede contaminar la evaluación y hacer incompatibles las transformaciones. Las métricas originales y las afirmaciones de generalización de la presentación **no deben interpretarse como evidencia de desempeño en producción**. El análisis de segmentación es exploratorio y no demuestra eficacia de campañas ni causalidad.

La sustitución de nombres no garantiza anonimato ni autorización para publicar los datos y documentos; también pueden existir referencias en imágenes, metadatos o versiones anteriores. El historial de Git conserva el contenido previo. Verifica los permisos y la confidencialidad antes de compartir este repositorio.

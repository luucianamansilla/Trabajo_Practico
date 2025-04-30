import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Descargar recursos de NLTK necesarios (se hace una sola vez)
nltk.download('punkt')       # Tokenizador de texto (separa palabras/oraciones)
nltk.download('stopwords')   # Lista de palabras vacías (the, and, etc.)
nltk.download('wordnet')     # Diccionario para lematización (formas base)
nltk.download('punkt_tab')   # Recurso adicional para la tokenización (soluciona el error)

# Corpus a analizar: lista de oraciones
corpus = [
    "Python is an interpreted and high-level language, while CPlus is a compiled and low-level language.",
    "JavaScript runs in web browsers, while Python is used in various applications, including data science and artificial intelligence.",
    "JavaScript is dynamically and weakly typed, while Rust is statically typed and ensures greater data security.",
    "Python and JavaScript are interpreted languages, while Java, CPlus, and Rust require compilation before execution.",
    "JavaScript is widely used in web development, while Go is ideal for servers and cloud applications.",
    "Python is slower than CPlus and Rust due to its interpreted nature.",
    "JavaScript has a strong ecosystem with Node.js for backend development, while Python is widely used in data science.",
    "JavaScript does not require compilation, while CPlus and Rust require code compilation before execution.",
    "Python and JavaScript have large communities and an extensive number of available libraries.",
    "Python is ideal for beginners, while Rust and CPlus are more suitable for experienced programmers."
]

# --- PREPARAR EL CORPUS ---

# Definir las stopwords en inglés
stop_words = set(stopwords.words('english'))

# Crear el lematizador
lemmatizer = WordNetLemmatizer()

# Lista donde vamos a guardar el corpus limpio
corpus_preparado = []

# Procesar cada oración del corpus
for texto in corpus:
    # Tokenizar (separar palabras) y pasar a minúsculas
    tokens = word_tokenize(texto.lower(), language='english')
    
    # Filtrar: eliminar palabras que no son alfabéticas o que son stopwords
    tokens_filtrados = [palabra for palabra in tokens if palabra.isalpha() and palabra not in stop_words]
    
    # Lematizar: pasar palabras a su forma raíz
    tokens_lematizados = [lemmatizer.lemmatize(palabra) for palabra in tokens_filtrados]
    
    # Unir las palabras de nuevo en una oración limpia
    frase_limpia = " ".join(tokens_lematizados)
    
    # Agregar al corpus limpio
    corpus_preparado.append(frase_limpia)

# Mostrar el corpus preparado
print("Corpus preparado:")
for linea in corpus_preparado:
    print(linea)

# --- GENERAR MATRIZ TF-IDF ---

# Crear el vectorizador TF-IDF
vectorizer = TfidfVectorizer()

# Ajustar y transformar el corpus limpio
tfidf_matrix = vectorizer.fit_transform(corpus_preparado)

# Mostrar la matriz TF-IDF como array
print("\nMatriz TF-IDF:\n")
print(tfidf_matrix.toarray())  # Cada fila = oración, cada columna = palabra

# Mostrar el vocabulario (las palabras analizadas)
print("\nVocabulario generado:\n")
print(vectorizer.get_feature_names_out())

# --- ANALIZAR PALABRAS FRECUENTES ---

# Lista para juntar todas las palabras de todo el corpus
tokens_totales = []

# Volvemos a tokenizar cada oración del corpus limpio
for texto in corpus_preparado:
    tokens = word_tokenize(texto.lower(), language='english')
    tokens_totales.extend(tokens)  # Agregar todas las palabras

# Contar frecuencia de cada palabra
frecuencia = {}
for palabra in tokens_totales:
    if palabra in frecuencia:
        frecuencia[palabra] += 1
    else:
        frecuencia[palabra] = 1

# Ordenar palabras de mayor a menor frecuencia
frecuencia_ordenada = sorted(frecuencia.items(), key=lambda x: x[1], reverse=True)

# Mostrar las 6 palabras más frecuentes
print("\nTop 6 palabras más frecuentes:")
for palabra, cantidad in frecuencia_ordenada[:6]:
    print(f"{palabra}: {cantidad}")

# Mostrar la palabra menos utilizada
print("\nPalabra menos utilizada:")
print(frecuencia_ordenada[-1])  # La última del ranking

# --- ANALIZAR PALABRAS REPETIDAS EN UNA ORACIÓN ---

# Elegir una oración específica (segunda oración)
oracion = corpus_preparado[1]

# Tokenizar esa oración
tokens_oracion = word_tokenize(oracion.lower(), language='english')

# Contar repeticiones de palabras
repeticiones = {}
for palabra in tokens_oracion:
    if palabra in repeticiones:
        repeticiones[palabra] += 1
    else:
        repeticiones[palabra] = 1

# Mostrar repeticiones en esa oración
print("\nPalabras repetidas en la oración seleccionada:")
for palabra, cantidad in repeticiones.items():
    print(f"{palabra}: {cantidad}")

# --- GRAFICAR LA DISTRIBUCIÓN DE FRECUENCIA ---

# Top 10 palabras para los gráficos
palabras_top = [p[0] for p in frecuencia_ordenada[:10]]
valores_top = [p[1] for p in frecuencia_ordenada[:10]]

# Gráfico de barras
plt.figure(figsize=(10, 5))
plt.bar(palabras_top, valores_top, color='skyblue')
plt.title("Top 10 Palabras Más Frecuentes (Barras)")
plt.xlabel("Palabras")
plt.ylabel("Frecuencia")
plt.xticks(rotation=45)
plt.show()

# Gráfico de torta (pastel)
etiquetas = [p[0] for p in frecuencia_ordenada[:5]]
valores = [p[1] for p in frecuencia_ordenada[:5]]

plt.figure(figsize=(6,6))
plt.pie(valores, labels=etiquetas, autopct='%1.1f%%', startangle=90)
plt.title("Distribución de las 5 Palabras Más Frecuentes (Torta)")
plt.show()

# Gráfico de líneas
plt.figure(figsize=(10,5))
plt.plot(palabras_top, valores_top, marker='o', linestyle='-', color='green')
plt.title("Frecuencia de Palabras (Gráfico de Línea)")
plt.xlabel("Palabras")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

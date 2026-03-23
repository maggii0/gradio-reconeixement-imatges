# Classificador de Residus amb Gradio

Aplicacio web amb Gradio que classifica imatges de residus en 3 categories:

- `♻ Reciclable`
- `🌱 Organic`
- `🚫 No Reciclable`

La classificacio es fa amb un model preentrenat d'imatges (`google/efficientnet-b0`) via HuggingFace `transformers`, i despres es tradueix a categories de residus amb regles de negoci.

## Que fa l'app

- Rep una imatge des de pujada de fitxer o webcam.
- Executa inferencia amb EfficientNet-B0.
- Extreu el `top-5` de prediccions (`softmax + torch.topk`).
- Mapeja cada etiqueta a categoria de residu amb `mapejar_a_categoria_residu()`.
- Mostra:
  - categoria recomanada + contenidor,
  - targeta HTML formatejada,
  - distribucio de confianca per categoria amb `gr.Label`.

## Arquitectura (4 blocs)

### Bloc 1 - Imports i model

Fitxer: `app.py`

- Imports: `gradio`, `transformers`, `PIL`, `torch`.
- Carrega lazy del model amb `carregar_model()` i cache `@lru_cache(maxsize=1)`.
- Seleccio automatica de dispositiu (`cuda` si disponible, si no `cpu`).

### Bloc 2 - Regles de negoci (mapeig)

- Diccionari `CATEGORIES_RESIDUS` amb:
  - etiqueta visual,
  - color,
  - contenidor,
  - exemples,
  - paraules clau.
- Funcio `mapejar_a_categoria_residu(label_imagenet)`:
  - normalitza text,
  - comprova coincidencies de keywords,
  - retorna `no_reciclable` per defecte.

### Bloc 3 - Prediccio

Funcio `classificar_imatge(image)`:

- Preprocessa imatge amb `AutoImageProcessor`.
- Executa inferencia amb `torch.inference_mode()`.
- Converteix logits a probabilitats amb `F.softmax`.
- Calcula top-k (`TOP_K = 5`).
- Acumula confiances per categoria (`category_scores`).
- Normalitza la distribucio i retorna:
  - HTML ric amb detall de resultat,
  - diccionari de confidencies per `gr.Label`.

### Bloc 4 - Interficie Gradio

- Construida amb `gr.Blocks` i layout de dues columnes.
- Esquerra:
  - `gr.Image` (upload/webcam),
  - boto `Classificar residu`,
  - boto `Netejar`.
- Dreta:
  - `gr.HTML` (targeta de resultat),
  - `gr.Label` (distribucio de confianca).
- Estils CSS passats al `launch(css=...)` per compatibilitat amb Gradio 6.

## Respostes de reflexio

- **Que es Pillow i per que s'instal·la?**
  - Pillow es la llibreria d'imatges de Python (fork modern de PIL).
  - S'utilitza per carregar i manipular imatges en format que `transformers` pot processar.

- **Que representen `outputs.logits`?**
  - Son puntuacions crues del model per cada classe ImageNet.
  - No son probabilitats directes.

- **Que fa `softmax` i per que l'apliquem?**
  - Converteix logits en probabilitats (0..1) que sumen 1.
  - Permet comparar confianca entre classes.

- **Per que `torch.topk(probs, 5)`? Per que 5?**
  - Recupera les 5 classes mes probables.
  - Dona context (no nomes una etiqueta), util per mapejar millor i explicar resultat.
  - Si puges el valor, veus alternatives menys probables i mes sorolloses.

- **Finalitat de `top5_labels` i `top5_scores`?**
  - `top5_labels`: noms humans de les classes mes probables.
  - `top5_scores`: probabilitats associades a aquestes classes.

## Instal·lacio i execucio

### 1) Crear virtual environment

```bash
python -m venv .venv
```

### 2) Activar-lo

Windows (PowerShell/CMD):

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3) Instal·lar dependencies

```bash
pip install torch pillow gradio "transformers[sentencepiece]"
```

### 4) Arrencar l'app

```bash
python app.py
```

Obre la URL local que imprimeix Gradio (normalment `http://127.0.0.1:7860`).

## Verificacio recomanada

- Puja una imatge de botella de plastic -> esperat: `Reciclable`.
- Puja una imatge de banana o fruita -> esperat: `Organic`.
- Revisa que el panell de confianca mostri les 3 categories.

## Estructura del projecte

```text
gradio_image_bot/
|- .venv/
|- app.py
|- README.md
|- Statement.pdf
```

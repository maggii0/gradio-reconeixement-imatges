import gradio as gr
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import torch.nn.functional as F

# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
model = AutoModelForImageClassification.from_pretrained("google/efficientnet-b0")
model.eval()


CATEGORIES_RESIDUS = {
    "reciclable": {
        "label": "♻️ Reciclable",
        "color": "#2ecc71",
        "contenidor": "Contenidor GROC o BLAU",
        "exemples": "Paper, cartro, plastic, metall, vidre",
        "keywords": ["bottle", "can", "container", "paper", "cardboard",
                     "plastic", "glass", "metal", "aluminum", "newspaper"],
    },
    "organic": {
        "label": "🌱 Organic",
        "color": "#8B4513",
        "contenidor": "Contenidor MARRO",
        "exemples": "Restes de menjar, closques, fulles",
        "keywords": ["food", "fruit", "vegetable", "plant", "leaf",
                     "banana", "apple", "salad", "organic", "compost"],
    },
    "no_reciclable": {
        "label": "🚫 No Reciclable",
        "color": "#e74c3c",
        "contenidor": "Contenidor NEGRE o GRIS",
        "exemples": "Bolquers, burilles, esponges, residus mixtos",
        "keywords": [],  # categoria per defecte
    },
}


def mapejar_a_categoria_residu(label_imagenet):
    """Map an ImageNet label to one of the 3 waste categories."""
    label_lower = label_imagenet.lower()
    for categoria, data in CATEGORIES_RESIDUS.items():
        for keyword in data["keywords"]:
            if keyword in label_lower:
                return categoria
    return "no_reciclable"


def classificar_imatge(image):
    """Classify an image and return the waste category with confidence."""
    if image is None:
        return "Puja una imatge per classificar", {}

    try:
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process image
        inputs = processor(images=image, return_tensors="pt")

        # Get model output
        with torch.no_grad():
            outputs = model(**inputs)

        # Get top 5 predictions
        probs = F.softmax(outputs.logits, dim=-1)[0]

        top5_probs, top5_ids = torch.topk(probs, 5)

        top5_labels = [model.config.id2label[idx.item()] for idx in top5_ids]

        top5_scores = [round(p.item(), 4) for p in top5_probs]

        # Agafa la informacio sobre les categories i suma les confiançes per cada categoria
        category_scores = {cat: 0.0 for cat in CATEGORIES_RESIDUS.keys()}
        for label, score in zip(top5_labels, top5_scores):
            categoria = mapejar_a_categoria_residu(label)
            category_scores[categoria] += score

        # Find best category
        best_categoria = max(category_scores, key=category_scores.get)
        best_conf = category_scores[best_categoria]
        cat_data = CATEGORIES_RESIDUS[best_categoria]

        # Build HTML response with dynamic color
        html_result = f"""
        <div style="border: 3px solid {cat_data['color']}; border-radius: 10px; padding: 20px;">
            <h2 style="color: {cat_data['color']}; margin: 0;">{cat_data['label']}</h2>
            <p><strong>Confianca:</strong> {best_conf:.1%}</p>
            <p><strong>Contenidor:</strong> {cat_data['contenidor']}</p>
            <p><strong>Exemples:</strong> {cat_data['exemples']}</p>
            <hr>
            <h4>Top 5 prediccions:</h4>
            <ol>
        """
        for label, score in zip(top5_labels, top5_scores):
            mapped = mapejar_a_categoria_residu(label)
            mapped_data = CATEGORIES_RESIDUS[mapped]
            html_result += f'<li>{label} ({score:.2%}) - <span style="color: {mapped_data["color"]}; font-weight: bold;">{mapped_data["label"]}</span></li>'
        html_result += "</ol></div>"

        # Confidence dict for label component (keys from CATEGORIES_RESIDUS)
        conf_per_categoria = {
            data["label"]: category_scores[cat]
            for cat, data in CATEGORIES_RESIDUS.items()
        }

        return html_result, conf_per_categoria

    except Exception as e:
        error_html = f"<div style='color: red; padding: 20px;'>Error: {str(e)}</div>"
        return error_html, {}


# Build Gradio interface
with gr.Blocks(title="Classificador de Residus", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Classificador de Residus")
    gr.Markdown("Puja una imatge d'un residu per classificar-lo en: Reciclable, Organic o No Reciclable.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Imatge del residu")
            classify_btn = gr.Button("Classificar", variant="primary")

        with gr.Column():
            result_output = gr.HTML(
                value="<div style='padding: 20px; border: 2px dashed #ccc; border-radius: 10px; text-align: center;'>Puja una imatge i prem Classificar</div>",
                label="Resultat"
            )
            confidence_output = gr.Label(label="Confianca per categoria", num_top_classes=3)

    classify_btn.click(
        fn=classificar_imatge,
        inputs=image_input,
        outputs=[result_output, confidence_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)

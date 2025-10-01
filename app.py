import tkinter as tk
from tkinter import filedialog, Text
from PIL import Image, ImageTk
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch

class AIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Model Hugging Face App")
        self.root.geometry("900x700")

        # Dropdown for task selection
        self.option_var = tk.StringVar(value="Text-to-Image")
        options = ["Text-to-Image", "Image-Classification", "Audio-Transcription"]
        tk.Label(root, text="Select Task:", font=("Arial", 12)).pack()
        tk.OptionMenu(root, self.option_var, *options).pack()

        # Input area for text
        tk.Label(root, text="Enter text (if applicable):").pack()
        self.input_box = Text(root, height=5, width=60)
        self.input_box.pack()

        # File selection
        self.file_btn = tk.Button(root, text="Choose File (for Image/Audio)", command=self.load_file)
        self.file_btn.pack()

        # Run button
        self.run_btn = tk.Button(root, text="Run Model", command=self.run_model)
        self.run_btn.pack()

        # Output label
        self.output_label = tk.Label(root, text="Output will appear here", wraplength=700, justify="left")
        self.output_label.pack(pady=10)

        # For displaying generated images
        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Explanations section
        tk.Label(root, text="Explanations (OOP Concepts):", font=("Arial", 12, "bold")).pack()
        self.explain_box = Text(root, height=12, width=80)
        self.explain_box.pack()
        self.show_explanations()

        self.file_path = None

        # --------- Load Hugging Face models once ---------
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if device=="cuda" else torch.float32
        ).to(device)
        self.image_classifier = pipeline(
            "image-classification", model="google/vit-base-patch16-224", device=0 if device=="cuda" else -1
        )
        self.audio_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")

    def load_file(self):
        self.file_path = filedialog.askopenfilename()
        self.output_label.config(text=f"Selected file: {self.file_path}")

    def run_model(self):
        task = self.option_var.get()

        if task == "Text-to-Image":
            text = self.input_box.get("1.0", tk.END).strip()
            if text:
                self.output_label.config(text="Generating image, please wait...")
                self.root.update()
                result = self.sd_pipe(text, num_inference_steps=20).images[0]
                result.thumbnail((400, 400))
                img_tk = ImageTk.PhotoImage(result)
                self.image_label.config(image=img_tk)
                self.image_label.image = img_tk
                self.output_label.config(text="Generated Image from text!")
            else:
                self.output_label.config(text="Please enter some text for text-to-image.")

        elif task == "Image-Classification":
            if self.file_path:
                self.output_label.config(text="Classifying image...")
                self.root.update()
                preds = self.image_classifier(self.file_path)
                formatted = "\n".join([f"{r['label']}: {r['score']:.3f}" for r in preds])
                self.output_label.config(text=formatted)
            else:
                self.output_label.config(text="Please choose an image file first.")

        elif task == "Audio-Transcription":
            if self.file_path:
                self.output_label.config(text="Transcribing audio...")
                self.root.update()
                result = self.audio_pipe(self.file_path)
                self.output_label.config(text=f"Transcription: {result['text']}")
            else:
                self.output_label.config(text="Please choose an audio file first.")

    def show_explanations(self):
        explanation_text = """
We used Object-Oriented Programming (OOP) concepts in this code:

1. Class (AIApp): Encapsulates the GUI app and its model logic.
2. Encapsulation: Methods and variables (like run_model, load_file, file_path) 
   are grouped inside the class, keeping code modular.
3. Abstraction: The user interacts only with buttons and inputs, 
   without worrying about how Hugging Face models work internally.
4. Polymorphism: The run_model method behaves differently depending on 
   the task selected (text-to-image, image classification, audio transcription).
5. Extendability: New tasks (e.g., text summarization) can be easily added 
   by extending the class.
"""
        self.explain_box.insert("1.0", explanation_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = AIApp(root)
    root.mainloop()

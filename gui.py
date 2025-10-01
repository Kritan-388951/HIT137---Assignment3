import tkinter as tk
from tkinter import filedialog, Text
from PIL import Image, ImageTk
from transformers import pipeline


# -------- OOP Class for the App --------
class AIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Model Hugging Face App")
        self.root.geometry("800x600")

        # Dropdown for input type
        self.option_var = tk.StringVar(value="Text-to-Image")
        options = ["Text-to-Image", "Image-Classification"]
        tk.Label(root, text="Select Task:").pack()
        tk.OptionMenu(root, self.option_var, *options).pack()

        # Input area
        self.input_box = Text(root, height=5, width=50)
        self.input_box.pack()

        # Button for selecting file (for image input)
        self.file_btn = tk.Button(root, text="Choose File", command=self.load_file)
        self.file_btn.pack()

        # Run button
        self.run_btn = tk.Button(root, text="Run Model", command=self.run_model)
        self.run_btn.pack()

        # Output area
        self.output_label = tk.Label(root, text="Output will appear here")
        self.output_label.pack()

        # For displaying generated images
        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Explanations
        tk.Label(root, text="Explanations (OOP Concepts):", font=("Arial", 12, "bold")).pack()
        self.explain_box = Text(root, height=10, width=70)
        self.explain_box.pack()
        self.show_explanations()

        self.file_path = None

    def load_file(self):
        self.file_path = filedialog.askopenfilename()
        self.output_label.config(text=f"Selected file: {self.file_path}")

    def run_model(self):
        task = self.option_var.get()

        if task == "Text-to-Image":
            text = self.input_box.get("1.0", tk.END).strip()
            if text:
                pipe = pipeline("text-to-image", model="runwayml/stable-diffusion-v1-5")
                result = pipe(text, num_inference_steps=20)[0]["image"]
                result.thumbnail((300, 300))
                img_tk = ImageTk.PhotoImage(result)
                self.image_label.config(image=img_tk)
                self.image_label.image = img_tk
                self.output_label.config(text="Generated Image from text!")

        elif task == "Image-Classification":
            if self.file_path:
                pipe = pipeline("image-classification", model="google/vit-base-patch16-224")
                preds = pipe(self.file_path)
                self.output_label.config(text=f"Prediction: {preds[0]['label']} (score: {preds[0]['score']:.2f})")
            else:
                self.output_label.config(text="Please choose an image file first.")

    def show_explanations(self):
        explanation_text = """
We used Object-Oriented Programming (OOP) concepts in this code:
1. Class (AIApp): Encapsulates the entire GUI app and its logic.
2. Encapsulation: All methods and variables related to the app are grouped 
   inside the AIApp class (e.g., run_model, load_file).
3. Abstraction: The user only interacts with buttons and inputs, without 
   needing to know how pipelines internally work.
4. Polymorphism: The run_model method behaves differently depending on 
   the task selected (text-to-image vs image-classification).
5. Inheritance (potential): This class could be extended later for 
   more models (e.g., audio transcription).
"""
        self.explain_box.insert("1.0", explanation_text)


# -------- Run the App --------
if __name__ == "__main__":
    root = tk.Tk()
    app = AIApp(root)
    root.mainloop()

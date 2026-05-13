import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np

# ================= CONFIGURATION =================
MODEL_PATH = r"C:\Users\lenovo\Desktop\garbageDS\garbage_model.pth"
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = 224
# =================================================

class GarbageClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EcoScan - Smart Garbage Classifier")
        self.root.geometry("900x700")
        self.root.configure(bg="#2c3e50")
        
        self.device = torch.device("cpu")
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.setup_ui()
        self.img_path = None

    def load_model(self):
        try:
            model = models.mobilenet_v2(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, len(CLASS_NAMES))
            )
            
            if not os.path.exists(MODEL_PATH):
                messagebox.showerror("Model Error", f"Model file not found at:\n{MODEL_PATH}")
                return None
            
            state_dict = torch.load(MODEL_PATH, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            messagebox.showerror("Initialization Error", str(e))
            return None

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#1abc9c", height=80)
        header.pack(fill=tk.X)
        tk.Label(header, text="♻️ ECOSCAN CLASSIFIER", font=("Helvetica", 24, "bold"), 
                 bg="#1abc9c", fg="white").pack(pady=20)

        # Main Container
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(expand=True, fill=tk.BOTH, padx=40, pady=20)

        # Left side: Image Display
        self.left_panel = tk.Frame(main_frame, bg="#34495e", bd=2, relief="flat")
        self.left_panel.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10)

        self.img_display = tk.Label(self.left_panel, text="No Image Selected", 
                                    font=("Helvetica", 12), bg="#34495e", fg="#ecf0f1")
        self.img_display.pack(expand=True, fill=tk.BOTH, pady=10)

        # Right side: Controls and Results
        right_panel = tk.Frame(main_frame, bg="#2c3e50", width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        # Buttons
        btn_style = {"font": ("Helvetica", 12, "bold"), "cursor": "hand2", "bd": 0, "height": 2}
        tk.Button(right_panel, text="📂 LOAD IMAGE", bg="#3498db", fg="white", 
                  command=self.open_image, **btn_style).pack(fill=tk.X, pady=10)
        
        tk.Button(right_panel, text="🔍 CLASSIFY", bg="#e67e22", fg="white", 
                  command=self.classify, **btn_style).pack(fill=tk.X, pady=10)

        # Result Label
        tk.Label(right_panel, text="PREDICTION", font=("Helvetica", 10, "bold"), 
                 bg="#2c3e50", fg="#bdc3c7").pack(pady=(20, 0))
        self.res_label = tk.Label(right_panel, text="---", font=("Helvetica", 20, "bold"), 
                                  bg="#2c3e50", fg="#1abc9c")
        self.res_label.pack(pady=5)

        # Confidence Bars
        self.bars = {}
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TProgressbar", thickness=15, troughcolor='#34495e', bordercolor='#2c3e50')

        for cls in CLASS_NAMES:
            bar_frame = tk.Frame(right_panel, bg="#2c3e50")
            bar_frame.pack(fill=tk.X, pady=3)
            
            tk.Label(bar_frame, text=cls.capitalize(), font=("Helvetica", 9), 
                     bg="#2c3e50", fg="#ecf0f1", width=10, anchor="w").pack(side=tk.LEFT)
            
            pb = ttk.Progressbar(bar_frame, length=180, mode='determinate', style="TProgressbar")
            pb.pack(side=tk.LEFT, padx=5)
            
            pct = tk.Label(bar_frame, text="0%", font=("Helvetica", 9), bg="#2c3e50", fg="#bdc3c7")
            pct.pack(side=tk.LEFT)
            self.bars[cls] = (pb, pct)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.img_path = path
            img = Image.open(path)
            # Resize for display maintaining aspect ratio
            img.thumbnail((500, 450))
            self.photo = ImageTk.PhotoImage(img)
            self.img_display.config(image=self.photo, text="")
            self.res_label.config(text="---", fg="#1abc9c")
            for pb, pct in self.bars.values():
                pb['value'] = 0
                pct.config(text="0%")

    def classify(self):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded.")
            return
        if not self.img_path:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        try:
            image = Image.open(self.img_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0].numpy()

            pred_idx = np.argmax(probs)
            prediction = CLASS_NAMES[pred_idx]
            
            self.res_label.config(text=prediction.upper())

            for i, cls in enumerate(CLASS_NAMES):
                score = float(probs[i]) * 100
                pb, pct = self.bars[cls]
                pb['value'] = score
                pct.config(text=f"{score:.1f}%")
                
        except Exception as e:
            messagebox.showerror("Classification Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = GarbageClassifierGUI(root)
    root.mainloop()
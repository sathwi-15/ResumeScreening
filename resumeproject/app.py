import os
import re
import sys
import time
import string
import shutil
import pickle
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Third-party libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import PyPDF2

# --- Color Palette Configuration (Sleek Dark Theme) ---
BG_DARK = "#0f172a"          # Very dark slate (Main background)
BG_CARD = "#1e293b"          # Dark slate (Cards/Panels)
BG_TEXT_AREA = "#020617"     # Near black slate (Text preview)
TEXT_LIGHT = "#f8fafc"       # White/off-white (Main text)
TEXT_MUTED = "#94a3b8"       # Muted slate grey (Labels/Subtitles)
ACCENT_BLUE = "#3b82f6"      # Bright blue (Buttons/Actions)
ACCENT_BLUE_HOVER = "#2563eb"# Hover blue
ACCENT_GREEN = "#10b981"     # Emerald green (FIT/Matched skills)
ACCENT_RED = "#ef4444"       # Coral red (UNFIT/Missing skills)
BORDER_COLOR = "#334155"     # Medium slate (Borders/Dividers)

class ResumeScreeningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Resume Screening System")
        self.root.geometry("1200x750")
        self.root.configure(bg=BG_DARK)
        
        # Keep window minimum size
        self.root.minsize(1000, 650)
        
        # State variables
        self.model = None
        self.vectorizer = None
        self.model_accuracy = 0.0
        self.class_report = ""
        self.dataset_size = 0
        self.lemmatizer_cache = {}
        
        # Safe offline/built-in stop words fallback
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
            'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
            'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
            'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
            'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
            'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
            'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
            'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', 
            "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
            'wouldn', "wouldn't"
        }
        self.lemmatizer = None
        
        # Create directories
        os.makedirs("uploaded_resumes", exist_ok=True)
        
        # Configure grid expansion weights for main window
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Show main dashboard immediately (No loading screen!)
        self.show_main_dashboard()
        
        # Start ML training/cache loading in a background thread
        threading.Thread(target=self.train_ml_pipeline, daemon=True).start()

    def init_nltk(self):
        """Programmatically download NLTK data if missing, using robust fallbacks."""
        # Try to download and load from NLTK
        try:
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Warning: Could not load NLTK stopwords ({str(e)}). Using built-in fallback stopwords.")

        try:
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            try:
                nltk.data.find('corpora/omw-1.4')
            except LookupError:
                nltk.download('omw-1.4', quiet=True)
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            print(f"Warning: Could not initialize NLTK lemmatizer ({str(e)}). Lemmatization will be bypassed.")

    def preprocess_text(self, text):
        """NLP Text Preprocessing Pipeline."""
        if not isinstance(text, str):
            return ""
        # 1. Lowercase conversion
        text = text.lower()
        # 2. Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        # 3. Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        # 4. Remove numbers
        text = re.sub(r'\d+', ' ', text)
        # 5. Remove punctuation
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        # 6. Remove stopwords and perform lemmatization
        words = text.split()
        cleaned_words = []
        for word in words:
            if word not in self.stop_words:
                if word in self.lemmatizer_cache:
                    lem_word = self.lemmatizer_cache[word]
                else:
                    if self.lemmatizer:
                        lem_word = self.lemmatizer.lemmatize(word)
                    else:
                        lem_word = word
                    self.lemmatizer_cache[word] = lem_word
                cleaned_words.append(lem_word)
        return ' '.join(cleaned_words)

    def train_ml_pipeline(self):
        """Loads data/cache, trains model or loads cached weights, evaluates performance."""
        try:
            cache_path = "model_cache.pkl"
            csv_path = "Resume.csv"
            
            # 1. Try to load from cache first
            use_cached = False
            if os.path.exists(cache_path) and os.path.exists(csv_path):
                try:
                    csv_mtime = os.path.getmtime(csv_path)
                    cache_mtime = os.path.getmtime(cache_path)
                    if cache_mtime > csv_mtime:
                        use_cached = True
                except Exception:
                    use_cached = False
            
            if use_cached:
                try:
                    self.update_loading_status("Loading cached ML model...")
                    with open(cache_path, "rb") as f:
                        cache_data = pickle.load(f)
                    
                    self.model = cache_data["model"]
                    self.vectorizer = cache_data["vectorizer"]
                    self.model_accuracy = cache_data["accuracy"]
                    self.class_report = cache_data["report"]
                    self.dataset_size = cache_data["dataset_size"]
                    
                    # Load NLTK in background just in case we need lemmatizer for new uploads
                    self.init_nltk()
                    
                    print("="*60)
                    print("AI RESUME SCREENING SYSTEM - INITIALIZATION COMPLETE (FROM CACHE)")
                    print(f"Dataset Rows Loaded: {self.dataset_size}")
                    print(f"Model Accuracy on Test Set: {self.model_accuracy * 100:.2f}%")
                    print("="*60)
                    print("Classification Report:")
                    print(self.class_report)
                    print("="*60)
                    
                    stats_text = f" Model Accuracy: {self.model_accuracy * 100:.2f}%  |  Dataset size: {self.dataset_size} "
                    self.root.after(0, lambda: self.update_stats_header(stats_text))
                    return
                except Exception as e:
                    print(f"Warning: Failed to load cached model ({str(e)}). Training from scratch...")
            
            # 2. Complete Training Pipeline (If no cache or cache loading failed)
            if not os.path.exists(csv_path):
                self.root.after(0, lambda: self.show_error_dialog(
                    "Missing Dataset", 
                    f"Dataset '{csv_path}' was not found in the workspace directory.\nPlease make sure it is in the root folder."
                ))
                return
            
            # Initialize NLTK first
            self.update_loading_status("Initializing NLP components...")
            self.init_nltk()
            
            # Load dataset
            self.update_loading_status("Loading Resume.csv dataset...")
            df = pd.read_csv(csv_path)
            self.dataset_size = len(df)
            
            # Handle null values
            df = df.dropna(subset=['Resume_str', 'Category'])
            
            # Preprocess Resume text column
            self.update_loading_status("Preprocessing resumes (NLP clean & lemmatize)...")
            df['Cleaned_Resume'] = df['Resume_str'].apply(self.preprocess_text)
            
            # Machine Learning Pipeline
            self.update_loading_status("Vectorizing text using TF-IDF...")
            self.vectorizer = TfidfVectorizer(max_features=5000)
            X = self.vectorizer.fit_transform(df['Cleaned_Resume'])
            y = df['Category']
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Random Forest Classifier
            self.update_loading_status("Training Random Forest Classifier model...")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Calculate accuracy and report
            y_pred = self.model.predict(X_test)
            self.model_accuracy = accuracy_score(y_test, y_pred)
            self.class_report = classification_report(y_test, y_pred, zero_division=0)
            
            # Save trained model and metrics to cache for instant future startups
            try:
                cache_data = {
                    "model": self.model,
                    "vectorizer": self.vectorizer,
                    "accuracy": self.model_accuracy,
                    "report": self.class_report,
                    "dataset_size": self.dataset_size
                }
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                print("Model cache successfully saved for faster future launches.")
            except Exception as cache_err:
                print(f"Warning: Could not save model cache ({str(cache_err)})")
            
            # Print model accuracy on startup (as requested)
            print("="*60)
            print("AI RESUME SCREENING SYSTEM - INITIALIZATION COMPLETE")
            print(f"Dataset Rows Loaded: {self.dataset_size}")
            print(f"Model Accuracy on Test Set: {self.model_accuracy * 100:.2f}%")
            print("="*60)
            print("Classification Report:")
            print(self.class_report)
            print("="*60)
            
            # Transition to stats display in header
            stats_text = f" Model Accuracy: {self.model_accuracy * 100:.2f}%  |  Dataset size: {self.dataset_size} "
            self.root.after(0, lambda: self.update_stats_header(stats_text))
            
        except Exception as e:
            error_msg = f"An error occurred during training: {str(e)}"
            print(error_msg)
            self.root.after(0, lambda: self.update_loading_status("ML Initialization Failed"))
            self.root.after(0, lambda: messagebox.showerror("ML Model Error", f"Failed to train classification model.\nYou will not be able to classify resume categories, but text extraction will still function.\n\nError: {str(e)}"))

    # --- GUI Layouts ---

    def update_loading_status(self, message):
        """Thread-safe update of the header status text."""
        if hasattr(self, 'stats_lbl') and self.stats_lbl.winfo_exists():
            self.root.after(0, lambda: self.stats_lbl.config(text=f" AI Engine Status: {message} ", fg=ACCENT_BLUE))

    def update_stats_header(self, text):
        """Updates header stats and enables View Report button."""
        if hasattr(self, 'stats_lbl') and self.stats_lbl.winfo_exists():
            self.stats_lbl.config(text=text, fg=ACCENT_BLUE)
        if hasattr(self, 'report_btn') and self.report_btn.winfo_exists():
            self.report_btn.config(state="normal", bg=BORDER_COLOR)

    def show_error_dialog(self, title, message):
        messagebox.showerror(title, message)
        self.root.quit()

    def show_main_dashboard(self):
        """Creates the full feature-rich dashboard."""
            
        # Main Dashboard Container
        self.main_container = tk.Frame(self.root, bg=BG_DARK)
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=15)
        
        # Configure layout rows/cols: 1 header row, 1 main row
        self.main_container.grid_rowconfigure(0, minsize=65, weight=0) # Header
        self.main_container.grid_rowconfigure(1, weight=1)             # Main Panes
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # 1. HEADER SECTION
        header_frame = tk.Frame(self.main_container, bg=BG_DARK)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        
        header_title = tk.Label(
            header_frame, 
            text="AI Resume Screening Dashboard", 
            font=("Segoe UI", 20, "bold"), 
            fg=TEXT_LIGHT, 
            bg=BG_DARK
        )
        header_title.pack(side="left")
        
        # Model stats frame (Accuracy Display)
        stats_frame = tk.Frame(header_frame, bg=BG_CARD, bd=1, relief="solid", highlightbackground=BORDER_COLOR)
        stats_frame.pack(side="right", padx=5)
        
        self.stats_lbl = tk.Label(
            stats_frame, 
            text=" AI Engine Status: Initializing... Please wait. ", 
            font=("Segoe UI", 9, "bold"), 
            fg=TEXT_MUTED, 
            bg=BG_CARD
        )
        self.stats_lbl.pack(side="left", padx=5, pady=6)
        
        # Button to show classification report
        self.report_btn = tk.Button(
            stats_frame,
            text="View Report",
            font=("Segoe UI", 8, "bold"),
            fg=TEXT_LIGHT,
            bg=BORDER_COLOR,
            activebackground=BG_DARK,
            activeforeground=TEXT_LIGHT,
            bd=0,
            cursor="hand2",
            padx=8,
            command=self.show_class_report_popup,
            state="disabled"
        )
        self.report_btn.pack(side="right", padx=(0, 5), pady=4)
        self.report_btn.bind("<Enter>", lambda e: self.report_btn.config(bg=ACCENT_BLUE) if self.report_btn['state'] == 'normal' else None)
        self.report_btn.bind("<Leave>", lambda e: self.report_btn.config(bg=BORDER_COLOR) if self.report_btn['state'] == 'normal' else None)
        
        # 2. PANELS DIVISION (Split Left and Right)
        self.panes_frame = tk.Frame(self.main_container, bg=BG_DARK)
        self.panes_frame.grid(row=1, column=0, sticky="nsew")
        self.panes_frame.grid_rowconfigure(0, weight=1)
        
        # ================= LEFT PANEL =================
        self.left_panel = tk.Frame(self.panes_frame, bg=BG_DARK)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=0)
        self.left_panel.grid_rowconfigure(0, weight=0) # File upload row
        self.left_panel.grid_rowconfigure(1, weight=1) # Resume text preview row
        
        # File Upload Box
        upload_card = tk.Frame(self.left_panel, bg=BG_CARD, bd=1, relief="solid", highlightbackground=BORDER_COLOR)
        upload_card.grid(row=0, column=0, sticky="ew", pady=(0, 10), ipady=12, ipadx=10)
        
        upload_title = tk.Label(
            upload_card,
            text="Resume Document Loader",
            font=("Segoe UI", 12, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_CARD
        )
        upload_title.pack(anchor="w", padx=15, pady=(5, 2))
        
        upload_sub = tk.Label(
            upload_card,
            text="Upload candidate resumes in PDF format to analyze fit and extract skills.",
            font=("Segoe UI", 9),
            fg=TEXT_MUTED,
            bg=BG_CARD
        )
        upload_sub.pack(anchor="w", padx=15, pady=(0, 12))
        
        # Upload Button with hover effect
        self.upload_btn = tk.Button(
            upload_card,
            text="Upload Resume PDF",
            font=("Segoe UI", 11, "bold"),
            fg=TEXT_LIGHT,
            bg=ACCENT_BLUE,
            activebackground=ACCENT_BLUE_HOVER,
            activeforeground=TEXT_LIGHT,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=18,
            pady=8,
            command=self.handle_pdf_upload
        )
        self.upload_btn.pack(side="left", padx=12)
        self.upload_btn.bind("<Enter>", lambda e: self.upload_btn.config(bg=ACCENT_BLUE_HOVER))
        self.upload_btn.bind("<Leave>", lambda e: self.upload_btn.config(bg=ACCENT_BLUE))
        
        # Evaluate Button with hover effect
        self.evaluate_btn = tk.Button(
            upload_card,
            text="Evaluate",
            font=("Segoe UI", 11, "bold"),
            fg=TEXT_LIGHT,
            bg=ACCENT_GREEN,
            activebackground="#059669",
            activeforeground=TEXT_LIGHT,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=18,
            pady=8,
            command=self.handle_generate_analysis
        )
        self.evaluate_btn.pack(side="left", padx=12)
        self.evaluate_btn.bind("<Enter>", lambda e: self.evaluate_btn.config(bg="#059669"))
        self.evaluate_btn.bind("<Leave>", lambda e: self.evaluate_btn.config(bg=ACCENT_GREEN))
        
        self.filename_label = tk.Label(
            upload_card,
            text="No file uploaded",
            font=("Segoe UI", 9, "italic"),
            fg=TEXT_MUTED,
            bg=BG_CARD
        )
        self.filename_label.pack(side="left", padx=10)
        
        # Resume Preview Box
        preview_card = tk.Frame(self.left_panel, bg=BG_CARD, bd=1, relief="solid", highlightbackground=BORDER_COLOR)
        preview_card.grid(row=1, column=0, sticky="nsew")
        preview_card.grid_rowconfigure(0, weight=0) # Card header
        preview_card.grid_rowconfigure(1, weight=1) # Text area
        
        preview_header = tk.Label(
            preview_card,
            text="Resume Text Preview",
            font=("Segoe UI", 12, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_CARD
        )
        preview_header.grid(row=0, column=0, sticky="w", padx=15, pady=10)
        
        # Scrollable Text Widget
        text_frame = tk.Frame(preview_card, bg=BG_TEXT_AREA)
        text_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        
        self.text_preview = tk.Text(
            text_frame,
            bg=BG_TEXT_AREA,
            fg=TEXT_LIGHT,
            insertbackground=TEXT_LIGHT,
            font=("Consolas", 10),
            relief="flat",
            state="disabled",
            wrap="word",
            padx=10,
            pady=10
        )
        self.text_preview.grid(row=0, column=0, sticky="nsew")
        
        # Styled scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_preview.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.text_preview.config(yscrollcommand=scrollbar.set)
        
        # ================= RIGHT PANEL =================
        self.right_panel = tk.Frame(self.panes_frame, bg=BG_DARK)
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=1) # Scrollable container for analysis details
        
        # Scrollable analysis container (to handle small screen heights gracefully)
        self.analysis_canvas = tk.Canvas(self.right_panel, bg=BG_DARK, highlightthickness=0)
        self.analysis_canvas.grid(row=0, column=0, sticky="nsew")
        
        canvas_scrollbar = ttk.Scrollbar(self.right_panel, orient="vertical", command=self.analysis_canvas.yview)
        canvas_scrollbar.grid(row=0, column=1, sticky="ns")
        self.analysis_canvas.config(yscrollcommand=canvas_scrollbar.set)
        
        # Frame inside the canvas
        self.analysis_scroll_frame = tk.Frame(self.analysis_canvas, bg=BG_DARK)
        self.analysis_canvas_window = self.analysis_canvas.create_window(
            (0, 0), window=self.analysis_scroll_frame, anchor="nw"
        )
        
        # Configure scroll boundaries on resize
        self.analysis_scroll_frame.bind("<Configure>", self.on_canvas_configure)
        self.analysis_canvas.bind("<Configure>", self.on_canvas_width_configure)
        
        # Populate analysis section with components
        self.setup_analysis_results(self.analysis_scroll_frame)
        
    def on_canvas_configure(self, event):
        """Update canvas scroll region when the scrollable frame changes size."""
        self.analysis_canvas.configure(scrollregion=self.analysis_canvas.bbox("all"))
        
    def on_canvas_width_configure(self, event):
        """Match width of the inner frame to the canvas width."""
        self.analysis_canvas.itemconfig(self.analysis_canvas_window, width=event.width)

    def setup_analysis_results(self, parent):
        """Constructs layout structure of the right pane dashboard cards."""
        parent.grid_columnconfigure(0, weight=1)
        
        # 1. Prediction & Score Summary Card
        self.summary_card = tk.Frame(parent, bg=BG_CARD, bd=1, relief="solid", highlightbackground=BORDER_COLOR)
        self.summary_card.grid(row=0, column=0, sticky="ew", pady=(0, 12), ipady=10, ipadx=10)
        self.summary_card.grid_columnconfigure(0, weight=1)
        self.summary_card.grid_columnconfigure(1, weight=1)
        
        # Category Column
        cat_frame = tk.Frame(self.summary_card, bg=BG_CARD)
        cat_frame.grid(row=0, column=0, sticky="nsew", padx=15, pady=10)
        
        cat_lbl = tk.Label(cat_frame, text="PREDICTED CATEGORY", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_CARD)
        cat_lbl.pack(anchor="w")
        
        self.category_val = tk.Label(cat_frame, text="Pending Upload", font=("Segoe UI", 16, "bold"), fg=ACCENT_BLUE, bg=BG_CARD)
        self.category_val.pack(anchor="w", pady=(5, 0))
        
        # Score Column
        score_frame = tk.Frame(self.summary_card, bg=BG_CARD)
        score_frame.grid(row=0, column=1, sticky="nsew", padx=15, pady=10)
        
        score_lbl = tk.Label(score_frame, text="RESUME MATCH SCORE", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_CARD)
        score_lbl.pack(anchor="w")
        
        self.score_val = tk.Label(score_frame, text="- / 10", font=("Segoe UI", 16, "bold"), fg=TEXT_LIGHT, bg=BG_CARD)
        self.score_val.pack(anchor="w", pady=(5, 0))
        
        # Custom Canvas Progress Bar
        self.progress_canvas = tk.Canvas(self.summary_card, height=12, bg=BG_DARK, highlightthickness=0)
        self.progress_canvas.grid(row=1, column=0, columnspan=2, sticky="ew", padx=15, pady=(5, 10))
        self.draw_progress_bar(0) # Init empty progress bar
        
        # 2. STATUS CARD (FIT / UNFIT)
        self.status_card = tk.Frame(parent, bg=BG_CARD, bd=1, relief="solid", highlightbackground=BORDER_COLOR)
        self.status_card.grid(row=1, column=0, sticky="ew", pady=(0, 12), ipady=12, ipadx=10)
        self.status_card.grid_columnconfigure(0, weight=1)
        
        status_title_lbl = tk.Label(
            self.status_card, 
            text="SCREENING DECISION STATUS", 
            font=("Segoe UI", 9, "bold"), 
            fg=TEXT_MUTED, 
            bg=BG_CARD
        )
        status_title_lbl.pack(anchor="w", padx=15, pady=(5, 5))
        
        self.status_val_badge = tk.Label(
            self.status_card,
            text="AWAITING CANDIDATE RESUME",
            font=("Segoe UI", 13, "bold"),
            fg=TEXT_MUTED,
            bg=BORDER_COLOR,
            padx=20,
            pady=10,
            relief="flat",
            bd=0
        )
        self.status_val_badge.pack(anchor="w", padx=15, fill="x", expand=True)

        # 3. SKILLS MATCH ANALYSIS CARD
        self.skills_card = tk.Frame(parent, bg=BG_CARD, bd=1, relief="solid", highlightbackground=BORDER_COLOR)
        self.skills_card.grid(row=2, column=0, sticky="ew", pady=(0, 12), ipady=10, ipadx=10)
        self.skills_card.grid_columnconfigure(0, weight=1)
        self.skills_card.grid_columnconfigure(1, weight=1)
        
        # Left: Matched Skills
        self.matched_skills_frame = tk.Frame(self.skills_card, bg=BG_CARD)
        self.matched_skills_frame.grid(row=0, column=0, sticky="nsew", padx=15, pady=10)
        
        matched_title = tk.Label(
            self.matched_skills_frame, 
            text="✓ MATCHED SKILLS", 
            font=("Segoe UI", 10, "bold"), 
            fg=ACCENT_GREEN, 
            bg=BG_CARD
        )
        matched_title.pack(anchor="w")
        
        self.matched_list_lbl = tk.Label(
            self.matched_skills_frame,
            text="No matches yet.",
            font=("Segoe UI", 10),
            fg=TEXT_MUTED,
            bg=BG_CARD,
            justify="left",
            anchor="w"
        )
        self.matched_list_lbl.pack(anchor="w", pady=(8, 0))
        
        # Right: Missing Skills
        self.missing_skills_frame = tk.Frame(self.skills_card, bg=BG_CARD)
        self.missing_skills_frame.grid(row=0, column=1, sticky="nsew", padx=15, pady=10)
        
        missing_title = tk.Label(
            self.missing_skills_frame, 
            text="✗ MISSING SKILLS", 
            font=("Segoe UI", 10, "bold"), 
            fg=ACCENT_RED, 
            bg=BG_CARD
        )
        missing_title.pack(anchor="w")
        
        self.missing_list_lbl = tk.Label(
            self.missing_skills_frame,
            text="No missing analysis yet.",
            font=("Segoe UI", 10),
            fg=TEXT_MUTED,
            bg=BG_CARD,
            justify="left",
            anchor="w"
        )
        self.missing_list_lbl.pack(anchor="w", pady=(8, 0))

        # 4. DECISION REASONING CARD
        self.reason_card = tk.Frame(parent, bg=BG_CARD, bd=1, relief="solid", highlightbackground=BORDER_COLOR)
        self.reason_card.grid(row=3, column=0, sticky="ew", pady=(0, 10), ipady=12, ipadx=10)
        
        reason_title = tk.Label(
            self.reason_card, 
            text="EVALUATION REASON & COMMENTS", 
            font=("Segoe UI", 9, "bold"), 
            fg=TEXT_MUTED, 
            bg=BG_CARD
        )
        reason_title.pack(anchor="w", padx=15, pady=(5, 5))
        
        self.reason_val_lbl = tk.Label(
            self.reason_card,
            text="Please upload a resume in PDF format to trigger the evaluation reasoning module.",
            font=("Segoe UI", 10),
            fg=TEXT_MUTED,
            bg=BG_CARD,
            justify="left",
            wraplength=480
        )
        self.reason_val_lbl.pack(anchor="w", padx=15, pady=(5, 0))
        
        # Dynamically wrap text when window is resized
        self.reason_val_lbl.bind("<Configure>", lambda e: self.reason_val_lbl.config(wraplength=e.width - 20))

    def draw_progress_bar(self, percentage, is_fit=None):
        """Draws custom progress bar on canvas indicating matching score."""
        self.progress_canvas.delete("all")
        width = self.progress_canvas.winfo_width()
        if width <= 1:
            width = 450 # Fallback width before canvas is mapped to screen
            
        # Draw background track
        self.progress_canvas.create_rectangle(
            0, 0, width, 12, fill=BG_DARK, outline="", width=0
        )
        
        # Decide fill color
        if is_fit is None:
            fill_color = BORDER_COLOR
        else:
            fill_color = ACCENT_GREEN if is_fit else ACCENT_RED
            
        # Draw progress fill
        fill_width = int((percentage / 100.0) * width)
        if fill_width > 0:
            self.progress_canvas.create_rectangle(
                0, 0, fill_width, 12, fill=fill_color, outline="", width=0
            )

    # --- Feature Actions & Business Logic ---

    def handle_pdf_upload(self):
        """Allows user to select PDF, copy to uploaded_resumes, extract text."""
        file_path = filedialog.askopenfilename(
            title="Upload Resume PDF",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if not file_path:
            return # User canceled
            
        try:
            # 1. Save file to uploaded_resumes/
            filename = os.path.basename(file_path)
            # Create a clean safe filename
            cleaned_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
            dest_path = os.path.join("uploaded_resumes", cleaned_filename)
            shutil.copy(file_path, dest_path)
            
            # Update label
            self.filename_label.config(text=f"Uploaded: {filename}", fg=ACCENT_BLUE)
            
            # 2. Extract text from PDF
            extracted_text = self.extract_text_from_pdf(dest_path)
            
            # 3. Handle empty resume
            if not extracted_text.strip():
                raise ValueError("The uploaded PDF does not contain any readable text. It might be scanned or empty.")
            
            # 4. Preview Text in Box
            self.text_preview.config(state="normal")
            self.text_preview.delete("1.0", tk.END)
            self.text_preview.insert("1.0", extracted_text)
            self.text_preview.config(state="disabled")
            
            # Reset analysis dashboard to pending until they click Generate
            self.reset_analysis_dashboard()
            
            # Show success dialog
            messagebox.showinfo("Success", f"Successfully loaded:\n{filename}\n\nClick 'Generate Analysis' to screen the candidate.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process PDF file:\n{str(e)}")
            self.filename_label.config(text="Upload failed", fg=ACCENT_RED)
            # Reset UI preview
            self.text_preview.config(state="normal")
            self.text_preview.delete("1.0", tk.END)
            self.text_preview.config(state="disabled")
            self.reset_analysis_dashboard()

    def handle_generate_analysis(self):
        """Screens the candidate resume and updates the dashboard when the user clicks 'Generate Analysis'."""
        # 1. Get raw text from the preview box
        extracted_text = self.text_preview.get("1.0", tk.END).strip()
        if not extracted_text:
            messagebox.showwarning("No Resume", "Please upload a Resume PDF first.")
            return
            
        # 2. Check if the model is ready
        if not self.model or not self.vectorizer:
            messagebox.showwarning("Model Not Ready", "The Machine Learning model is still initializing in the background.\nPlease wait a moment and try again.")
            return
            
        try:
            # Change cursor to hourglass/watch
            self.root.config(cursor="watch")
            self.root.update_idletasks()
            
            # Execute analysis
            self.run_resume_analysis(extracted_text)
            
            # Reset cursor
            self.root.config(cursor="")
            
            # Show success dialog
            messagebox.showinfo("Success", "Resume analysis generated successfully!")
            
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("Analysis Error", f"Failed to run screening logic:\n{str(e)}")

    def extract_text_from_pdf(self, pdf_path):
        """Wrapper for PDF text extraction using PyPDF2."""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            raise ValueError(f"Invalid PDF file or format error: {str(e)}")

    def check_skill_match(self, skill_name, text):
        """Search for required skill in text with word boundaries and aliases."""
        text_lower = text.lower()
        if skill_name == "python":
            return bool(re.search(r'\bpython\b', text_lower))
        elif skill_name == "machine learning":
            return bool(re.search(r'\bmachine[- ]learning\b|\bml\b', text_lower))
        elif skill_name == "sql":
            return bool(re.search(r'\bsql\b|\bmysql\b|\bpostgresql\b|\bsqlite\b', text_lower))
        elif skill_name == "flask":
            return bool(re.search(r'\bflask\b', text_lower))
        return False

    def run_resume_analysis(self, resume_text):
        """Runs the NLP preprocess, Model prediction, and Skill matching logic."""
        if not self.model or not self.vectorizer:
            messagebox.showerror("System Error", "Machine learning model is not ready.")
            return
            
        # NLP Preprocessing
        cleaned_text = self.preprocess_text(resume_text)
        
        # Predict Category
        X_vec = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(X_vec)[0]
        
        # Skill matching logic
        required_skills = ["python", "machine learning", "sql", "flask"]
        matched_skills = []
        missing_skills = []
        
        for skill in required_skills:
            # We search in raw resume_text to preserve aliases/contexts, and cleaned_text
            if self.check_skill_match(skill, resume_text) or self.check_skill_match(skill, cleaned_text):
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)
                
        # Score formula: (number_of_matched_skills / total_required_skills) * 10
        num_matched = len(matched_skills)
        total_required = len(required_skills)
        score = (num_matched / total_required) * 10
        
        # Rules: score >= 7 is FIT, else UNFIT
        is_fit = score >= 7
        fit_status = "FIT FOR COMPANY" if is_fit else "UNFIT FOR COMPANY"
        
        # Determine Reason
        if is_fit:
            reason = "Candidate matches required technical parameters (Score >= 7.0/10) and exhibits suitable qualification."
        else:
            reason = "Candidate lacks required technical skills."
            
        # Update Dashboard UI Elements
        self.category_val.config(text=prediction.upper(), fg=ACCENT_BLUE)
        self.score_val.config(text=f"{score:.1f} / 10", fg=TEXT_LIGHT)
        
        # Draw progress bar on canvas
        self.draw_progress_bar(int(score * 10), is_fit)
        
        # Status Badge update
        status_bg = ACCENT_GREEN if is_fit else ACCENT_RED
        self.status_val_badge.config(
            text=fit_status,
            fg=TEXT_LIGHT,
            bg=status_bg
        )
        
        # Display skills hyphen list with correct capitalization
        skill_display_map = {
            "python": "Python",
            "machine learning": "Machine Learning",
            "sql": "SQL",
            "flask": "Flask"
        }
        
        if matched_skills:
            matched_text = "\n".join([f"- {skill_display_map[s]}" for s in matched_skills])
        else:
            matched_text = "- None"
        self.matched_list_lbl.config(text=matched_text, fg=TEXT_LIGHT)
        
        if missing_skills:
            missing_text = "\n".join([f"- {skill_display_map[s]}" for s in missing_skills])
        else:
            missing_text = "- None"
        self.missing_list_lbl.config(text=missing_text, fg=TEXT_LIGHT)
        
        # Reason display
        self.reason_val_lbl.config(text=reason, fg=TEXT_LIGHT)
        
        # Show right panel and divide space (40% left, 60% right)
        if hasattr(self, 'panes_frame') and hasattr(self, 'right_panel'):
            self.panes_frame.grid_columnconfigure(0, weight=4, uniform="panes")
            self.panes_frame.grid_columnconfigure(1, weight=5, uniform="panes")
            self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(20, 0))
            
            # Recalculate canvas scrolling and layout sizes
            self.root.update_idletasks()
            self.analysis_canvas.configure(scrollregion=self.analysis_canvas.bbox("all"))
        
    def reset_analysis_dashboard(self):
        """Resets the dashboard results to pending state and hides the right panel."""
        self.category_val.config(text="Pending Upload", fg=ACCENT_BLUE)
        self.score_val.config(text="- / 10", fg=TEXT_LIGHT)
        self.draw_progress_bar(0)
        
        self.status_val_badge.config(
            text="AWAITING CANDIDATE RESUME",
            fg=TEXT_MUTED,
            bg=BORDER_COLOR
        )
        self.matched_list_lbl.config(text="No matches yet.", fg=TEXT_MUTED)
        self.missing_list_lbl.config(text="No missing analysis yet.", fg=TEXT_MUTED)
        self.reason_val_lbl.config(
            text="Please upload a resume in PDF format to trigger the evaluation reasoning module.",
            fg=TEXT_MUTED
        )
        
        # Hide right panel and expand left panel to occupy 100% width
        if hasattr(self, 'right_panel'):
            self.right_panel.grid_remove()
        if hasattr(self, 'panes_frame'):
            self.panes_frame.grid_columnconfigure(0, weight=1, minsize=0)
            self.panes_frame.grid_columnconfigure(1, weight=0, minsize=0)

    # --- Popups ---

    def show_class_report_popup(self):
        """Displays the classification report in a scrollable, styled text popup window."""
        popup = tk.Toplevel(self.root)
        popup.title("Model Classification Report")
        popup.geometry("620x500")
        popup.configure(bg=BG_DARK)
        popup.transient(self.root)
        popup.grab_set()
        
        # Header title
        title_lbl = tk.Label(
            popup,
            text="Random Forest Model Performance",
            font=("Segoe UI", 14, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_DARK
        )
        title_lbl.pack(anchor="w", padx=20, pady=(20, 5))
        
        sub_lbl = tk.Label(
            popup,
            text=f"Precision, Recall, and F1-Score breakdown across all {len(self.class_report.splitlines()) - 5} categories.",
            font=("Segoe UI", 9),
            fg=TEXT_MUTED,
            bg=BG_DARK
        )
        sub_lbl.pack(anchor="w", padx=20, pady=(0, 15))
        
        # Text Frame
        report_frame = tk.Frame(popup, bg=BG_TEXT_AREA, bd=1, relief="solid", highlightbackground=BORDER_COLOR)
        report_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        report_frame.grid_rowconfigure(0, weight=1)
        report_frame.grid_columnconfigure(0, weight=1)
        
        text_widget = tk.Text(
            report_frame,
            bg=BG_TEXT_AREA,
            fg=TEXT_LIGHT,
            font=("Consolas", 10),
            relief="flat",
            wrap="none",
            padx=10,
            pady=10
        )
        text_widget.grid(row=0, column=0, sticky="nsew")
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(report_frame, orient="vertical", command=text_widget.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(report_frame, orient="horizontal", command=text_widget.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        
        text_widget.config(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        # Insert report content
        report_content = f"Model Accuracy: {self.model_accuracy * 100:.2f}%\n\n" + self.class_report
        text_widget.insert("1.0", report_content)
        text_widget.config(state="disabled")

# --- Application Entry Point ---
if __name__ == "__main__":
    # Ensure app starts inside the workspace directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    root = tk.Tk()
    app = ResumeScreeningApp(root)
    root.mainloop()

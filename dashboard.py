import json
from pathlib import Path
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CowDashboardGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cow Identification Dashboard")
        self.root.geometry("1000x700")
        
        # Load data
        if not self.load_data():
            return
        
        self.create_widgets()
    
    def load_data(self):
        """Load logs and gallery data"""
        if not Path('detection_logs.json').exists():
            messagebox.showerror("Error", "No logs found. Run cow_identification.py first.")
            self.root.destroy()
            return False
        
        with open('detection_logs.json', 'r') as f:
            self.logs = json.load(f)
        
        with open('gallery.json', 'r') as f:
            self.gallery = json.load(f)
        
        # Calculate statistics
        self.cow_appearances = defaultdict(int)
        for log in self.logs:
            self.cow_appearances[log['cow_id']] += 1
        
        return True
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Title
        title = tk.Label(self.root, text="🐄 COW IDENTIFICATION DASHBOARD", 
                        font=("Arial", 20, "bold"), bg="#2E7D32", fg="white", pady=10)
        title.pack(fill=tk.X)
        
        # Statistics Frame
        stats_frame = tk.LabelFrame(self.root, text="📊 Statistics", font=("Arial", 12, "bold"), padx=10, pady=10)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        stats_text = f"""Total Cows Registered: {len(self.gallery)}
Total Detections: {len(self.logs)}
Total Frames Processed: {self.logs[-1]['frame'] if self.logs else 0}"""
        tk.Label(stats_frame, text=stats_text, font=("Arial", 11), justify=tk.LEFT).pack()
        
        # Main content frame
        content_frame = tk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left: Cow Frequency Table
        left_frame = tk.LabelFrame(content_frame, text="🐮 Cow Appearance Frequency", font=("Arial", 11, "bold"))
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Treeview for cow frequency
        tree_frame = tk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree = ttk.Treeview(tree_frame, columns=("Cow ID", "Detections"), show="headings", yscrollcommand=scrollbar.set)
        self.tree.heading("Cow ID", text="Cow ID")
        self.tree.heading("Detections", text="Detections")
        self.tree.column("Cow ID", width=150)
        self.tree.column("Detections", width=100)
        self.tree.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.tree.yview)
        
        for cow_id, count in sorted(self.cow_appearances.items(), key=lambda x: x[1], reverse=True):
            self.tree.insert("", tk.END, values=(cow_id, count))
        
        # Right: Chart
        right_frame = tk.LabelFrame(content_frame, text="📈 Detection Chart", font=("Arial", 11, "bold"))
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.create_chart(right_frame)
        
        # Bottom: Recent Detections
        recent_frame = tk.LabelFrame(self.root, text="🕒 Recent Detections (Last 10)", font=("Arial", 11, "bold"))
        recent_frame.pack(fill=tk.X, padx=10, pady=10)
        
        recent_text = tk.Text(recent_frame, height=6, font=("Courier", 9))
        recent_text.pack(fill=tk.X, padx=5, pady=5)
        
        for log in self.logs[-10:]:
            recent_text.insert(tk.END, f"Frame {log['frame']:4d}: {log['cow_id']} (conf: {log['confidence']:.2f})\n")
        recent_text.config(state=tk.DISABLED)
        
        # Refresh button
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="🔄 Refresh", command=self.refresh, font=("Arial", 10), bg="#4CAF50", fg="white", padx=20).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="❌ Close", command=self.root.quit, font=("Arial", 10), bg="#f44336", fg="white", padx=20).pack(side=tk.LEFT, padx=5)
    
    def create_chart(self, parent):
        """Create bar chart for cow detections"""
        fig, ax = plt.subplots(figsize=(5, 4))
        
        cow_ids = list(self.cow_appearances.keys())
        counts = list(self.cow_appearances.values())
        
        ax.bar(cow_ids, counts, color='#2E7D32')
        ax.set_xlabel('Cow ID', fontsize=10)
        ax.set_ylabel('Detections', fontsize=10)
        ax.set_title('Cow Detection Frequency', fontsize=11, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def refresh(self):
        """Refresh dashboard data"""
        self.root.destroy()
        main()

def main():
    root = tk.Tk()
    app = CowDashboardGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

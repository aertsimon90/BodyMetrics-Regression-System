# BodyMetrics Regression System

import zevihanthosa
import random
import json
import sys 
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class System:
    def __init__(self, lobes=[[3,12],[12,6,3]], lobesset=[[1.8, 1.6], [0.8, 0.6]], genderlobes=[[3,12],[12,4,1]], genderlobesset=[[1.8, 1.6], [0.8, 0.6]], learning=0.05, truely=1, momentumexc=0.9, maxage=100, maxheight=200, maxkg=150):
        self.malemodel = zevihanthosa.Brain(lobes=lobes, learning=learning, truely=truely, momentumexc=momentumexc)
        self.femalemodel = zevihanthosa.Brain(lobes=lobes, learning=learning, truely=truely, momentumexc=momentumexc)
        self.gendermodel = zevihanthosa.Brain(lobes=genderlobes, learning=learning, truely=truely, momentumexc=momentumexc)
        self.malemodel.lobesset = lobesset
        self.femalemodel.lobesset = lobesset
        self.gendermodel.lobesset = genderlobesset
        self.data = []
        self.maxage = maxage
        self.maxheight = maxheight
        self.maxkg = maxkg
    def add_data(self, age, height, kg, gender):
        if age == 0 or height == 0 or kg == 0 or gender == 0.5:
            raise ValueError("Please use real values (not blank or zero)")
        age = min(age, self.maxage)/self.maxage
        height = min(height, self.maxheight)/self.maxheight
        kg = min(kg, self.maxkg)/self.maxkg
        gender = max(min(gender,1),0)
        self.data.append([age,height,kg,gender])
    def train(self, epoch=40):
        for i in range(epoch):
            shuffleddata = self.data.copy()
            random.shuffle(shuffleddata)
            for datai, data in enumerate(shuffleddata):
                print(f"Training... {(i/epoch)*100:.1f}% Epoch {(datai/len(shuffleddata))*100:.1f}% Dataset")
                age, height, kg, gender = data
                for useage in range(2):
                    for useheight in range(2):
                        for usekg in range(2):
                            if useage==0 and usekg==0 and useheight==0:
                                continue
                            age2 = age if useage==1 else 0
                            height2 = height if useheight==1 else 0
                            kg2 = kg if usekg==1 else 0
                            self.gendermodel.process([age2,height2,kg2], [gender])
                            self.gendermodel.process([age2,height2,kg2], [gender], naturalderiv=True)
                            if gender > 0.5:
                                self.femalemodel.process([age2,height2,kg2], [age,height,kg])
                                self.femalemodel.process([age2,height2,kg2], [age,height,kg], naturalderiv=True)
                            else:
                                self.malemodel.process([age2,height2,kg2], [age,height,kg])
                                self.malemodel.process([age2,height2,kg2], [age,height,kg], naturalderiv=True)
    def guess(self, age, height, kg, gender):
        age = min(age, self.maxage)/self.maxage
        height = min(height, self.maxheight)/self.maxheight
        kg = min(kg, self.maxkg)/self.maxkg
        gender = max(min(gender,1),0)
        if gender == 0.5:
            gender = self.gendermodel.process([age,height,kg], train=False)[0]
        if gender > 0.5:
            ageoutput, heightoutput, kgoutput = self.femalemodel.process([age,height,kg], train=False)
        else:
            ageoutput, heightoutput, kgoutput = self.malemodel.process([age,height,kg], train=False)
        return int(ageoutput*self.maxage), int(heightoutput*self.maxheight), int(kgoutput*self.maxkg), f"{(1-gender)*100:.1f}% Male {(gender)*100:.1f}% Female"
    def import_example_data(self, maxage=101):
        male_data = [[1, 76.0, 10.0, 0], [2, 88.0, 12.0, 0], [3, 96.0, 14.0, 0], [4, 103.0, 16.0, 0], [5, 110.0, 18.0, 0], [6, 116.0, 21.0, 0], [7, 122.0, 23.0, 0], [8, 126.0, 26.0, 0], [9, 131.5, 29.0, 0], [10, 137.0, 32.0, 0], [11, 143.0, 35.5, 0], [12, 149.0, 39.0, 0], [13, 156.03, 44.67, 0], [14, 163.07, 50.35, 0], [15, 170.1, 56.02, 0], [16, 173.4, 60.78, 0], [17, 175.2, 64.41, 0], [18, 175.7, 66.9, 0], [19, 176.5, 68.95, 0], [20, 177.0, 70.3, 0], [21, 176.75, 73.35, 0], [22, 176.51, 76.4, 0], [23, 176.26, 79.44, 0], [24, 176.02, 82.49, 0], [25, 175.77, 85.54, 0], [26, 175.82, 86.43, 0], [27, 175.87, 87.31, 0], [28, 175.92, 88.2, 0], [29, 175.97, 89.08, 0], [30, 176.02, 89.96, 0], [31, 176.08, 90.85, 0], [32, 176.13, 91.74, 0], [33, 176.18, 92.62, 0], [34, 176.23, 93.5, 0], [35, 176.28, 94.39, 0], [36, 176.28, 94.34, 0], [37, 176.28, 94.28, 0], [38, 176.28, 94.23, 0], [39, 176.28, 94.17, 0], [40, 176.28, 94.12, 0], [41, 176.28, 94.07, 0], [42, 176.28, 94.01, 0], [43, 176.28, 93.96, 0], [44, 176.28, 93.9, 0], [45, 176.28, 93.85, 0], [46, 176.18, 93.65, 0], [47, 176.08, 93.45, 0], [48, 175.97, 93.25, 0], [49, 175.87, 93.05, 0], [50, 175.77, 92.85, 0], [51, 175.67, 92.65, 0], [52, 175.57, 92.45, 0], [53, 175.46, 92.25, 0], [54, 175.36, 92.05, 0], [55, 175.26, 91.85, 0], [56, 175.18, 91.79, 0], [57, 175.11, 91.73, 0], [58, 175.03, 91.67, 0], [59, 174.96, 91.61, 0], [60, 174.88, 91.56, 0], [61, 174.8, 91.5, 0], [62, 174.73, 91.44, 0], [63, 174.65, 91.38, 0], [64, 174.58, 91.32, 0], [65, 174.5, 91.26, 0], [66, 174.35, 90.91, 0], [67, 174.2, 90.55, 0], [68, 174.05, 90.2, 0], [69, 173.9, 89.84, 0], [70, 173.75, 89.49, 0], [71, 173.6, 89.14, 0], [72, 173.45, 88.78, 0], [73, 173.3, 88.43, 0], [74, 173.15, 88.07, 0], [75, 173.0, 87.72, 0], [76, 172.74, 87.0, 0], [77, 172.49, 86.28, 0], [78, 172.23, 85.56, 0], [79, 171.97, 84.84, 0], [80, 171.72, 84.12, 0], [81, 171.46, 83.39, 0], [82, 171.2, 82.67, 0], [83, 170.94, 81.95, 0], [84, 170.69, 81.23, 0], [85, 170.43, 80.51, 0], [86, 170.17, 79.79, 0], [87, 169.92, 79.07, 0], [88, 169.66, 78.35, 0], [89, 169.4, 77.63, 0], [90, 169.15, 76.91, 0], [91, 168.89, 76.18, 0], [92, 168.63, 75.46, 0], [93, 168.37, 74.74, 0], [94, 168.12, 74.02, 0], [95, 167.86, 73.3, 0], [96, 167.6, 72.58, 0], [97, 167.35, 71.86, 0], [98, 167.09, 71.14, 0], [99, 166.83, 70.42, 0], [100, 166.58, 69.7, 0]]
        female_data = [[1, 74.5, 9.2, 1], [2, 86.5, 11.5, 1], [3, 95.0, 13.9, 1], [4, 102.0, 15.9, 1], [5, 109.0, 18.0, 1], [6, 115.0, 20.5, 1], [7, 121.0, 23.0, 1], [8, 127.0, 26.0, 1], [9, 132.5, 28.5, 1], [10, 138.0, 32.0, 1], [11, 144.5, 36.0, 1], [12, 150.5, 40.0, 1], [13, 156.0, 44.5, 1], [14, 159.5, 48.5, 1], [15, 161.5, 51.5, 1], [16, 162.0, 53.5, 1], [17, 162.5, 54.5, 1], [18, 162.5, 55.5, 1], [19, 162.6, 56.5, 1], [20, 162.6, 57.5, 1], [21, 162.6, 59.0, 1], [22, 162.6, 60.5, 1], [23, 162.6, 62.0, 1], [24, 162.6, 63.5, 1], [25, 162.6, 65.0, 1], [26, 162.6, 66.0, 1], [27, 162.6, 67.0, 1], [28, 162.6, 68.0, 1], [29, 162.6, 69.0, 1], [30, 162.6, 70.0, 1], [31, 162.6, 71.0, 1], [32, 162.6, 72.0, 1], [33, 162.6, 73.0, 1], [34, 162.6, 74.0, 1], [35, 162.6, 75.0, 1], [36, 162.6, 75.5, 1], [37, 162.6, 76.0, 1], [38, 162.6, 76.5, 1], [39, 162.6, 77.0, 1], [40, 162.6, 77.5, 1], [41, 162.6, 78.0, 1], [42, 162.6, 78.5, 1], [43, 162.6, 79.0, 1], [44, 162.6, 79.5, 1], [45, 162.6, 80.0, 1], [46, 162.5, 80.2, 1], [47, 162.4, 80.4, 1], [48, 162.3, 80.6, 1], [49, 162.2, 80.8, 1], [50, 162.1, 81.0, 1], [51, 162.0, 81.2, 1], [52, 161.9, 81.4, 1], [53, 161.8, 81.6, 1], [54, 161.7, 81.8, 1], [55, 161.6, 82.0, 1], [56, 161.5, 81.8, 1], [57, 161.4, 81.6, 1], [58, 161.3, 81.4, 1], [59, 161.2, 81.2, 1], [60, 161.1, 81.0, 1], [61, 161.0, 80.8, 1], [62, 160.9, 80.6, 1], [63, 160.8, 80.4, 1], [64, 160.7, 80.2, 1], [65, 160.6, 80.0, 1], [66, 160.4, 79.5, 1], [67, 160.2, 79.0, 1], [68, 160.0, 78.5, 1], [69, 159.8, 78.0, 1], [70, 159.6, 77.5, 1], [71, 159.4, 77.0, 1], [72, 159.2, 76.5, 1], [73, 159.0, 76.0, 1], [74, 158.8, 75.5, 1], [75, 158.6, 75.0, 1], [76, 158.3, 74.2, 1], [77, 158.0, 73.4, 1], [78, 157.7, 72.6, 1], [79, 157.4, 71.8, 1], [80, 157.1, 71.0, 1], [81, 156.8, 70.2, 1], [82, 156.5, 69.4, 1], [83, 156.2, 68.6, 1], [84, 155.9, 67.8, 1], [85, 155.6, 67.0, 1], [86, 155.3, 66.2, 1], [87, 155.0, 65.4, 1], [88, 154.7, 64.6, 1], [89, 154.4, 63.8, 1], [90, 154.1, 63.0, 1], [91, 153.8, 62.2, 1], [92, 153.5, 61.4, 1], [93, 153.2, 60.6, 1], [94, 152.9, 59.8, 1], [95, 152.6, 59.0, 1], [96, 152.3, 58.2, 1], [97, 152.0, 57.4, 1], [98, 151.7, 56.6, 1], [99, 151.4, 55.8, 1], [100, 151.1, 55.0, 1]]  
        data = []
        data += male_data[:maxage]
        data += female_data[:maxage]
        random.shuffle(data)
        for dat in data:
            self.add_data(dat[0],dat[1],dat[2],dat[3])

def save():
	global aimodel
	try:
		with open("model.json", "w") as f:
			f.write(json.dumps({"malemodel": zevihanthosa.save(aimodel.malemodel), "femalemodel": zevihanthosa.save(aimodel.femalemodel), "gendermodel": zevihanthosa.save(aimodel.gendermodel), "data": aimodel.data, "maxes": [aimodel.maxage, aimodel.maxheight, aimodel.maxkg]}))
	except Exception as e:
		print("Error in saving model:",e)

def load(autocreate=True):
	global aimodel
	try:
		with open("model.json", "r") as f:
			aimodel = System()
			data = json.loads(f.read())
			aimodel.malemodel = zevihanthosa.load(data["malemodel"])
			aimodel.femalemodel = zevihanthosa.load(data["femalemodel"])
			aimodel.gendermodel = zevihanthosa.load(data["gendermodel"])
			aimodel.data = data["data"]
			aimodel.maxage, aimodel.maxheight, aimodel.maxkg = data["maxes"][0], data["maxes"][1], data["maxes"][2]
	except Exception as e:
		print("Error in loading model:",e)
		if autocreate:
			print("Creating AI Model...")
			aimodel = System()
			aimodel.import_example_data()
			aimodel.train()
			save()

def createnewmodel(lobes=[[3,12],[12,6,3]], lobesset=[[1.8, 1.6], [0.8, 0.6]], genderlobes=[[3,12],[12,4,1]], genderlobesset=[[1.8, 1.6], [0.8, 0.6]], learning=0.05, truely=1, momentumexc=0.9, maxage=100, maxheight=200, maxkg=150, pretrain=True, pretrainmaxage=101, pretrainepoch=40):
	global aimodel
	aimodel = System(lobes=lobes, lobesset=lobesset, genderlobes=genderlobes, genderlobesset=genderlobesset, learning=learning, truely=truely, momentumexc=momentumexc, maxage=maxage, maxheight=maxheight, maxkg=maxkg)
	if pretrain:
		aimodel.import_example_data(pretrainmaxage)
		aimodel.train(epoch=pretrainepoch)
	save()

def addnewdata(age, height, kg, gender, power=1):
	global aimodel
	for _ in range(power):
		aimodel.add(age,height,kg,gender)

def trainmodel(epoch):
	global aimodel
	aimodel.train(epoch)

def guess(age,height,kg,gender):
	global aimodel
	return aimodel.guess(age,height,kg,gender)

aimodel = None
load()

def run_cli():
    global aimodel
    if len(sys.argv) == 1:
        print("\n--- BodyMetrics CLI Help ---")
        print("Usage: python script.py [command] [options]")
        print("Commands:")
        print("  new    [--learning=0.05] [--maxage=100] [--no-pretrain]")
        print("  add    [age] [height] [kg] [gender: 0 or 1]")
        print("  train  [epochs]")
        print("  guess  [age] [height] [kg] [gender: 0, 1, or 0.5]")
        print("  save/load")
        return
    cmd = sys.argv[1].lower()
    if cmd == "new":
        learning, truely, momentum = 0.05, 1.0, 0.9
        maxage, maxheight, maxkg = 100, 200, 150
        pretrain, pretrainepoch = True, 32
        for arg in sys.argv[2:]:
            if "=" in arg:
                key, val = arg.replace("--","").split("=")
                if key == "learning": learning = float(val)
                elif key == "maxage": maxage = int(val)
            if arg == "--no-pretrain": pretrain = False
        createnewmodel(learning, truely, momentum, maxage, maxheight, maxkg, pretrain, pretrainepoch=pretrainepoch)
        print("Success: New model initialized.")
    elif cmd == "add" and len(sys.argv) >= 6:
        aimodel.add_data(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
        save()
        print(f"Data Added: Age {sys.argv[2]}, Gender {sys.argv[5]}")
    elif cmd == "train" and len(sys.argv) >= 3:
        aimodel.train(epoch=int(sys.argv[2]))
        save()
        print(f"Training complete for {sys.argv[2]} epochs.")
    elif cmd == "guess" and len(sys.argv) >= 6:
        a, h, w, g_text = aimodel.guess(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
        print(f"\nPrediction Results ({g_text}):")
        print(f"Age: {a} | Height: {h}cm | Weight: {w}kg")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BodyMetrics Regression System | Human Autoencoder")
        self.geometry("900x500")
        self.configure(bg="#f0f2f5")
        style = ttk.Style()
        style.theme_use('clam')
        self.create_main_screen()
    def create_main_screen(self):
        for w in self.winfo_children(): w.destroy()
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill="both", expand=True)
        ttk.Label(main_frame, text="BodyMetrics Regression/Autoencoder Engine", font=("Segoe UI", 18, "bold")).pack(pady=(0,20))
        input_card = ttk.LabelFrame(main_frame, text=" Input Parameters ", padding=15)
        input_card.pack(fill="x", pady=10)
        self.sliders = {}
        for name, maxv in [("Age", aimodel.maxage), ("Height", aimodel.maxheight), ("Weight", aimodel.maxkg)]:
            row = ttk.Frame(input_card)
            row.pack(fill="x", pady=5)
            ttk.Label(row, text=f"{name}:", width=12).pack(side="left")
            s = ttk.Scale(row, from_=0, to=maxv, orient="horizontal")
            s.pack(side="left", fill="x", expand=True, padx=10)
            l = ttk.Label(row, text="0", width=5)
            l.pack(side="left")
            s.config(command=lambda v, lbl=l: lbl.config(text=f"{int(float(v))}"))
            self.sliders[name.lower()] = s
        gender_row = ttk.Frame(input_card)
        gender_row.pack(fill="x", pady=10)
        ttk.Label(gender_row, text="Gender:", width=12).pack(side="left")
        self.gender_var = tk.DoubleVar(value=0.5)
        ttk.Radiobutton(gender_row, text="Male", variable=self.gender_var, value=0.0).pack(side="left", padx=15)
        ttk.Radiobutton(gender_row, text="Blank (Predict)", variable=self.gender_var, value=0.5).pack(side="left", padx=15)
        ttk.Radiobutton(gender_row, text="Female", variable=self.gender_var, value=1.0).pack(side="left", padx=15)
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=20, fill="x")
        buttons = [
            ("Predict", self.predict),
            ("BMI Analysis", self.bmi_analysis),
            ("Random Bio", self.random_gen),
            ("Add Data", self.add_data_window),
            ("Train", self.train_ui),
            ("Review", self.review_window),
            ("Clear Data", self.reset_data),
            ("New Model", self.new_model_window)
        ]
        for text, cmd in buttons:
            ttk.Button(btn_frame, text=text, command=cmd).pack(side="left", expand=True, padx=5)
        self.res_card = ttk.LabelFrame(main_frame, text=" AI Inference Output ", padding=15)
        self.res_card.pack(fill="both", expand=True)
        self.res_label = ttk.Label(self.res_card, text="Waiting for input...", font=("Consolas", 11))
        self.res_label.pack()
    def predict(self):
        try:
            age, h, w = self.sliders["age"].get(), self.sliders["height"].get(), self.sliders["weight"].get()
            pa, ph, pw, g_text = aimodel.guess(age, h, w, self.gender_var.get())
            self.res_label.config(text=f"Model: {g_text}\nPredicted State: {pa}y, {ph}cm, {pw}kg")
        except Exception as e: messagebox.showerror("Error", str(e))
    def add_data_window(self):
        win = tk.Toplevel(self)
        win.title("Add Training Data")
        win.geometry("300x300")
        f = ttk.Frame(win, padding=20)
        f.pack(fill="both")
        entries = {}
        for lbl in ["Age", "Height", "Weight"]:
            ttk.Label(f, text=f"{lbl}:").pack(anchor="w")
            e = ttk.Entry(f); e.pack(fill="x", pady=(0, 10))
            entries[lbl.lower()] = e
        g_var = tk.DoubleVar(value=0)
        ttk.Radiobutton(f, text="Male", variable=g_var, value=0).pack(anchor="w")
        ttk.Radiobutton(f, text="Female", variable=g_var, value=1).pack(anchor="w")
        def submit():
            try:
                aimodel.add_data(int(entries["age"].get()), float(entries["height"].get()), float(entries["weight"].get()), g_var.get())
                save(); messagebox.showinfo("Done", "Data saved.")
            except: messagebox.showerror("Error", "Invalid Input")
        ttk.Button(f, text="Save Data", command=submit).pack(pady=20, fill="x")
    def train_ui(self):
        epochs = simpledialog.askinteger("Train", "Number of Epochs:", initialvalue=40)
        if epochs:
            aimodel.train(epoch=epochs)
            save(); messagebox.showinfo("Success", f"Trained for {epochs} epochs.")
    def review_window(self):
        win = tk.Toplevel(self)
        win.title("Comparative Model Review & Accuracy Analysis")
        win.geometry("1000x900")
        data = np.array(aimodel.data)
        accuracy_score = 0
        total_error_pct = 0
        if len(data) > 0:
            errors = []
            for d in aimodel.data:
                age_act = d[0] * aimodel.maxage
                h_act = d[1] * aimodel.maxheight
                w_act = d[2] * aimodel.maxkg
                gen_act = d[3]
                _, h_pred, w_pred, _ = aimodel.guess(age_act, 0, 0, gen_act)
                h_err = abs(h_act - h_pred) / h_act if h_act > 0 else 0
                w_err = abs(w_act - w_pred) / w_act if w_act > 0 else 0
                errors.append((h_err + w_err) / 2)
            mean_error = np.mean(errors)
            accuracy_score = max(0, 100 * (1 - mean_error))
        fig, (ax_h, ax_w) = plt.subplots(2, 1, figsize=(10, 8))
        fig.tight_layout(pad=6.0)
        fig.suptitle(f"Model Performance Analysis\nOverall Accuracy: {accuracy_score:.2f}%", fontsize=14, fontweight='bold', color='#2c3e50')
        ages = np.linspace(0, aimodel.maxage, 50)
        m_h, f_h, m_w, f_w = [], [], [], []
        for a in ages:
            _, h, w, _ = aimodel.guess(a, 0, 0, 0); m_h.append(h); m_w.append(w)
            _, h, w, _ = aimodel.guess(a, 0, 0, 1); f_h.append(h); f_w.append(w)
        ax_h.plot(ages, m_h, color='dodgerblue', label="Male AI Trend", linewidth=2.5)
        ax_h.plot(ages, f_h, color='crimson', label="Female AI Trend", linewidth=2.5)
        if len(data) > 0:
            m_pts = data[data[:, 3] == 0]
            f_pts = data[data[:, 3] == 1]
            ax_h.scatter(m_pts[:,0]*aimodel.maxage, m_pts[:,1]*aimodel.maxheight, c='dodgerblue', alpha=0.15, s=15, label="Male Training Data")
            ax_h.scatter(f_pts[:,0]*aimodel.maxage, f_pts[:,1]*aimodel.maxheight, c='crimson', alpha=0.15, s=15, label="Female Training Data")
        ax_h.set_title("Age vs Height (Regression Convergence)", fontsize=11, loc='left')
        ax_h.set_ylabel("Height (cm)")
        ax_h.legend(loc='lower right', fontsize='small')
        ax_h.grid(True, linestyle='--', alpha=0.5)
        ax_w.plot(ages, m_w, color='dodgerblue', linestyle='--', label="Male AI Trend", linewidth=2.5)
        ax_w.plot(ages, f_w, color='crimson', linestyle='--', label="Female AI Trend", linewidth=2.5)
        if len(data) > 0:
            ax_w.scatter(m_pts[:,0]*aimodel.maxage, m_pts[:,2]*aimodel.maxkg, c='dodgerblue', alpha=0.15, s=15)
            ax_w.scatter(f_pts[:,0]*aimodel.maxage, f_pts[:,2]*aimodel.maxkg, c='crimson', alpha=0.15, s=15)
        ax_w.set_title("Age vs Weight (Regression Convergence)", fontsize=11, loc='left')
        ax_w.set_ylabel("Weight (kg)")
        ax_w.set_xlabel("Age (Years)")
        ax_w.legend(loc='lower right', fontsize='small')
        ax_w.grid(True, linestyle='--', alpha=0.5)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        info_frame = ttk.Frame(win, padding=5)
        info_frame.pack(fill="x")
        status_text = f"Analyzed {len(data)} samples | Reliability: {'High' if accuracy_score > 90 else 'Medium' if accuracy_score > 75 else 'Low'}"
        ttk.Label(info_frame, text=status_text, font=("Segoe UI", 9, "italic")).pack(side="left")
        ttk.Label(info_frame, text=f"MAPE: {100-accuracy_score:.2f}%", font=("Segoe UI", 9, "bold")).pack(side="right")
    def new_model_window(self):
        win = tk.Toplevel(self)
        win.title("Create New AI Model")
        win.geometry("900x750")
        win.resizable(True, True)
        win.grab_set()
        f = ttk.Frame(win, padding="20 20 20 20")
        f.pack(fill="both", expand=True)
        ttk.Label(f, text="Brain Designer", font=("Segoe UI", 16, "bold")).pack(pady=(0, 20))
        tab_control = ttk.Notebook(f)
        body_tab = ttk.Frame(tab_control)
        gender_tab = ttk.Frame(tab_control)
        tab_control.add(body_tab, text='Body Models (Male/Female)')
        tab_control.add(gender_tab, text='Gender Model')
        tab_control.pack(fill="both", expand=True, pady=10)
        body_lobes_default = [[3, 12], [12, 6, 3]]
        body_lobesset_default = [[1.8, 1.6], [0.8, 0.6]]
        gender_lobes_default = [[3, 12], [12, 4, 1]]
        gender_lobesset_default = [[1.8, 1.6], [0.8, 0.6]]
        body_lobe_data = []
        gender_lobe_data = []
        def add_lobe(parent, lobe_data, initial_layers=[], initial_rates=[1.8, 1.6]):
            lobe_index = len(lobe_data)
            lobe_frame = ttk.Frame(parent)
            lobe_frame.pack(fill="x", pady=5)
            ttk.Label(lobe_frame, text=f"Lobe {lobe_index + 1}").pack(side="left", padx=5)
            trainrate_entry = ttk.Entry(lobe_frame, width=5)
            trainrate_entry.insert(0, str(initial_rates[0]))
            trainrate_entry.pack(side="left", padx=5)
            ttk.Label(lobe_frame, text="Train Rate").pack(side="left", padx=5)
            learnrate_entry = ttk.Entry(lobe_frame, width=5)
            learnrate_entry.insert(0, str(initial_rates[1]))
            learnrate_entry.pack(side="left", padx=5)
            ttk.Label(lobe_frame, text="Learn Rate").pack(side="left", padx=5)
            layer_entries = []
            def add_layer(initial=10):
                layer_entry = ttk.Entry(lobe_frame, width=5)
                layer_entry.insert(0, str(initial))
                layer_entry.pack(side="left", padx=5)
                def remove_this_layer():
                    layer_entry.destroy()
                    remove_btn.destroy()
                    if layer_entry in layer_entries:
                        layer_entries.remove(layer_entry)
                remove_btn = ttk.Button(lobe_frame, text="X", width=2, command=remove_this_layer)
                remove_btn.pack(side="left")
                layer_entries.append(layer_entry)
            add_layer_btn = ttk.Button(lobe_frame, text="+ Layer", command=lambda: add_layer(10))
            add_layer_btn.pack(side="left", padx=10)
            def remove_lobe():
                lobe_frame.destroy()
                if (trainrate_entry, learnrate_entry, layer_entries) in lobe_data:
                    lobe_data.remove((trainrate_entry, learnrate_entry, layer_entries))
            remove_lobe_btn = ttk.Button(lobe_frame, text="Remove Lobe", command=remove_lobe)
            remove_lobe_btn.pack(side="left", padx=5)
            for layer_size in initial_layers:
                add_layer(layer_size)
            lobe_data.append((trainrate_entry, learnrate_entry, layer_entries))
        add_body_lobe_btn = ttk.Button(body_tab, text="Add Lobe", command=lambda: add_lobe(body_tab, body_lobe_data))
        add_body_lobe_btn.pack(pady=10)
        add_gender_lobe_btn = ttk.Button(gender_tab, text="Add Lobe", command=lambda: add_lobe(gender_tab, gender_lobe_data))
        add_gender_lobe_btn.pack(pady=10)
        for i in range(len(body_lobes_default)):
            add_lobe(body_tab, body_lobe_data, body_lobes_default[i], body_lobesset_default[i])
        for i in range(len(gender_lobes_default)):
            add_lobe(gender_tab, gender_lobe_data, gender_lobes_default[i], gender_lobesset_default[i])
        ttk.Label(f, text="Model Configuration", font=("Segoe UI", 14, "bold")).pack(pady=(20, 10))
        vars = {
            "learning": tk.DoubleVar(value=0.05),
            "truely": tk.DoubleVar(value=1.0),
            "momentumexc": tk.DoubleVar(value=0.9),
            "maxage": tk.IntVar(value=100),
            "maxheight": tk.IntVar(value=200),
            "maxkg": tk.IntVar(value=150),
            "pretrain": tk.BooleanVar(value=True),
            "pretrainmaxage": tk.IntVar(value=101),
            "pretrainepoch": tk.IntVar(value=40),
        }
        param_frame = ttk.Frame(f)
        param_frame.pack(fill="x", pady=10)
        def create_param_row(label_text, var_key, from_val, to_val, row_idx, is_int=False):
            ttk.Label(param_frame, text=label_text).grid(row=row_idx, column=0, sticky="w", pady=5)
            scale = ttk.Scale(param_frame, from_=from_val, to=to_val, orient="horizontal", variable=vars[var_key])
            scale.grid(row=row_idx, column=1, sticky="ew", padx=10)
            val_lbl = ttk.Label(param_frame, text="", width=6)
            val_lbl.grid(row=row_idx, column=2, sticky="w")
            def update_label(*args):
                val = vars[var_key].get()
                val_lbl.config(text=f"{int(val)}" if is_int else f"{val:.3f}")
            vars[var_key].trace_add("write", update_label)
            update_label()
        params = [
            ("Learning Rate:", "learning", 0.001, 1, 0, False),
            ("Truely Factor:", "truely", 0.1, 1.0, 1, False),
            ("Momentum:", "momentumexc", 0.0, 1.0, 2, False),
            ("Max Age:", "maxage", 50, 120, 3, True),
            ("Max Height (cm):", "maxheight", 10, 250, 4, True),
            ("Max Weight (kg):", "maxkg", 10, 300, 5, True),
            ("Pre-train Epochs:", "pretrainepoch", 1, 500, 6, True),
            ("Pre-train Data Age:", "pretrainmaxage", 10, 100, 7, True),
        ]
        for p in params:
            create_param_row(*p)
        ttk.Label(param_frame, text="Enable Pre-training:").grid(row=8, column=0, sticky="w", pady=10)
        ttk.Checkbutton(param_frame, variable=vars["pretrain"]).grid(row=8, column=1, sticky="w")
        param_frame.columnconfigure(1, weight=1)
        def on_create():
            try:
                lobes = []
                lobesset = []
                for train_e, learn_e, layer_es in body_lobe_data:
                    rates = [float(train_e.get()), float(learn_e.get())]
                    layers = [int(e.get()) for e in layer_es]
                    if not layers:
                        raise ValueError("Each lobe must have at least one layer")
                    lobes.append(layers)
                    lobesset.append(rates)
                genderlobes = []
                genderlobesset = []
                for train_e, learn_e, layer_es in gender_lobe_data:
                    rates = [float(train_e.get()), float(learn_e.get())]
                    layers = [int(e.get()) for e in layer_es]
                    if not layers:
                        raise ValueError("Each lobe must have at least one layer")
                    genderlobes.append(layers)
                    genderlobesset.append(rates)
                createnewmodel(
                    lobes=lobes,
                    lobesset=lobesset,
                    genderlobes=genderlobes,
                    genderlobesset=genderlobesset,
                    learning=vars["learning"].get(),
                    truely=vars["truely"].get(),
                    momentumexc=vars["momentumexc"].get(),
                    maxage=vars["maxage"].get(),
                    maxheight=vars["maxheight"].get(),
                    maxkg=vars["maxkg"].get(),
                    pretrain=vars["pretrain"].get(),
                    pretrainmaxage=vars["pretrainmaxage"].get(),
                    pretrainepoch=vars["pretrainepoch"].get()
                )
                messagebox.showinfo("Success", "New AI Model created and pretrained successfully.")
                win.destroy()
                self.create_main_screen()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create model: {e}")
        btn_f = ttk.Frame(f)
        btn_f.pack(pady=30, fill="x")
        ttk.Button(btn_f, text="Cancel", command=win.destroy, width=15).pack(side="left", padx=5)
        ttk.Button(btn_f, text="Generate Model", command=on_create, width=20).pack(side="left", padx=5)
    def bmi_analysis(self):
        try:
            age, h, w = self.sliders["age"].get(), self.sliders["height"].get(), self.sliders["weight"].get()
            pa, ph, pw, g_text = aimodel.guess(age, h, w, self.gender_var.get())
            height_m = ph / 100
            if height_m > 0:
                bmi = pw / (height_m ** 2)
                category = ""
                if bmi < 18.5: category = "Underweight"
                elif bmi < 25: category = "Normal"
                elif bmi < 30: category = "Overweight"
                else: category = "Obese"
                messagebox.showinfo("BMI Analysis", f"Predicted BMI: {bmi:.1f}\nCategory: {category}")
            else:
                messagebox.showwarning("Error", "Height cannot be zero.")
        except Exception as e: messagebox.showerror("Error", str(e))
    def reset_data(self):
        if messagebox.askyesno("Confirm", "Delete all training data? (Model weights will remain)"):
            aimodel.data = []
            save()
            messagebox.showinfo("Success", "Data history cleared.")
    def random_gen(self):
        import random
        r_age = random.randint(1, aimodel.maxage)
        r_gender = random.choice([0.0, 1.0])
        self.sliders["age"].set(r_age)
        self.gender_var.set(r_gender)
        self.predict()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli()
    else:
        app = App()
        app.mainloop()
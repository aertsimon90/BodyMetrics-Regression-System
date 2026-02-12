# BodyMetrics Regression System

import zevihanthosa
import random
import json
import sys 
import tkinter as tk
from tkinter import ttk, messagebox

class System:
	def __init__(self, learning=0.05, truely=1, momentumexc=0.9, maxage=100, maxheight=200, maxkg=150):
		self.model = zevihanthosa.Brain(lobes=[[3,12],[12,3]], learning=learning, truely=truely, momentumexc=momentumexc)
		self.model.lobesset = [[1.8, 1.6], [0.8, 0.6]]
		self.data = []
		self.maxage = maxage
		self.maxheight = maxheight
		self.maxkg = maxkg
	def add_data(self, age, height, kg):
		if age == 0 or height == 0 or kg == 0:
			raise ValueError("Please use real values (not blank or zero)")
		age = min(age, self.maxage)/self.maxage
		height = min(height, self.maxheight)/self.maxheight
		kg = min(kg, self.maxkg)/self.maxkg
		self.data.append([age,height,kg])
	def train(self, epoch=32):
		for i in range(epoch):
			print(f"Training... {(i/epoch)*100:.0f}%")
			for data in self.data:
				age, height, kg = data
				for useage in range(2):
					for useheight in range(2):
						for usekg in range(2):
							if useage==0 and usekg==0 and useheight==0:
								continue
							age2 = age if useage==1 else 0
							height2 = height if useheight==1 else 0
							kg2 = kg if usekg==1 else 0
							self.model.process([age2,height2,kg2], [age,height,kg])
							self.model.process([age2,height2,kg2], [age,height,kg], naturalderiv=True)
	def guess(self, age, height, kg):
		age = min(age, self.maxage)/self.maxage
		height = min(height, self.maxheight)/self.maxheight
		kg = min(kg, self.maxkg)/self.maxkg
		ageoutput, heightoutput, kgoutput = self.model.process([age,height,kg], train=False)
		return int(ageoutput*self.maxage), int(heightoutput*self.maxheight), int(kgoutput*self.maxkg)
	def import_example_data(self, maxage=101):
		example_data = [[1, 74.6, 9.6], [2, 86.4, 12.4], [3, 95.1, 14.5], [4, 102.3, 16.8], [5, 108.3, 18.8], [6, 114.3, 20.8], [7, 119.5, 23.4], [8, 127.8, 25.5], [9, 132.3, 28.6], [10, 138.7, 32.4], [11, 146.3, 37.6], [12, 152.4, 42.1], [13, 160.0, 49.5], [14, 167.1, 55.3], [15, 173.3, 62.1], [16, 173.2, 64.9], [17, 174.5, 67.0], [18, 175.7, 70.0], [19, 176.9, 72.1], [20, 178.2, 74.5], [21, 179.3, 77.2], [22, 180.3, 79.6], [23, 181.2, 81.4], [24, 182.0, 83.3], [25, 183.1, 85.8], [26, 183.2, 87.8], [27, 183.2, 89.0], [28, 183.2, 90.7], [29, 183.2, 92.5], [30, 183.2, 94.3], [31, 183.2, 96.1], [32, 183.2, 97.9], [33, 183.2, 99.7], [34, 183.2, 101.5], [35, 183.2, 103.3], [36, 183.2, 105.1], [37, 183.2, 106.9], [38, 183.2, 108.7], [39, 183.2, 110.5], [40, 183.2, 112.3], [41, 174.8, 91.2], [42, 174.8, 92.2], [43, 174.8, 93.2], [44, 174.8, 94.2], [45, 174.8, 95.2], [46, 173.8, 94.2], [47, 173.8, 93.2], [48, 173.8, 92.2], [49, 173.8, 91.2], [50, 173.8, 90.2], [51, 175.0, 93.0], [52, 174.0, 93.0], [53, 174.0, 92.0], [54, 174.0, 92.0], [55, 174.0, 91.0], [56, 174.0, 91.0], [57, 174.0, 90.0], [58, 173.0, 90.0], [59, 173.0, 90.0], [60, 172.8, 84.0], [61, 172.0, 89.0], [62, 172.0, 89.0], [63, 172.0, 88.0], [64, 172.0, 88.0], [65, 171.0, 87.0], [66, 171.0, 87.0], [67, 171.0, 86.0], [68, 171.0, 86.0], [69, 170.0, 85.0], [70, 170.0, 82.5], [71, 170.0, 84.0], [72, 170.0, 83.0], [73, 169.0, 82.0], [74, 169.0, 82.0], [75, 169.0, 81.0], [76, 169.0, 81.0], [77, 168.0, 80.0], [78, 168.0, 79.0], [79, 168.0, 79.0], [80, 168.0, 78.0], [81, 167.0, 77.0], [82, 167.0, 76.0], [83, 167.0, 76.0], [84, 166.0, 75.0], [85, 166.0, 75.0], [86, 166.0, 74.0], [87, 166.0, 74.0], [88, 165.0, 73.0], [89, 165.0, 73.0], [90, 164.9, 73.1], [91, 164.0, 71.0], [92, 164.0, 70.0], [93, 163.0, 70.0], [94, 163.0, 69.0], [95, 162.0, 68.0], [96, 162.0, 68.0], [97, 161.0, 67.0], [98, 161.0, 66.0], [99, 160.0, 66.0], [100, 160.8, 67.4]]
		example_data = example_data[:maxage]
		random.shuffle(example_data)
		for data in example_data:
			self.add_data(data[0], data[1], data[2])

def save():
	global aimodel
	try:
		with open("model.json", "w") as f:
			f.write(json.dumps({"brain": zevihanthosa.save(aimodel.model), "data": aimodel.data, "maxes": [aimodel.maxage, aimodel.maxheight, aimodel.maxkg]}))
	except Exception as e:
		print("Error in saving model:",e)

def load(autocreate=True):
	global aimodel
	try:
		with open("model.json", "r") as f:
			aimodel = System()
			data = json.loads(f.read())
			aimodel.model = zevihanthosa.load(data["brain"])
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

def createnewmodel(learning=0.05, truely=1, momentumexc=0.9, maxage=100, maxheight=200, maxkg=150, pretrain=True, pretrainmaxage=101, pretrainepoch=32):
	global aimodel
	aimodel = System(learning=learning, truely=truely, momentumexc=momentumexc, maxage=maxage, maxheight=maxheight, maxkg=maxkg)
	if pretrain:
		aimodel.import_example_data(pretrainmaxage)
		aimodel.train(epoch=pretrainepoch)
	save()

def addnewdata(age, height, kg, power=1):
	global aimodel
	for _ in range(power):
		aimodel.add(age,height,kg)

def trainmodel(epoch):
	global aimodel
	aimodel.train(epoch)

def guess(age,height,kg):
	global aimodel
	return aimodel.guess(age,height,kg)

aimodel = None
load()

def run_cli():
    global aimodel
    if len(sys.argv) == 1:
        print("No arguments. Use: python script.py [command] [options]")
        print("Commands: new, add, train, guess, save, load")
        return
    cmd = sys.argv[1].lower()
    if cmd == "new":
        learning = 0.05
        truely = 1.0
        momentumexc = 0.9
        maxage = 100
        maxheight = 200
        maxkg = 150
        pretrain = True
        pretrainmaxage = 101
        pretrainepoch = 32
        i = 2
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg.startswith("--learning="):
                learning = float(arg.split("=",1)[1])
            elif arg.startswith("--truely="):
                truely = float(arg.split("=",1)[1])
            elif arg.startswith("--momentum="):
                momentumexc = float(arg.split("=",1)[1])
            elif arg.startswith("--maxage="):
                maxage = int(arg.split("=",1)[1])
            elif arg.startswith("--maxheight="):
                maxheight = int(arg.split("=",1)[1])
            elif arg.startswith("--maxkg="):
                maxkg = int(arg.split("=",1)[1])
            elif arg == "--no-pretrain":
                pretrain = False
            elif arg.startswith("--pretrainmaxage="):
                pretrainmaxage = int(arg.split("=",1)[1])
            elif arg.startswith("--pretrainepoch="):
                pretrainepoch = int(arg.split("=",1)[1])
            i += 1
        aimodel = System(
            learning=learning,
            truely=truely,
            momentumexc=momentumexc,
            maxage=maxage,
            maxheight=maxheight,
            maxkg=maxkg
        )
        if pretrain:
            aimodel.import_example_data(pretrainmaxage)
            aimodel.train(epoch=pretrainepoch)
        save()
        print("New model created.")
    elif cmd == "add" and len(sys.argv) >= 5:
        try:
            age = int(sys.argv[2])
            height = float(sys.argv[3])
            kg = float(sys.argv[4])
            aimodel.add_data(age, height, kg)
            save()
            print(f"Data added: {age}y, {height}cm, {kg}kg")
        except Exception as e:
            print("Add error:", e)
    elif cmd == "train" and len(sys.argv) >= 3:
        try:
            epochs = int(sys.argv[2])
            aimodel.train(epoch=epochs)
            save()
            print(f"Trained {epochs} epochs.")
        except Exception as e:
            print("Train error:", e)
    elif cmd == "guess" and len(sys.argv) >= 5:
        try:
            age = int(sys.argv[2])
            height = float(sys.argv[3])
            kg = float(sys.argv[4])
            a, h, w = aimodel.guess(age, height, kg)
            print(f"Prediction: age={a}, height={h}cm, weight={w}kg")
        except Exception as e:
            print("Guess error:", e)
    elif cmd == "save":
        save()
        print("Model manual saved.")
    else:
        print("Unknown command or missing arguments.")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Body Metrics")
        self.geometry("720x400")
        self.resizable(False, False)
        self.create_main_screen()
    def create_main_screen(self):
        for w in self.winfo_children():
            w.destroy()
        main = ttk.Frame(self, padding=16)
        main.pack(fill="both", expand=True)
        ttk.Label(main, text="Body Metrics Regression", font="Helvetica 15 bold").pack(pady=(12,20))
        sliders_frame = ttk.LabelFrame(main, text="Input", padding=10)
        sliders_frame.pack(fill="x", pady=8)
        self.sliders = {}
        for name, maxv, unit in [
            ("Age",    100, "years"),
            ("Height", 200, "cm"),
            ("Weight", 150, "kg")
        ]:
            row = ttk.Frame(sliders_frame)
            row.pack(fill="x", pady=5)
            ttk.Label(row, text=name, width=10).pack(side="left", padx=(0,8))
            scale = ttk.Scale(row, from_=0, to=maxv, orient="horizontal")
            scale.pack(side="left", fill="x", expand=True, padx=6)
            val = ttk.Label(row, text="0", width=6)
            val.pack(side="left", padx=(8,0))
            scale.config(command=lambda v, l=val: l.config(text=f"{int(float(v))}"))
            self.sliders[name.lower()] = scale
        btns = ttk.Frame(main)
        btns.pack(pady=20)
        ttk.Button(btns, text="Predict",   command=self.predict, width=14).pack(side="left", padx=6)
        ttk.Button(btns, text="New Model", command=self.new_model_window, width=14).pack(side="left", padx=6)
        ttk.Button(btns, text="Add Data",  command=self.add_data_window, width=14).pack(side="left", padx=6)
        ttk.Button(btns, text="Train",     command=self.train_window, width=14).pack(side="left", padx=6)
        ttk.Button(btns, text="Review",     command=self.review_window, width=14).pack(side="left", padx=6)
        self.result = ttk.Label(main, text="", font="Helvetica 11", wraplength=680)
        self.result.pack(pady=10, fill="x")
    def predict(self):
        try:
            age    = self.sliders["age"].get()
            height = self.sliders["height"].get()
            weight = self.sliders["weight"].get()

            pa, ph, pw = aimodel.guess(
                0 if age < 0.5 else int(age),
                0 if height < 0.5 else height,
                0 if weight < 0.5 else weight
            )
            lines = []
            if age < 0.5:
                lines.append(f"Age    : {pa} years (predicted)")
            else:
                lines.append(f"Age    : {int(age)} → {pa}")
            if height < 0.5:
                lines.append(f"Height : {ph} cm (predicted)")
            else:
                lines.append(f"Height : {int(height)} → {ph} cm")
            if weight < 0.5:
                lines.append(f"Weight : {pw} kg (predicted)")
            else:
                lines.append(f"Weight : {int(weight)} → {pw} kg")
            self.result.config(text="\n".join(lines), foreground="black")
        except Exception as e:
            self.result.config(text=f"Error: {e}", foreground="red")
    def new_model_window(self):
        win = tk.Toplevel(self)
        win.title("Create New Model")
        win.geometry("520x400")
        win.resizable(False, False)
        f = ttk.Frame(win, padding=16)
        f.pack(fill="both", expand=True)
        vars = {
            "learning": tk.DoubleVar(value=0.05),
            "truely": tk.DoubleVar(value=1.0),
            "momentumexc": tk.DoubleVar(value=0.9),
            "maxage": tk.IntVar(value=100),
            "maxheight": tk.IntVar(value=200),
            "maxkg": tk.IntVar(value=150),
            "pretrain": tk.BooleanVar(value=True),
            "pretrainmaxage": tk.IntVar(value=101),
            "pretrainepoch": tk.IntVar(value=32),
        }
        labels = [
            ("Learning rate", "learning", 0.001, 0.2, 0.001),
            ("Truely", "truely", 0.1, 2.0, 0.1),
            ("Momentum", "momentumexc", 0.0, 1.0, 0.01),
            ("Max Age", "maxage", 20, 150, 1),
            ("Max Height (cm)", "maxheight", 100, 250, 1),
            ("Max Weight (kg)", "maxkg", 50, 300, 1),
            ("Pre-train epochs", "pretrainepoch", 5, 200, 1),
            ("Pre-train max age", "pretrainmaxage", 20, 150, 1),
        ]
        for i, (txt, key, minv, maxv, step) in enumerate(labels):
            row = ttk.Frame(f)
            row.pack(fill="x", pady=5)
            ttk.Label(row, text=txt, width=18).pack(side="left")
            if key == "pretrain":
                ttk.Checkbutton(row, variable=vars["pretrain"]).pack(side="left", padx=10)
            else:
                scale = ttk.Scale(row, from_=minv, to=maxv, orient="horizontal", variable=vars[key], length=220)
                scale.pack(side="left", fill="x", expand=True, padx=8)
                val_lbl = ttk.Label(row, text=f"{vars[key].get():.2f}" if "." in str(vars[key].get()) else str(vars[key].get()), width=8)
                val_lbl.pack(side="left")
                def upd(v, lbl=val_lbl, var=vars[key]):
                    if "." in str(var.get()):
                        lbl.config(text=f"{float(v):.3f}")
                    else:
                        lbl.config(text=str(int(float(v))))
                scale.config(command=upd)
        def create():
            try:
                global aimodel
                aimodel = System(
                    learning=vars["learning"].get(),
                    truely=vars["truely"].get(),
                    momentumexc=vars["momentumexc"].get(),
                    maxage=vars["maxage"].get(),
                    maxheight=vars["maxheight"].get(),
                    maxkg=vars["maxkg"].get()
                )
                if vars["pretrain"].get():
                    aimodel.import_example_data(vars["pretrainmaxage"].get())
                    aimodel.train(epoch=vars["pretrainepoch"].get())
                save()
                messagebox.showinfo("Success", "New model created.")
                win.destroy()
                self.create_main_screen()
            except Exception as e:
                messagebox.showerror("Error", str(e))
        ttk.Button(f, text="Create Model", command=create).pack(pady=20)
    def add_data_window(self):
        win = tk.Toplevel(self)
        win.title("Add Data")
        win.geometry("250x180")
        f = ttk.Frame(win, padding=20)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text="Age:").grid(row=0, column=0, sticky="w", pady=4)
        e_age = ttk.Entry(f)
        e_age.grid(row=0, column=1, pady=4)
        ttk.Label(f, text="Height (cm):").grid(row=1, column=0, sticky="w", pady=4)
        e_h = ttk.Entry(f)
        e_h.grid(row=1, column=1, pady=4)
        ttk.Label(f, text="Weight (kg):").grid(row=2, column=0, sticky="w", pady=4)
        e_w = ttk.Entry(f)
        e_w.grid(row=2, column=1, pady=4)
        def add():
            try:
                age = int(e_age.get())
                h = float(e_h.get())
                w = float(e_w.get())
                aimodel.add_data(age, h, w)
                save()
                messagebox.showinfo("OK", "Data added.")
            except Exception as ex:
                messagebox.showerror("Error", str(ex))
        ttk.Button(f, text="Add", command=add).grid(row=3, column=0, columnspan=2, pady=12)
    def train_window(self):
        win = tk.Toplevel(self)
        win.title("Train")
        win.geometry("200x120")
        f = ttk.Frame(win, padding=20)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text="Epochs:").grid(row=0, column=0, sticky="w", pady=6)
        e = ttk.Entry(f)
        e.insert(0, "40")
        e.grid(row=0, column=1, pady=6)
        def start():
            try:
                ep = int(e.get())
                aimodel.train(epoch=ep)
                save()
                messagebox.showinfo("Done", f"Trained {ep} epochs.")
                win.destroy()
            except Exception as ex:
                messagebox.showerror("Error", str(ex))
        ttk.Button(f, text="Start", command=start).grid(row=1, column=0, columnspan=2, pady=12)
    def review_window(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import numpy as np
        
        win = tk.Toplevel(self)
        win.title("Model Regression Analysis")
        win.geometry("900x600")
        ages = list(range(aimodel.maxage + 1))
        pred_heights = []
        pred_weights = []
        for a in ages:
            _, h, w = aimodel.guess(a, 0, 0)
            pred_heights.append(h)
            pred_weights.append(w)
        actual_data = np.array(aimodel.data)
        actual_ages = actual_data[:, 0] * aimodel.maxage
        actual_heights = actual_data[:, 1] * aimodel.maxheight
        actual_weights = actual_data[:, 2] * aimodel.maxkg
        total_error_pct = 0
        for d in aimodel.data:
            a_val, h_val, w_val = d[0]*aimodel.maxage, d[1]*aimodel.maxheight, d[2]*aimodel.maxkg
            _, gh, gw = aimodel.guess(a_val, 0, 0)
            h_err = abs(h_val - gh) / h_val if h_val != 0 else 0
            w_err = abs(w_val - gw) / w_val if w_val != 0 else 0
            total_error_pct += (h_err + w_err) / 2
        avg_error = (total_error_pct / len(aimodel.data)) * 100
        accuracy = max(0, 100 - avg_error)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Model Analysis (Overall Accuracy: {accuracy:.2f}%)", fontsize=14)
        ax1.plot(ages, pred_heights, color='blue', label='AI Prediction', linewidth=2)
        ax1.scatter(actual_ages, actual_heights, color='red', s=10, alpha=0.5, label='Training Data')
        ax1.set_title("Age vs Height")
        ax1.set_xlabel("Age")
        ax1.set_ylabel("Height (cm)")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax2.plot(ages, pred_weights, color='green', label='AI Prediction', linewidth=2)
        ax2.scatter(actual_ages, actual_weights, color='orange', s=10, alpha=0.5, label='Training Data')
        ax2.set_title("Age vs Weight")
        ax2.set_xlabel("Age")
        ax2.set_ylabel("Weight (kg)")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        info_text = f"Analyzed {len(aimodel.data)} data points. Mean Deviation: {avg_error:.2f}%"
        ttk.Label(win, text=info_text, font="Helvetica 10 italic").pack(pady=5)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli()
    else:
        app = App()
        app.mainloop()
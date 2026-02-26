import os
import json
import random
from flask import Flask, request, jsonify

try:
    import zevihanthosa
except ImportError:
    print("Error: zevihanthosa library not found.")

app = Flask(__name__)
app.secret_key = os.urandom(24)

ADMIN_PASSKEY = "admin123" 
aimodel = None

class System:
    def __init__(self, lobes=[[3,12],[12,6,3]], lobesset=[[1.8, 1.6], [0.8, 0.6]], genderlobes=[[3,12],[12,4,1]], genderlobesset=[[1.8, 1.6], [0.8, 0.6]], learning=0.05, truely=1, momentumexc=0.9, maxage=100, maxheight=200, maxkg=150):
        self.malemodel = zevihanthosa.Brain(lobes=lobes, learning=learning, truely=truely, momentumexc=momentumexc)
        self.femalemodel = zevihanthosa.Brain(lobes=lobes, learning=learning, truely=truely, momentumexc=momentumexc)
        self.gendermodel = zevihanthosa.Brain(lobes=genderlobes, learning=learning, truely=truely, momentumexc=momentumexc)
        self.malemodel.lobesset = lobesset
        self.femalemodel.lobesset = lobesset
        self.gendermodel.lobesset = genderlobesset
        self.data = []
        self.maxage, self.maxheight, self.maxkg = maxage, maxheight, maxkg
    def add_data(self, age, height, kg, gender):
        age_n = min(float(age), self.maxage)/self.maxage
        height_n = min(float(height), self.maxheight)/self.maxheight
        kg_n = min(float(kg), self.maxkg)/self.maxkg
        gender = max(min(float(gender), 1), 0)
        self.data.append([age_n, height_n, kg_n, gender])
    def train(self, epoch=40):
        for i in range(epoch):
            shuffleddata = self.data.copy()
            random.shuffle(shuffleddata)
            for data in shuffleddata:
                age, height, kg, gender = data
                for useage in range(2):
                    for useheight in range(2):
                        for usekg in range(2):
                            if useage == 0 and usekg == 0 and useheight == 0: continue
                            a2, h2, k2 = (age if useage == 1 else 0), (height if useheight == 1 else 0), (kg if usekg == 1 else 0)
                            self.gendermodel.process([a2, h2, k2], [gender])
                            self.gendermodel.process([a2, h2, k2], [gender], naturalderiv=True)
                            target_model = self.femalemodel if gender > 0.5 else self.malemodel
                            target_model.process([a2, h2, k2], [age, height, kg])
                            target_model.process([a2, h2, k2], [age, height, kg], naturalderiv=True)
    def guess(self, age, height, kg, gender):
        age_n, h_n, k_n = float(age)/self.maxage, float(height)/self.maxheight, float(kg)/self.maxkg
        current_gender = float(gender)
        if current_gender == 0.5:
            current_gender = self.gendermodel.process([age_n, h_n, k_n], train=False)[0]
        model = self.femalemodel if current_gender > 0.5 else self.malemodel
        ao, ho, ko = model.process([age_n, h_n, k_n], train=False)
        return int(ao*self.maxage), int(ho*self.maxheight), int(ko*self.maxkg), current_gender


def save_model():
    with open("model.json", "w") as f:
        data = {
            "malemodel": zevihanthosa.save(aimodel.malemodel),
            "femalemodel": zevihanthosa.save(aimodel.femalemodel),
            "gendermodel": zevihanthosa.save(aimodel.gendermodel),
            "data": aimodel.data,
            "maxes": [aimodel.maxage, aimodel.maxheight, aimodel.maxkg]
        }
        f.write(json.dumps(data))

def load_model():
    global aimodel
    try:
        if os.path.exists("model.json"):
            with open("model.json", "r") as f:
                data = json.loads(f.read())
                aimodel = System(maxage=data["maxes"][0], maxheight=data["maxes"][1], maxkg=data["maxes"][2])
                aimodel.malemodel = zevihanthosa.load(data["malemodel"])
                aimodel.femalemodel = zevihanthosa.load(data["femalemodel"])
                aimodel.gendermodel = zevihanthosa.load(data["gendermodel"])
                aimodel.data = data["data"]
                print("Model loaded from model.json")
        else:
            aimodel = System()
            print("New system initialized.")
    except Exception as e:
        print(f"Loading error: {e}")
        aimodel = System()

@app.route('/api/predict', methods=['POST'])
def api_predict():
    req = request.json
    try:
        pa, ph, pw, conf = aimodel.guess(req['age'], req['height'], req['weight'], req['gender'])
        return jsonify({
            "age": pa, "height": ph, "weight": pw, 
            "male_conf": round((1-conf)*100, 1), 
            "female_conf": round(conf*100, 1)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/secure-action', methods=['POST'])
def secure_action():
    req = request.json
    if req.get('key') != ADMIN_PASSKEY:
        return jsonify({"error": "Invalid Security Key"}), 403
    
    action = req.get('action')
    try:
        if action == "train":
            epochs = int(req.get('epochs', 40))
            aimodel.train(epoch=epochs)
            save_model()
            return jsonify({"message": f"Successfully trained for {epochs} epochs."})
        elif action == "add":
            aimodel.add_data(req['age'], req['height'], req['weight'], req['gender'])
            save_model()
            return jsonify({"message": "Data point recorded successfully."})
        elif action == "reset":
            aimodel.data = []
            save_model()
            return jsonify({"message": "Training data history wiped."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BodyMetrics AI | Neural Regression System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Space Grotesk', sans-serif; background-color: #060910; color: #f1f5f9; }
        .glass { background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.08); }
        .input-accent { border-color: rgba(59, 130, 246, 0.3); }
        .input-accent:focus { border-color: #3b82f6; ring-color: #3b82f6; }
        .predict-mode { border: 2px dashed #3b82f6 !important; background: rgba(59, 130, 246, 0.05); }
    </style>
</head>
<body class="p-6 md:p-12 min-h-screen">
    <div class="max-w-6xl mx-auto">
        <div class="flex flex-col md:flex-row justify-between items-center mb-12 gap-6">
            <div>
                <h1 class="text-5xl font-bold bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400 bg-clip-text text-transparent">BodyMetrics AI</h1>
                <p class="text-slate-500 mt-2 font-light">Neural Network Body Data Autoencoder & Completer</p>
            </div>
            <div class="flex gap-4">
                <button onclick="openModal('adminModal')" class="flex items-center gap-2 px-6 py-3 bg-slate-900 border border-slate-800 rounded-xl hover:bg-slate-800 transition">
                    <i data-lucide="settings" class="w-4 h-4"></i> Admin Panel
                </button>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-10">
            <div class="lg:col-span-1 glass p-8 rounded-3xl space-y-8 h-fit">
                <h2 class="text-xl font-semibold flex items-center gap-2 pb-4 border-b border-slate-800">
                    <i data-lucide="layers" class="text-blue-500"></i> Parameters
                </h2>

                <div class="space-y-4">
                    <div class="flex justify-between items-end">
                        <label class="text-sm font-medium opacity-70 uppercase tracking-widest">Age (0-100)</label>
                        <span id="ageVal" class="text-lg font-bold text-blue-400">0 (PREDICT)</span>
                    </div>
                    <input type="range" id="age" min="0" max="100" value="0" step="1" oninput="updateUI('age')" class="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer">
                </div>

                <div class="space-y-4">
                    <div class="flex justify-between items-end">
                        <label class="text-sm font-medium opacity-70 uppercase tracking-widest">Height (0-200 cm)</label>
                        <span id="heightVal" class="text-lg font-bold text-blue-400">0 (PREDICT)</span>
                    </div>
                    <input type="range" id="height" min="0" max="200" value="0" step="1" oninput="updateUI('height')" class="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer">
                </div>

                <div class="space-y-4">
                    <div class="flex justify-between items-end">
                        <label class="text-sm font-medium opacity-70 uppercase tracking-widest">Weight (0-150 kg)</label>
                        <span id="weightVal" class="text-lg font-bold text-blue-400">0 (PREDICT)</span>
                    </div>
                    <input type="range" id="weight" min="0" max="150" value="0" step="1" oninput="updateUI('weight')" class="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer">
                </div>

                <div class="space-y-4">
                    <label class="text-sm font-medium opacity-70 uppercase tracking-widest">Gender Orientation</label>
                    <div class="grid grid-cols-3 gap-2">
                        <button onclick="setGender(0)" id="g0" class="g-btn py-3 rounded-xl bg-slate-900 border border-slate-800 text-xs font-bold transition uppercase">Male</button>
                        <button onclick="setGender(0.5)" id="g05" class="g-btn py-3 rounded-xl bg-blue-600 border border-blue-400 text-xs font-bold transition uppercase shadow-lg shadow-blue-500/20">Auto</button>
                        <button onclick="setGender(1)" id="g1" class="g-btn py-3 rounded-xl bg-slate-900 border border-slate-800 text-xs font-bold transition uppercase">Female</button>
                    </div>
                    <input type="hidden" id="gender" value="0.5">
                </div>

                <button onclick="runInference()" class="w-full py-5 bg-gradient-to-br from-blue-600 to-indigo-700 rounded-2xl font-bold uppercase tracking-wider hover:from-blue-500 hover:to-indigo-600 transition-all transform active:scale-95 shadow-xl shadow-blue-500/10">
                    PREDICT DATA
                </button>
            </div>

            <div class="lg:col-span-2 space-y-6">
                <div class="glass p-10 rounded-3xl h-full flex flex-col justify-between border-blue-500/20 relative overflow-hidden">
                    <div class="absolute top-0 right-0 p-8 opacity-5">
                        <i data-lucide="network" class="w-64 h-64"></i>
                    </div>

                    <div>
                        <h2 class="text-2xl font-bold mb-10 flex items-center gap-3">
                            <i data-lucide="cpu" class="text-emerald-500"></i> AI Output Log
                        </h2>

                        <div id="outputDisplay" class="grid grid-cols-1 md:grid-cols-3 gap-6 opacity-30 transition-opacity duration-1000">
                            <div class="p-6 rounded-2xl bg-slate-900/50 border border-slate-800">
                                <span class="text-slate-500 text-xs uppercase block mb-2">Age Prediction</span>
                                <div class="text-4xl font-mono font-bold" id="resAge">--</div>
                            </div>
                            <div class="p-6 rounded-2xl bg-slate-900/50 border border-slate-800">
                                <span class="text-slate-500 text-xs uppercase block mb-2">Height Prediction</span>
                                <div class="text-4xl font-mono font-bold" id="resHeight">--</div>
                            </div>
                            <div class="p-6 rounded-2xl bg-slate-900/50 border border-slate-800">
                                <span class="text-slate-500 text-xs uppercase block mb-2">Weight Prediction</span>
                                <div class="text-4xl font-mono font-bold" id="resWeight">--</div>
                            </div>
                        </div>
                    </div>

                    <div class="mt-12 space-y-6">
                        <div class="flex justify-between items-center px-2">
                            <div class="flex items-center gap-2">
                                <div class="w-3 h-3 rounded-full bg-blue-500"></div>
                                <span class="text-sm font-bold tracking-widest text-blue-400">MALE BIAS</span>
                            </div>
                            <div class="flex items-center gap-2">
                                <span class="text-sm font-bold tracking-widest text-pink-400">FEMALE BIAS</span>
                                <div class="w-3 h-3 rounded-full bg-pink-500"></div>
                            </div>
                        </div>
                        
                        <div class="w-full h-4 bg-slate-900 rounded-full overflow-hidden flex border border-slate-800 shadow-inner">
                            <div id="maleBar" class="h-full bg-gradient-to-r from-blue-700 to-blue-500 transition-all duration-1000" style="width: 50%"></div>
                            <div id="femaleBar" class="h-full bg-gradient-to-l from-pink-700 to-pink-500 transition-all duration-1000" style="width: 50%"></div>
                        </div>

                        <div class="flex justify-between text-xs font-mono text-slate-500">
                            <span id="malePerc">50.0% Probability</span>
                            <span id="femalePerc">50.0% Probability</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div id="adminModal" class="fixed inset-0 bg-black/90 hidden z-50 flex items-center justify-center p-6 backdrop-blur-sm">
        <div class="glass max-w-lg w-full p-10 rounded-3xl space-y-8 relative border-white/10 shadow-2xl">
            <button onclick="closeModal('adminModal')" class="absolute top-6 right-6 text-slate-500 hover:text-white">
                <i data-lucide="x"></i>
            </button>
            <h3 class="text-2xl font-bold flex items-center gap-3">
                <i data-lucide="shield-check" class="text-blue-500"></i> Admin Center
            </h3>
            
            <div class="space-y-4">
                <div class="space-y-2">
                    <label class="text-xs text-slate-400 uppercase">System Passkey</label>
                    <input type="password" id="adminKey" class="w-full bg-slate-900 p-4 rounded-xl border border-slate-800 outline-none focus:border-blue-500 transition shadow-inner text-white">
                </div>
                
                <div class="grid grid-cols-2 gap-4 pt-4">
                    <button onclick="handleSecure('train')" class="p-4 bg-slate-800 hover:bg-slate-700 rounded-xl transition flex flex-col items-center gap-2 border border-slate-700">
                        <i data-lucide="refresh-cw" class="w-5 h-5 text-blue-400"></i>
                        <span class="text-xs font-bold">RE-TRAIN ENGINE</span>
                    </button>
                    <button onclick="handleSecure('reset')" class="p-4 bg-slate-800 hover:bg-red-900/20 rounded-xl transition flex flex-col items-center gap-2 border border-slate-700">
                        <i data-lucide="trash-2" class="w-5 h-5 text-red-500"></i>
                        <span class="text-xs font-bold">WIPE DATA</span>
                    </button>
                </div>
                
                <div class="p-6 bg-blue-500/5 rounded-2xl border border-blue-500/10 mt-4">
                    <h4 class="text-sm font-bold text-blue-400 mb-4">Add Ground-Truth Data</h4>
                    <div class="grid grid-cols-2 gap-3">
                        <input type="number" id="addAge" placeholder="Age" class="bg-slate-900 p-3 rounded-lg border border-slate-800 text-xs">
                        <input type="number" id="addHeight" placeholder="Height" class="bg-slate-900 p-3 rounded-lg border border-slate-800 text-xs">
                        <input type="number" id="addWeight" placeholder="Weight" class="bg-slate-900 p-3 rounded-lg border border-slate-800 text-xs">
                        <select id="addGender" class="bg-slate-900 p-3 rounded-lg border border-slate-800 text-xs">
                            <option value="0">Male</option>
                            <option value="1">Female</option>
                        </select>
                    </div>
                    <button onclick="handleSecure('add')" class="w-full mt-4 py-3 bg-blue-600 rounded-xl text-xs font-bold uppercase tracking-widest hover:bg-blue-500 transition">
                        Insert Data Record
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        lucide.createIcons();

        function updateUI(id) {
            const val = document.getElementById(id).value;
            const display = document.getElementById(id + 'Val');
            if (val == 0) {
                display.innerText = "0 (PREDICT)";
                display.classList.add('text-blue-400');
                display.classList.remove('text-slate-400');
            } else {
                display.innerText = val;
                display.classList.remove('text-blue-400');
                display.classList.add('text-slate-400');
            }
        }

        function setGender(val) {
            document.getElementById('gender').value = val;
            const btns = document.querySelectorAll('.g-btn');
            btns.forEach(b => {
                b.classList.remove('bg-blue-600', 'border-blue-400', 'shadow-lg', 'shadow-blue-500/20');
                b.classList.add('bg-slate-900', 'border-slate-800');
            });
            const targetId = val == 0 ? 'g0' : (val == 0.5 ? 'g05' : 'g1');
            const target = document.getElementById(targetId);
            target.classList.add('bg-blue-600', 'border-blue-400', 'shadow-lg', 'shadow-blue-500/20');
            target.classList.remove('bg-slate-900', 'border-slate-800');
        }

        async function runInference() {
            const loader = document.getElementById('outputDisplay');
            loader.style.opacity = "0.2";
            
            const payload = {
                age: document.getElementById('age').value,
                height: document.getElementById('height').value,
                weight: document.getElementById('weight').value,
                gender: document.getElementById('gender').value
            };

            try {
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                
                loader.style.opacity = "1";
                document.getElementById('resAge').innerText = data.age;
                document.getElementById('resHeight').innerText = data.height + "cm";
                document.getElementById('resWeight').innerText = data.weight + "kg";
                
                document.getElementById('maleBar').style.width = data.male_conf + "%";
                document.getElementById('femaleBar').style.width = data.female_conf + "%";
                document.getElementById('malePerc').innerText = data.male_conf + "% Probability";
                document.getElementById('femalePerc').innerText = data.female_conf + "% Probability";
            } catch (e) {
                alert("Inference Error: Neural Engine Not Ready");
            }
        }

        async function handleSecure(action) {
            const key = document.getElementById('adminKey').value;
            const payload = { action, key };

            if(action === 'add') {
                payload.age = document.getElementById('addAge').value;
                payload.height = document.getElementById('addHeight').value;
                payload.weight = document.getElementById('addWeight').value;
                payload.gender = document.getElementById('addGender').value;
            }

            const res = await fetch('/api/secure-action', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if(data.error) alert("Access Denied: " + data.error);
            else {
                alert("System Update: " + data.message);
                if(action === 'train') closeModal('adminModal');
            }
        }

        function openModal(id) { document.getElementById(id).classList.remove('hidden'); }
        function closeModal(id) { document.getElementById(id).classList.add('hidden'); }
    </script>
</body>
</html>
"""

@app.route("/")
def main_path():
    return HTML_TEMPLATE

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)
"""BODYGUARD AI starter: train, test, and run a small local neural network."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from gehirn.neural_network import NeuralIntentNetwork
from training.train import train_model
from gehirn.tokenizer import tokenize

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "daten" / "trainingsdaten.json"
MODEL_PATH = ROOT / "speicher" / "modell.json"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modell fehlt. Bitte zuerst 'python main.py --train' ausführen.")
    return NeuralIntentNetwork.load(MODEL_PATH)

def main():
    parser = argparse.ArgumentParser(description="BODYGUARD AI - lokales, kleines neuronales Netz")
    parser.add_argument("--check", action="store_true", help="Projektstruktur und Trainingsdaten prüfen")
    parser.add_argument("--train", action="store_true", help="Netzwerk mit den enthaltenen Trainingsdaten trainieren")
    parser.add_argument("--predict", type=str, help="Text klassifizieren, z.B. --predict 'Hallo'")
    args = parser.parse_args()

    if args.check:
        with DATA_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data.get("examples", [])) >= 10
        assert len(data.get("intents", [])) >= 2
        print(f"BODYGUARD AI CHECK: OK — {len(data['examples'])} Trainingsbeispiele, {len(data['intents'])} Klassen.")
        return 0
    if args.train:
        result = train_model(DATA_PATH, MODEL_PATH)
        print(f"Training abgeschlossen. Beispiele: {result['examples']}, Klassen: {result['classes']}, Verlust: {result['loss']:.4f}")
        print(f"Modell gespeichert: {MODEL_PATH}")
        return 0
    if args.predict is not None:
        model = load_model()
        prediction = model.predict(args.predict)
        print(json.dumps(prediction, ensure_ascii=False, indent=2))
        return 0

    parser.print_help()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

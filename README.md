# BODYGUARD AI — offline neural-network starter

Version 0.2.0. This repository contains a **small, functioning feed-forward neural network** that learns to classify short German messages into eight intents. It is a foundation for experimentation, not a finished humanoid-robot brain or a general-purpose language model.

## Free and licensing
- No third-party Python packages are required; training and runtime use the Python standard library.
- Software code: MIT License.
- Included project-authored example dataset: CC0 1.0.
- No paid APIs, telemetry, automatic downloads, or internet access are required at runtime.
- GitHub Actions uses GitHub-hosted runners; availability/usage limits depend on GitHub's current plan and policy.
- Hardware, electricity, and internet used to download this repository are not cost-free by definition.
- Future model/data additions must have their own license and attribution checked.

## Requirements
Python 3.10 or newer. You can edit files on GitHub without installing Python, but to run locally you need Python. GitHub Actions can run the checks for you.

## Run
From the repository root:

```bash
python main.py --check
python main.py --train
python main.py --predict "Hallo, kannst du mir helfen?"
python -m unittest discover -s tests -v
```

The first training command creates `speicher/modell.json`. The included dataset has 46 small examples across 8 classes. It is deliberately small and only teaches basic intent labels. It does not teach object recognition, speech-to-text, robotics, or general knowledge.

## Start the local demo API
First train the model, then:

```bash
python -m server.app
```

Health check: `http://127.0.0.1:8765/health`

POST JSON to `http://127.0.0.1:8765/predict`:
```json
{"text":"Hallo"}
```

The server binds only to loopback (this PC) by default. Do not expose it to a LAN or the internet without device authentication, request limits, and secure transport. It does not accept movement commands or raw camera/audio data.

## Offline operation
After repository, Python, and any future model/data files have been downloaded, the program runs and trains without internet. GitHub Actions itself is a cloud service and is not offline; the eventual robot/server runtime is designed to be local.

## What is actually trained?
A one-hidden-layer neural network: token-presence vector → tanh hidden layer → softmax intent probabilities. It trains weights using backpropagation and cross-entropy-style gradients, then saves them to JSON. This is a real small neural network, but it is not comparable to a large language model.

## Safety and privacy
- `real_hardware_enabled` and movement control remain disabled.
- Never use this demo classifier to decide physical interventions or control motors.
- Keep emergency stop, motor limits, collision protection, and safe behavior during network loss on the robot itself.
- Raw audio/video is not stored or transmitted by this demo.
- Model confidence is not a safety guarantee.

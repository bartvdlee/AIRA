**Automated Impact on Rights Assessment (AIRA)**

AIRA is a tool to (partially) automate a Fundamental Rights Impact Assessment (FRIA) as required by the EU AI Act (AIA).

The tool assumes you have a LiteLLM instance running on http:localhost:4000 and have at least one LLM configured on LiteLLM.
For setting up LiteLLM, please see: https://github.com/BerriAI/litellm

**Features:**


**Installation (Ubuntu 24.04):**
Installation steps assume Python (3.12) and Git is installed on the system.

Clone the repo:
git clone https://github.com/bartvdlee/AIRA.git

Go into the repo folder:
cd AIRA

Make sure you have a virtual environment setup for the tool. The following command creates a virtual environment called AIRA-venv:
python3 -m venv AIRA-venv

This environment can be activated with the following command:
source AIRA-venv/bin/activate

The required packages can be installed by executing the following command:
pip install -r requirements.txt

The program can be started by running:
python run.py

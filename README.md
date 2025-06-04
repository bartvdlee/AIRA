# Automated Impact on Rights Assessment (AIRA)

AIRA is a tool to (partially) automate a Fundamental Rights Impact Assessment (FRIA) as required by the EU AI Act (AIA).

The tool assumes you have a LiteLLM instance running on `http://localhost:4000` and have at least one LLM configured on LiteLLM.
For setting up LiteLLM, please see: https://github.com/BerriAI/litellm

## Features
- Can use any LLM (local or through API) via LiteLLM
- Can generate potential human right impact with RAG
- Gives justifications for the generations
- Creation and local storage of (FAISS) vector store (**Important: never use an untrusted vector store as they can contain virusses**)
- Stores the context that was used when using RAG
- Ability to store multiple reports
- Can be resumed after an interrupted generation

## Installation (Ubuntu 24.04)
Installation steps assume Python (3.12) and Git is installed on the system.

**Clone the repo:**
```
git clone https://github.com/bartvdlee/AIRA.git
```

**Go into the repo folder:**
```
cd AIRA
```

Make sure you have a virtual environment setup for the tool. \
**The following command creates a virtual environment called AIRA-venv:**
```
python3 -m venv AIRA-venv
```

**This environment can be activated with the following command:**
```
source AIRA-venv/bin/activate
```

**The required packages can be installed by executing the following command:**
```
pip install -r requirements.txt
```

**Create a directory for storing the reports:**
```
mkdir reports
```

**Store the ECHR treaty text as ECHR.pdf:**
```
can e.g be found at https://www.echr.coe.int/documents/d/echr/Convention_ENG
```

**Add the ecthr_cases dataset (https://archive.org/details/ECtHR-NAACL2021/) used by the RAG to the current directory:**
```
git lfs install
git clone https://huggingface.co/datasets/AUEB-NLP/ecthr_cases
```

**Optional: create an environmental variable for your LiteLLM API key**
```
export OPENAI_API_KEY=[your API key]
```

**The program can be started by running:**
```
python run.py
```

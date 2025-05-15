"""
This class provides functionality to store and retrieve data at different
stages of the pipeline processing. Data is saved as JSON files in a dedicated
directory structure organized by report name.

Parameters
----------
report_name : str
    The name of the report, used to create a dedicated directory structure
    for storing the pipeline data.

Attributes
----------
report_dir : Path
    Path to the report directory.
data_dir : Path
    Path to the data directory within the report directory.

Methods
-------
save_data(stage, data)
    Save data for a specific pipeline stage to a JSON file.
load_data(stage)
    Load data from a specific pipeline stage JSON file.
data_exists(stage)
    Check if data exists for a specific pipeline stage.
list_data()
    List all available data files.
    
Examples
--------
>>> data_manager = DataManager("my_report")
>>> data_manager.save_data("extraction", {"text": "extracted content"})
>>> data = data_manager.load_data("extraction")
>>> data_manager.data_exists("extraction")
True
>>> data_manager.list_data()
['extraction']
"""


import json
from pathlib import Path
from typing import Any, List

class DataManager:
    """
    Manages data for the AIRA pipeline, allowing saving and loading results.
    """
    
    def __init__(self, report_name: str):
        self.report_dir = Path(f"reports/{report_name}")
        self.data_dir = self.report_dir / "data"
        
        # Create datas directory if it doesn't exist
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def save_data(self, stage: str, data: Any) -> None:
        """Save data for a specific pipeline stage to a data file."""
        data_file = self.data_dir / f"{stage}.json"
        
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"data saved: {stage}")
    
    def load_data(self, stage: str) -> Any:
        """Load data from a specific pipeline stage data."""
        data_file = self.data_dir / f"{stage}.json"
        
        if not data_file.exists():
            return None
            
        with open(data_file, 'r') as f:
            data = json.load(f)
            
        print(f"data loaded: {stage}")
        return data
    
    def data_exists(self, stage: str) -> bool:
        """Check if a data exists for a specific stage."""
        data_file = self.data_dir / f"{stage}.json"
        return data_file.exists()
    
    def list_data(self) -> List[str]:
        """List all available checkpoints."""
        if not self.data_dir.exists():
            return []
        
        return [file.stem for file in self.data_dir.glob("*.json")]
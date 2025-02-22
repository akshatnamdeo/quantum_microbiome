import os
import pathlib

def create_directory_structure(base_path):
    """
    Creates the directory structure for the quantum microbiome project.
    
    Args:
        base_path (str): The base path where the project structure will be created
    """
    # Create main project directory
    project_root = pathlib.Path(base_path)
    project_root.mkdir(exist_ok=True)
    
    # Define the directory structure
    directories = [
        'data/raw',
        'data/processed',
        'src/data_processing',
        'src/models',
        'src/utils'
    ]
    
    # Create directories
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
    
    # Define Python files to create
    python_files = {
        'src/data_processing': ['__init__.py', 'clean_data.py', 'merge_databases.py', 'preprocess.py'],
        'src/models': ['__init__.py', 'growth_predictor.py', 'interaction_network.py', 
                      'metabolite_lstm.py', 'pathway_transition.py', 'receptor_classifier.py'],
        'src/utils': ['__init__.py', 'data_loader.py', 'model_utils.py']
    }
    
    # Create Python files
    for directory, files in python_files.items():
        dir_path = project_root / directory
        for file in files:
            file_path = dir_path / file
            if not file_path.exists():
                file_path.touch()

def create_requirements_file(base_path):
    """
    Creates a requirements.txt file with basic dependencies.
    
    Args:
        base_path (str): The base path where requirements.txt will be created
    """
    requirements = [
        'numpy',
        'pandas',
        'scikit-learn',
        'torch',
        'networkx',
        'matplotlib',
        'seaborn',
        'pytest',
        'black',
        'flake8',
        'python-dotenv'
    ]
    
    with open(os.path.join(base_path, 'requirements.txt'), 'w') as f:
        f.write('\n'.join(requirements))

def create_readme(base_path):
    """
    Creates a README.md file with basic project information.
    
    Args:
        base_path (str): The base path where README.md will be created
    """
    readme_content = """# Quantum Microbiome Project

A computational framework for analyzing microbiome-gut-brain axis interactions using quantum-inspired algorithms.

## Project Structure

```
quantum_microbiome/
├── data/
│   ├── raw/            # Original, immutable data
│   └── processed/      # Cleaned and processed data
├── src/
│   ├── data_processing/
│   ├── models/
│   └── utils/
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

[Add usage instructions here]

## License

[Add license information here]
"""
    
    with open(os.path.join(base_path, 'README.md'), 'w') as f:
        f.write(readme_content)

def main():
    """
    Main function to create the project structure.
    """
    # Get the current directory
    current_dir = os.getcwd()
    project_path = os.path.join(current_dir, 'quantum_microbiome')
    
    # Create the project structure
    create_directory_structure(project_path)
    create_requirements_file(project_path)
    create_readme(project_path)
    
    print(f"Project structure created successfully at: {project_path}")

if __name__ == "__main__":
    main()
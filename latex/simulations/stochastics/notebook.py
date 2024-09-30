import os
import subprocess

# This script converts all Jupyter notebooks in a directory to PDF.

# pip install jupyter nbconvert -  ENSURE YOU HAVE THESE PACKAGES INSTALLED

# TO USE THIS TOOL - RUN THE FOLLOWING COMMAND IN THE TERMINAL - python notebook.py

def convert_notebook_to_pdf(notebook_path):
    """
    Convert a Jupyter notebook to PDF.
    
    Parameters:
    notebook_path (str): The path to the Jupyter notebook file.
    """
    try:
        # Construct the command to convert the notebook to PDF
        command = f"jupyter nbconvert --to pdf {notebook_path}"
        
        # Execute the command
        subprocess.run(command, shell=True, check=True)
        
        print(f"Successfully converted {notebook_path} to PDF.")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {notebook_path} to PDF: {e}")

def convert_all_notebooks_in_directory(directory_path):
    """
    Convert all Jupyter notebooks in a directory to PDF.
    
    Parameters:
    directory_path (str): The path to the directory containing Jupyter notebooks.
    """
    for filename in os.listdir(directory_path):
        if filename.endswith(".ipynb"):
            notebook_path = os.path.join(directory_path, filename)
            convert_notebook_to_pdf(notebook_path)

if __name__ == "__main__":
    # Specify the directory containing the Jupyter notebooks
    notebooks_directory = "./"
    
    # Convert all notebooks in the specified directory to PDF
    convert_all_notebooks_in_directory(notebooks_directory)
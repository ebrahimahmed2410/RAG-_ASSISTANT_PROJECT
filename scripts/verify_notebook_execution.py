"""
Programmatically executes all code cells in notebooks/rag_pipeline.ipynb
to verify that Kernel -> Restart & Run All completes without any errors.
"""

import json
import os
import sys

def verify_notebook():
    nb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks", "rag_pipeline.ipynb"))
    print(f"[*] Verifying notebook execution: {nb_path}")
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    print(f"[*] Found {len(code_cells)} code cells to execute.")
    
    # Execution namespace
    namespace = {}
    
    # Change working directory to notebooks/ folder, exactly as Jupyter does
    nb_dir = os.path.dirname(nb_path)
    orig_cwd = os.getcwd()
    os.chdir(nb_dir)
    
    try:
        for idx, cell in enumerate(code_cells, start=1):
            source_code = "".join(cell.get("source", []))
            # Filter out display() calls if IPython is not loaded
            cleaned_code = source_code.replace("display(", "print(")
            
            print(f"--- Executing Code Cell {idx}/{len(code_cells)} ---")
            try:
                exec(cleaned_code, namespace)
                print(f"[+] Cell {idx} executed successfully.")
            except Exception as cell_err:
                print(f"[-] Execution error in Cell {idx}: {cell_err}")
                raise cell_err
                
        print("\n[OK] All notebook code cells executed successfully from top to bottom!")
    finally:
        os.chdir(orig_cwd)

if __name__ == "__main__":
    verify_notebook()

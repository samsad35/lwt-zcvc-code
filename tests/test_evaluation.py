import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to python path so we can import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

def run_evaluations():
    """
    Simulation d'une suite d'évaluations.
    Ce script peut être étendu pour charger les vrais modèles et
    calculer des métriques sur un jeu de test complet.
    """
    print("--- Lancement des évaluations ---")
    
    # Création du dossier de sortie s'il n'existe pas
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulation de calcul de scores
    results = {
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "gcca_hubert_wavlm_score": 0.87,
            "gaussianity_average_hubert": 42.5,
            "gaussianity_average_wavlm": 68.3
        },
        "status": "success"
    }
    
    # Enregistrement des résultats
    output_file = output_dir / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"✅ Évaluations terminées. Résultats enregistrés dans : {output_file}")

if __name__ == "__main__":
    run_evaluations()

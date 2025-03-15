import time
import json
import os
import psutil
import numpy as np
from utils import eval_utils

class MetricsLogger:
    def __init__(self, args, model_name):
        self.start_time = time.time()
        self.args = args
        self.model_name = model_name
        self.metrics = {
            "model_name": model_name,
            "dataset": args.data,
            "workload_type": args.wl_type,
            "training_config": {
                "batch_size": args.batch_size,
                "epochs": args.n_epochs,
                "learning_rate": args.lr,
            },
            "performance": {
                "training_time": 0,
                "avg_query_exec_time": 0,
                "e2e_time": 0
            },
            "errors": {
                "p_error": {
                    "50th": 0,  # median
                    "90th": 0,
                    "95th": 0,
                    "99th": 0
                },
                "q_error": {
                    "50th": 0,  # median
                    "90th": 0,
                    "95th": 0,
                    "99th": 0
                }
            },
            "mwse_loss": 0,
            "resource_usage": {
                "cpu_percent": [],
                "memory_mb": []
            }
        }
        
    def update_p_error(self, p_error_test):
        """Update P-error using the same percentiles as calc_p_error.py"""
        n = len(p_error_test)
        ratios = [0.5, 0.9, 0.95, 0.99]
        
        for ratio, key in zip(ratios, ["50th", "90th", "95th", "99th"]):
            idx = int(n * ratio)
            self.metrics["errors"]["p_error"][key] = float(p_error_test[idx])

    def update_q_error(self, card_preds, true_cards):
        """Update Q-error using eval_utils.generic_calc_q_error"""
        q_error = eval_utils.generic_calc_q_error(card_preds, true_cards)
        q_error = np.sort(q_error)  # Sort for percentile calculation
        n = len(q_error)
        ratios = [0.5, 0.9, 0.95, 0.99]
        
        for ratio, key in zip(ratios, ["50th", "90th", "95th", "99th"]):
            idx = int(n * ratio)
            self.metrics["errors"]["q_error"][key] = float(q_error[idx])

    def save(self):
        """Save the metrics to a JSON file."""
        # Define the path where the metrics will be saved
        log_dir = self.args.experiments_dir
        log_file = os.path.join(log_dir, f"{self.model_name}_metrics.json")

        # Ensure the directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Write the metrics to a JSON file
        with open(log_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        print(f"Metrics saved to {log_file}")

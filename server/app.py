
# Ensure src is importable when running directly
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""Flask-based server for LNG-GeoEnv inference API"""

import json
import logging
from flask import Flask, jsonify
from src.lng_geoenv.runner import run_task

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)


@app.route("/reset", methods=["POST"])
def reset():
    return {"status": "ok"}, 200


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "lng-geoenv"}), 200


@app.route("/inference", methods=["GET"])
def inference():
    """Run all tasks and return results"""
    try:
        tasks = ["stable", "volatile", "war"]
        results = []

        for task in tasks:
            logger.info(f"Running task: {task}")
            result = run_task(task, seed=42)
            results.append(result)

        avg_score = sum(r["score"] for r in results) / len(results)

        response = {
            "environment": "lng-geoenv",
            "tasks": results,
            "average_score": avg_score,
            "execution_status": "success",
        }

        logger.info("Inference completed successfully")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        return jsonify({"error": str(e), "execution_status": "failed"}), 500


@app.route("/task/<task_name>", methods=["GET"])
def run_single_task(task_name):
    """Run a single task"""
    try:
        if task_name not in ["stable", "volatile", "war"]:
            return jsonify(
                {
                    "error": f"Unknown task: {task_name}. Valid options: stable, volatile, war"
                }
            ), 400

        logger.info(f"Running task: {task_name}")
        result = run_task(task_name, seed=42)

        return jsonify(
            {"task": task_name, "result": result, "execution_status": "success"}
        ), 200

    except Exception as e:
        logger.error(f"Task {task_name} failed: {str(e)}")
        return jsonify({"error": str(e), "execution_status": "failed"}), 500


@app.route("/", methods=["GET"])
def index():
    """API documentation"""
    return jsonify(
        {
            "service": "LNG-GeoEnv Inference Server",
            "version": "0.1.0",
            "endpoints": {
                "/health": "GET - Health check",
                "/inference": "GET - Run all tasks (stable, volatile, war)",
                "/task/<task_name>": "GET - Run a specific task",
                "/": "GET - This documentation",
            },
        }
    ), 200


def create_app():
    """Application factory"""
    return app


def main():
    """Entry point for server"""
    logger.info("Starting LNG-GeoEnv server on 0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()

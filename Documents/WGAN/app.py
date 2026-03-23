from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import threading
import os
from train import trainer

app = Flask(__name__, static_folder='.', static_url_path='')
# Enable CORS so our frontend can freely communicate
CORS(app)

training_thread = None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "is_training": trainer.is_training,
        "epoch": trainer.current_epoch,
        "total_epochs": trainer.total_epochs,
        "step": trainer.current_step,
        "total_steps": trainer.total_steps,
        "critic_loss": trainer.latest_loss_C,
        "generator_loss": trainer.latest_loss_G,
        "prob_real": trainer.latest_prob_real,
        "prob_fake": trainer.latest_prob_fake,
        "latest_samples": trainer.latest_samples_base64,
        "eta_seconds": trainer.eta_seconds,
        "elapsed_seconds": trainer.elapsed_seconds
    })

@app.route('/api/history', methods=['GET'])
def history():
    """Return full training history for charts"""
    return jsonify({
        "labels": trainer.history_labels,
        "critic_loss": trainer.loss_history_C,
        "generator_loss": trainer.loss_history_G,
        "prob_real": trainer.prob_history_real,
        "prob_fake": trainer.prob_history_fake
    })

@app.route('/api/start', methods=['POST'])
def start_training():
    global training_thread
    if not trainer.is_training:
        # Get epoch count from request, default to 100
        data = request.get_json(silent=True) or {}
        num_epochs = data.get('epochs', 100)
        num_epochs = max(1, min(1000, int(num_epochs)))  # Clamp between 1 and 1000
        
        training_thread = threading.Thread(target=trainer.train_loop, args=(num_epochs,))
        training_thread.start()
        return jsonify({"status": f"Training started for {num_epochs} epochs"}), 200
    return jsonify({"status": "Already training"}), 400

@app.route('/api/stop', methods=['POST'])
def stop_training():
    if trainer.is_training:
        trainer.stop_training()
        return jsonify({"status": "Stopping training..."}), 200
    return jsonify({"status": "Not currently training"}), 400

@app.route('/api/generate', methods=['GET'])
def generate_images():
    trainer.generate_sample_for_frontend()
    return jsonify({
        "image": trainer.latest_samples_base64
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

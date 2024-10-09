from flask import Flask, request, jsonify
import replicate
import os
import time

app = Flask(__name__)

# Get the tokens from environment variables
huggingface_token = os.getenv("HUGGINGFACE_API_TOKEN")
replicate_token = os.getenv("REPLICATE_API_TOKEN")

# Initialize Replicate with the token
replicate.Client(api_token=replicate_token)

@app.route('/initialise_model', methods=['POST'])
def initialise_model():
    uuid = request.json.get('uuid')
    if not uuid:
        return jsonify({"error": "UUID is required"}), 400

    model = replicate.models.create(
        owner="amirtens",
        name=uuid,
        visibility="private",  # or "public" if you prefer
        hardware="gpu-t4",  # Replicate will override this for fine-tuned models
        description="A fine-tuned FLUX.1 model"
    )
    
    model_name = model.name
    model_owner = model.owner
    model_url = f"https://replicate.com/{model_owner}/{model_name}"

    return jsonify({
        "model_name": model_name,
        "model_owner": model_owner,
        "model_url": model_url
    })


@app.route('/create_training', methods=['POST'])
def create_training():
    model_owner = request.form.get('model_owner')
    model_name = request.form.get('model_name')
    steps = request.form.get('steps', 1000)
    
    if not model_owner or not model_name or 'file' not in request.files:
        return jsonify({"error": "Model owner, model name, and input images are required"}), 400

    input_images_file = request.files['file']
    
    if not input_images_file:
        return jsonify({"error": "No file uploaded"}), 400

    # Save the uploaded file temporarily
    input_images_path = 'temp_input_images.zip'
    input_images_file.save(input_images_path)
    
    try:
        # Create the training
        training = replicate.trainings.create(
            version="ostris/flux-dev-lora-trainer:4ffd32160efd92e956d39c5338a9b8fbafca58e03f791f6d8011f3e20e8ea6fa",
            input={
                "input_images": open(input_images_path, "rb"),
                "steps": steps,
                "hf_token": huggingface_token,   # optional
            },
            destination=f"{model_owner}/{model_name}"
        )

        # Poll the training status until it's done
        training_id = training.id
        while True:
            status = replicate.trainings.get(training_id).status
            if status in ["completed", "failed"]:
                break
            time.sleep(10)  # Wait before checking again

        return jsonify({
            "training_status": status,
            "training_url": f"https://replicate.com/p/{training_id}"
        })

    finally:
        # Ensure the file is deleted after the training request is completed
        if os.path.exists(input_images_path):
            os.remove(input_images_path)

@app.route('/make_inference', methods=['POST'])
def make_inference():
    model_owner = request.json.get('model_owner')
    model_name = request.json.get('model_name')
    version_id = request.json.get('version_id')
    prompt = request.json.get('prompt')

    if not model_owner or not model_name or not version_id or not prompt:
        return jsonify({"error": "Model owner, model name, version ID, and prompt are required"}), 400

    output = replicate.run(
        f"{model_owner}/{model_name}:{version_id}",
        input={
            "prompt": prompt,
            "num_inference_steps": 28,
            "guidance_scale": 7.5,
            "model": "dev",
        }
    )

    return jsonify({
        "generated_image_url": output
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7008)

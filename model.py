import replicate
import os

#Getting the hugginface_token
huggingface_token = os.getenv("HUGGINGFACE_API_TOKEN")
replicate_token = os.getenv("REPLICATE_API_TOKEN")


def initialise_model(uuid):
    #Create a private, empty model on replicate
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
    print(f"Model created: {model_name}")
    print(f"Model url: {model_url}")
    print("The entire model object is ", model)
    #print(f"Model URL: https://replicate.com/{model_owner}/{model_name}")
    return model_name, model_owner, model_url
    

model_name, model_owner, model_url = initialise_model("12999")


def create_training(model_owner, model_name):

    # Now use this model as the destination for your training
    training = replicate.trainings.create(
        version="ostris/flux-dev-lora-trainer:4ffd32160efd92e956d39c5338a9b8fbafca58e03f791f6d8011f3e20e8ea6fa",
        input={
            "input_images": open("/path/to/your/local/training-images.zip", "rb"),
            "steps": 1000,
            "hf_token": huggingface_token,   # optional
        },
        destination=f"{model_owner}/{model_name}"
    )

    print(f"Training started: {training.status}")
    print(f"Training URL: https://replicate.com/p/{training.id}")
    return training.id


#For making inference
def make_inference(model_owner, model_name, version_id, prompt):

    output = replicate.run(
        f"{model_owner}/{model_name}:{version_id}",
        input={
            "prompt": prompt,
            "num_inference_steps": 28,
            "guidance_scale": 7.5,
            "model": "dev",
        }
    )

    print(f"Generated image URL: {output}")
    return output_url




#update the path for zip file
#Get the images on client, zip on client or server, save on aws and send url

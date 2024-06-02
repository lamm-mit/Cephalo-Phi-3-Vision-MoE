## Cephalo Mixture-of-Expert models based on Phi-3-Vision architecture

Cephalo is a series of multimodal materials science focused vision large language models (V-LLMs) designed to integrate visual and linguistic data for advanced understanding and interaction in human-AI or multi-agent AI frameworks. 

![image/png](https://cdn-uploads.huggingface.co/production/uploads/623ce1c6b66fedf374859fe7/kl5GWBP9WS0D4uwd1t3S7.png)

This repository explains how to create a Mixture-of-Expert model based on one or several Phi-3-Vision-128K models. The model architecture is as follows:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/623ce1c6b66fedf374859fe7/b7BK8ZtDzTMsyFDi0wP3w.png)

Model weights and examples are provided at: [https://huggingface.co/lamm-mit/Cephalo-Phi-3-MoE-vision-128k-3x4b-beta](https://huggingface.co/lamm-mit/Cephalo-Phi-3-MoE-vision-128k-3x4b-beta)

## Download MoE Model and Sample inference code

```markdown
pip install transformers -U
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor,AutoConfig  

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #number of parameters in b
    return total_params/1e9, trainable_params/1e9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name_moe = "lamm-mit/Cephalo-Phi-3-MoE-vision-128k-3x4b-beta"

processor = AutoProcessor.from_pretrained(model_name_moe, trust_remote_code=True) 
moe_model = AutoModelForCausalLM.from_pretrained(
    model_name_moe,
    trust_remote_code=True,  torch_dtype=torch.bfloat16,    
).to(device)
count_parameters(moe_model)
```

## Make a Phi-3-V-MoE model from scratch using several pre-trained models

You can either download .py files from this repository (see folder `./Phi_3V_MoE/') or download them directly from our Hugging Face repository. 

These codes implement the Phi-3-V and the Mixture-of-Expert Vision model from scratch. 

```markdown
pip install huggingface_hub
```

```python
from huggingface_hub import HfApi, hf_hub_download
from tqdm.notebook import tqdm
import os
import shutil

# Repository details
repo_id = "lamm-mit/Cephalo-Phi-3-MoE-vision-128k-3x4b-beta"
api = HfApi()

# List all files in the repository
files_in_repo = api.list_repo_files(repo_id)

# Filter for .py files
py_files = [file for file in files_in_repo if file.endswith('.py')]

# Directory to save the downloaded files
save_dir = "./Phi_3V_MoE/"
os.makedirs(save_dir, exist_ok=True)

# Download each .py file
for file_name in tqdm(py_files):
    file_path = hf_hub_download(repo_id=repo_id, filename=file_name)
    new_path = os.path.join(save_dir, file_name)
    shutil.move(file_path, new_path)
    print(f"Downloaded: {file_name}")

print("Download completed.")
```

Download models that will form the experts, as well as the base model 

```python
from Phi_3V_MoE.moe_phi3_v import Phi3VForCausalLMMoE, Phi3VForCausalLMMoEConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Model specialized in bio-inspired/mechanics and materials
model_name_1 = f"lamm-mit/Cephalo-Phi-3-vision-128k-4b-beta"
model_1 = AutoModelForCausalLM.from_pretrained(
    model_name_1,
    trust_remote_code=True,  torch_dtype=torch.bfloat16, 
    
).to(device)

#Original model
model_name_2 = f"microsoft/Phi-3-vision-128k-instruct"
model_2 = AutoModelForCausalLM.from_pretrained(
    model_name_2,
    trust_remote_code=True,  torch_dtype=torch.bfloat16, 
    
).to(device)

#Model trained on conversion of images to LaTeX formulas
model_name_3 = f"lamm-mit/Cephalo-LaTeX-Phi-3-vision-128k-4b-alpha"
model_3 = AutoModelForCausalLM.from_pretrained(
    model_name_3,
    trust_remote_code=True,  torch_dtype=torch.bfloat16, 
    
).to(device)

dtype = torch.bfloat16  # Desired dtype for new layers in MoE model

# Initialize the models
base_model = copy.deepcopy(model_2)  # Your base model
expert_models = [model_1, model_2,  model_3  ]  # List of expert models
 
# Load a processor (e.g. from base model)
processor = AutoProcessor.from_pretrained(model_name_2, trust_remote_code=True) 

# Create the config
config =  AutoConfig.from_pretrained(model_name_2, trust_remote_code=True)

# Create the MoE model
moe_config = Phi3VForCausalLMMoEConfig(config=config, k=1, num_expert_models=len (expert_models))
moe_model = Phi3VForCausalLMMoE(moe_config, base_model, expert_models,  layer_dtype = dtype).to(device)

count_parameters(expert_models[0]),count_parameters(moe_model)
```

### Training the gating networks

To train the gating networks, you need to provide sample prompts for each of the experts. Sample prompts consist of text and image data. You must match the number of experts you have, designed by k above. 

To get text data, you can use the processor/chat template:

```python
messages = [ {"role": "user", "content": "<|image_1|>\nWhat is shown in this image, and what is the relevance for materials design?"}, ]
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt
```

In the following example we show how training of the gating layers is done. The training set consists of images and prompt. The first item in the list are the prompts for expert 1, the second item the prompts for expert 2, and so on. 

Sample training set and process to train (for simplicity we use only three images, one characteristic of each expert):
```python
from PIL import Image
import requests

image_1 = Image.open(requests.get("https://d2r55xnwy6nx47.cloudfront.net/uploads/2018/02/Ants_Lede1300.jpg", stream=True).raw) 
image_2 = Image.open(requests.get("https://https://images.pexels.com/photos/106399/pexels-photo-106399.jpeg", stream=True).raw) 
image_3 = Image.open(requests.get("https://upload.wikimedia.org/wikipedia/commons/a/a0/Euplectella_aspergillum_Okeanos.jpg", stream=True).raw) 

prompts_per_expert = [
    [{"text": "<|user|>\n<|image_1|>\nPrompt 1 for expert 1<|end|>\n<|assistant|>\n", "image": [image_1]}, 
     {"text": "<|user|>\n<|image_1|>\nPrompt 2 for expert 1<|end|>\n<|assistant|>\n", "image": [image_1]}],

    [{"text": "<|user|>\n<|image_1|>\nPrompt 1 for expert 2<|end|>\n<|assistant|>\n", "image": [image_2]}, 
     {"text": "<|user|>\n<|image_1|>\nPrompt 2 for expert 2<|end|>\n<|assistant|>\n", "image": [image_2]}],

    [{"text": "<|user|>\n<|image_1|>\nPrompt 1 for expert 3<|end|>\n<|assistant|>\n", "image": [image_3]}, 
     {"text": "<|user|>\n<|image_1|>\nPrompt 2 for expert 3<|end|>\n<|assistant|>\n", "image": [image_3]}],
]

# Train gating layers using the provided prompts
gating_layer_params = moe_model.train_gating_layer_params_from_hidden_states(processor, prompts_per_expert,
                                              epochs=1000,
                                              loss_steps=100,
                                              lr=5e-5,
                                          )

# Set parameters
moe_model.set_gating_layer_params(gating_layer_params)
```

![image/png](https://cdn-uploads.huggingface.co/production/uploads/623ce1c6b66fedf374859fe7/xzZwBIw1yYr9v7xYblCNZ.png)

### Peparing gating network for full training

To freeze all parameters in the model except for the gating neural networks, you can use:

```python
freeze_except_gating_layers(moe_model)
count_parameters(moe_model)
```
You can unfreeze:
```python
un_freeze_all(moe_model)
```

Define FT_repo_id to push on HF hub/save model:
```
FT_repo_id='xxxxx/' #<repo_ID>
```

```
from datasets import load_dataset

train_dataset = load_dataset("lamm-mit/Cephalo-Wikipedia-Materials", split="train")
```

```python
import random

class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image = example["image"]
            question = example["query"] 
            answer = example["answer"]            
            messages = [ {
                            "role": "user",  "content": '<|image_1|>\n'+question},
                           {"role": "assistant", "content": f"{answer}"}, ]
                
            text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                
            images.append(image)
             
        batch = processor(text=text, images=[image], return_tensors="pt", padding=True
            
        labels = batch["input_ids"].clone() 
        labels[labels <0] = -100 

        batch["labels"] = labels

        return batch

data_collator = MyDataCollator(processor)
```
Then set up trainer, and train:
```python
from transformers import TrainingArguments, Trainer

optim = "paged_adamw_8bit"

training_args = TrainingArguments(
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=250,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_steps=25,
    output_dir="output_training",
    optim=optim,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=16,
    #fp16=True,
    bf16=True,  
    push_to_hub_model_id=FT_repo_id,
    remove_unused_columns=False,
    report_to="none",
)

trainer = Trainer(
    model=moe_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
```

## More details on inference 

### Chat Format

Given the nature of the training data, the Cephalo-Phi-3-vision-128k-4b-beta model is best suited for a single image input wih prompts using the chat format as follows. 
You can provide the prompt as a single image with a generic template as follow:
```raw
<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n 
```

where the model generates the text after `<|assistant|>` . For multi-turn conversations, the prompt should be formatted as follows:

```raw
<|user|>\n<|image_1|>\n{prompt_1}<|end|>\n<|assistant|>\n{response_1}<|end|>\n<|user|>\n{prompt_2}<|end|>\n<|assistant|>\n 
```

## Sample inference code

This code snippets show how to get quickly started on a GPU:

```python
from PIL import Image 
import requests
from transformers import AutoModelForCausalLM, AutoProcessor,AutoConfig  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name_moe = "lamm-mit/Cephalo-Phi-3-MoE-vision-128k-3x4b-beta"

processor = AutoProcessor.from_pretrained(model_name_moe, trust_remote_code=True) 
moe_model = AutoModelForCausalLM.from_pretrained(
    model_name_moe,
    trust_remote_code=True,  torch_dtype=torch.bfloat16,    
).to(device)

question = "What is shown in this image, and what is the relevance for materials design? Include a discussion of multi-agent AI."

messages = [ 
    {"role": "user", "content": f"<|image_1|>\n{question}"}, 
    ] 

url = "https://d2r55xnwy6nx47.cloudfront.net/uploads/2018/02/Ants_Lede1300.jpg" 

image = Image.open(requests.get(url, stream=True).raw) 

prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0") 

generation_args = { 
                    "max_new_tokens": 256, 
                    "temperature": 0.1, 
                    "do_sample": True, 
                    "stop_strings": ['<|end|>',
                                     '<|endoftext|>'],
                    "tokenizer": processor.tokenizer,
                  } 

generate_ids = moe_model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

# remove input tokens 
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

print(response) 
```
Sample output:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/623ce1c6b66fedf374859fe7/5n6oRNHrfwHkBX0QertZp.png)
<small>Image by [Vaishakh Manohar](https://www.quantamagazine.org/the-simple-algorithm-that-ants-use-to-build-bridges-20180226/)</small>

<pre style="white-space: pre-wrap;">
The image shows a group of red ants (Solenopsis invicta) climbing over a vertical wooden post. The ants are using their long legs and antennae to navigate the rough surface of the wood, demonstrating their ability to adapt to different materials and environments. This behavior is relevant for materials design because it highlights the importance of considering the interactions between materials and living organisms, such as ants, when designing new materials.

Multi-agent AI (Artificial Intelligence) is a field of study that focuses on the development of AI systems that can work together with other AI systems to achieve a common goal. In the context of this image, multi-agent AI could be used to design materials that are more compatible with the natural behaviors of living organisms, such as ants, and that can adapt to different environments and conditions.

By studying the behavior of ants and other living organisms, researchers can gain insights into how materials can be designed to better interact with these organisms and to better mimic their natural behaviors. This can lead to the development of new materials that are more sustainable, efficient, and effective in a variety of applications.

In summary, the image of red ants climbing over a wooden post highlights the importance of considering the interactions between materials and living organisms when designing new materials, and the potential of multi-agent AI to help achieve this goal.
</pre>

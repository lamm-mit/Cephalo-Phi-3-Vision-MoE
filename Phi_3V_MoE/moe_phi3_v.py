from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import copy

from .modeling_phi3_v import Phi3VForCausalLM, Phi3MLP
from .configuration_phi3_v import Phi3VConfig

from torch.optim import Adam
from typing import Optional, Tuple

from transformers import (
    PreTrainedModel,
    AutoConfig,
)

import torch.nn.functional as F

import matplotlib.pyplot as plt

# Define the Gating Layer
class GatingLayer(nn.Module):
    def __init__(self, input_dim, num_experts, k, layer_dtype=torch.float16):
        super(GatingLayer, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.gate = nn.Linear(input_dim, num_experts).to(dtype=layer_dtype)

    def forward(self, x):
        gate_scores = torch.softmax(self.gate(x), dim=-1)
        topk_values, topk_indices = torch.topk(gate_scores, self.k, dim=-1)
        topk_values = F.softmax(topk_values, dim=-1)

        return topk_values, topk_indices


class MoE(nn.Module):
    def __init__(self, input_dim, experts, gating_layer, config):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.gating_layer = gating_layer
        self.output_dim = config.hidden_size

    def forward(self, x):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            gate_values, gate_indices = self.gating_layer(x)
            batch_size, seq_length, _ = x.size()

            # Stack all expert parameters for efficient processing
            expert_outputs = []
            for expert in self.experts:
                up_states = expert.gate_up_proj(x.view(-1, x.size(-1)))  # Flatten to [batch_size * seq_length, input_dim]
                gate, up_states = up_states.chunk(2, dim=-1)
                up_states = up_states * expert.activation_fn(gate)
                expert_output = expert.down_proj(up_states)
                expert_outputs.append(expert_output.view(batch_size, seq_length, -1))

            expert_outputs = torch.stack(expert_outputs, dim=-1)  # Shape: [batch_size, seq_length, hidden_size, num_experts]
            
            # Use torch.gather to select the expert outputs based on gate_indices
            expanded_gate_indices = gate_indices.unsqueeze(-2).expand(-1, -1, self.output_dim, -1)  # Shape: [batch_size, seq_length, hidden_size, k]
            selected_expert_outputs = torch.gather(expert_outputs, -1, expanded_gate_indices)  # Shape: [batch_size, seq_length, hidden_size, k]

            # Weight the selected expert outputs by gate values
            gate_values = gate_values.unsqueeze(-2)  # Shape: [batch_size, seq_length, 1, k]
            weighted_expert_outputs = selected_expert_outputs * gate_values  # Shape: [batch_size, seq_length, hidden_size, k]

            # Sum the weighted expert outputs across the k dimension
            moe_output = weighted_expert_outputs.sum(dim=-1)  # Shape: [batch_size, seq_length, hidden_size]

        return moe_output.to(self.gating_layer.gate.weight.dtype)
        

# Define the ModifiedPhi3DecoderLayer Layer
class ModifiedPhi3DecoderLayer(nn.Module):
    def __init__(self, original_layer, moe_layer):
        super(ModifiedPhi3DecoderLayer, self).__init__()
        self.self_attn = original_layer.self_attn
        self.mlp = moe_layer
        self.input_layernorm = original_layer.input_layernorm
        self.resid_attn_dropout = original_layer.resid_attn_dropout
        self.resid_mlp_dropout = original_layer.resid_mlp_dropout
        self.post_attention_layernorm = original_layer.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        with torch.autocast(device_type="cuda", dtype=hidden_states.dtype):
            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            attn_outputs = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            attn_output = attn_outputs[0]
            hidden_states = residual + self.resid_attn_dropout(attn_output)

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            
            hidden_states = self.mlp(hidden_states) 

            
            hidden_states = residual + self.resid_mlp_dropout(hidden_states)

            outputs = (hidden_states,)

            if output_attentions:
                outputs += (attn_outputs[1],)

            if use_cache:
                outputs += (attn_outputs[2],)

        return outputs


#Define Phi3VForCausalLMMoEConfig
class Phi3VForCausalLMMoEConfig(Phi3VConfig):
    model_type = "phi3_v_moe"

    def __init__(self, config=None, k=1, num_expert_models=2, use_embeddings_in_router=False, **kwargs):
        if config is not None:
            kwargs.update(config.to_dict())
        super().__init__(**kwargs)
        self.k = k
        self.num_expert_models = num_expert_models
        self.architectures = "Phi3VForCausalLMMoE"
        self.auto_map = {
            "AutoConfig": "moe_phi3_v.Phi3VForCausalLMMoEConfig",
            "AutoModelForCausalLM": "moe_phi3_v.Phi3VForCausalLMMoE",
        }
        self.use_embeddings_in_router=use_embeddings_in_router


#Define MoE Model
class Phi3VForCausalLMMoE(Phi3VForCausalLM):
    config_class = Phi3VForCausalLMMoEConfig

    def __init__(
        self,
        config,
        base_model=None,
        expert_models=None,
        layer_dtype=torch.bfloat16,
        **kwargs,
    ):
        super().__init__(config)

        self.layer_dtype = layer_dtype
        self.custom_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        k = self.config.k
        self.num_layers = len(base_model.model.layers) if base_model else 0
        

        self.config.auto_map = {
            "AutoConfig": "moe_phi3_v.Phi3VForCausalLMMoEConfig",
            "AutoModelForCausalLM": "moe_phi3_v.Phi3VForCausalLMMoE",
        }

        self.use_embeddings_in_router=config.use_embeddings_in_router
        print ("Use embeddigs in router: ", self.use_embeddings_in_router )


        self.model = base_model or Phi3VForCausalLM(
            self.config
        )  

        if base_model and expert_models:
            self.num_expert_models = len(expert_models)
            self._init_moe_layers(base_model, expert_models, k, layer_dtype)
        else:
            print(
                "Init function called and generating dummy experts: k=",
                k,
                "experts=",
                self.config.num_expert_models,
            )
            num_dummy_experts = self.config.num_expert_models
            self._init_moe_layers_with_dummy_experts(
                self.model, k, num_dummy_experts, layer_dtype
            )

        self.config.model_type = "phi3_v_moe"
        

    def _init_base_model(self):
        return PreTrainedModel(self.config)

    def _init_moe_layers(self, base_model, expert_models, k, layer_dtype):
        self.num_layers = len(base_model.model.layers)
        for i in tqdm(range(self.num_layers)):
            experts = []
            for expert_model in expert_models:
                expert = copy.deepcopy(expert_model.model.layers[i].mlp).to(
                    dtype=layer_dtype
                )
                experts.append(expert)

            gating_layer = GatingLayer(
                input_dim=self.config.hidden_size,
                num_experts=len(experts),
                k=k,
                layer_dtype=layer_dtype,
            )
            moe_layer = MoE(
                input_dim=self.config.hidden_size,
                experts=experts,
                gating_layer=gating_layer,
                config=self.config,
            ).to(dtype=layer_dtype)

            self.model.model.layers[i] = ModifiedPhi3DecoderLayer(
                self.model.model.layers[i], moe_layer
            ).to(dtype=layer_dtype)

    def _init_moe_layers_with_dummy_experts(
        self, base_model, k, num_dummy_experts, layer_dtype
    ):
        self.num_layers = len(base_model.model.layers)

        for i in tqdm(range(self.num_layers)):
            experts = []
            for _ in range(num_dummy_experts):
                dummy_expert = Phi3MLP(self.config).to(dtype=layer_dtype)
                experts.append(dummy_expert)

            gating_layer = GatingLayer(
                input_dim=self.config.hidden_size,
                num_experts=len(experts),
                k=k,
                layer_dtype=layer_dtype,
            )
            moe_layer = MoE(
                input_dim=self.config.hidden_size,
                experts=experts,
                gating_layer=gating_layer,
                config=self.config,
            ).to(dtype=layer_dtype)

            self.model.model.layers[i] = ModifiedPhi3DecoderLayer(
                self.model.model.layers[i], moe_layer
            ).to(dtype=layer_dtype)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Initialize the model using the superclass method
        model = super(Phi3VForCausalLMMoE, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        return model
    def plot_loss_histories(self, all_loss_histories, loss_steps, filename="loss_history.svg"):
        plt.figure(figsize=(12, 8))
        for layer_idx, loss_history in enumerate(all_loss_histories):
            plt.plot(
                    range(0, len(loss_history) * loss_steps, loss_steps),
                    loss_history,
                    label=f'Layer {layer_idx}',
                    linewidth=2,  # Thicker line
                    marker='o'  # Circle marker for each data point
                )
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss History per Layer, MoE Gating Network')
        plt.legend()
        plt.grid(True)
        try:
            plt.savefig(filename)
        except:
            print("Figure file save failed...")
        plt.show()

    def train_gating_layer_params_from_hidden_states(self, processor, prompts_per_expert, epochs=1000, loss_steps=100,
                                                    lr=1e-4, layer_offset=0):
        self.to(self.custom_device)
        self.eval()

        print ('btype:', self.layer_dtype, 'device=', self.custom_device)
    
        all_gating_layer_params = []
        all_loss_histories = []  # To store loss histories for each layer
    
        expert_hidden_states_per_layer = [[] for _ in range(self.num_layers)]
    
        # Collect hidden states for each expert
        for prompts in tqdm(prompts_per_expert, desc="Processing Prompts"):
            for prompt in tqdm(prompts, desc="Processing Single Prompt", leave=False):
                inputs = processor(text=prompt['text'], images=prompt['image'], return_tensors="pt").to(self.custom_device).to(self.layer_dtype)
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                    for layer_idx in tqdm(range(self.num_layers)):
                        hidden_state = hidden_states[layer_idx+layer_offset].mean(dim=1)  # Averaging over the sequence dimension
                        
                        expert_hidden_states_per_layer[layer_idx].append(hidden_state)
    
        # Train the gating layers
        for layer_idx in tqdm(range(self.num_layers), desc="Training Gating Layers"):
            print(f"Training gating layer parameters for layer {layer_idx}")
    
            # Ensure we have hidden states collected for the current layer
            if not expert_hidden_states_per_layer[layer_idx]:
                raise ValueError(f"No hidden states collected for layer {layer_idx}")
    
            # Aggregate hidden states for each expert and stack them
            expert_hidden_states = []
            num_prompts_per_expert = len(prompts_per_expert[0])
            for i in range(len(prompts_per_expert)):
                hidden_states_for_expert = expert_hidden_states_per_layer[layer_idx][i * num_prompts_per_expert: (i + 1) * num_prompts_per_expert]
                hidden_state_avg = torch.stack(hidden_states_for_expert).mean(dim=0)
                expert_hidden_states.append(hidden_state_avg)
            expert_hidden_states = torch.stack(expert_hidden_states).to(self.layer_dtype)
    
            input_dim = self.config.hidden_size  
            num_experts = self.config.num_expert_models
            class SimpleGatingLayer(nn.Module):
                def __init__(self, input_dim, num_experts, layer_dtype=torch.bfloat16):
                    super(SimpleGatingLayer, self).__init__()
                    self.gate = nn.Linear(input_dim, num_experts).to(dtype=layer_dtype)
    
                def forward(self, x):
                    #return torch.softmax(self.gate(x), dim=-1)
                    return self.gate(x)    

            gating_layer = SimpleGatingLayer(self.config.hidden_size, num_experts, layer_dtype=self.layer_dtype).to(self.custom_device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = Adam(gating_layer.parameters(), lr=lr)    
    
            loss_history = []
    
            for epoch in tqdm(range(epochs), desc=f"Training Gating Layer {layer_idx}"):
                optimizer.zero_grad()
                # Reshape expert_hidden_states to match (batch_size, input_dim)
                expert_hidden_states_reshaped = expert_hidden_states.view(-1, input_dim)
                outputs = gating_layer(expert_hidden_states_reshaped)
                labels = torch.arange(num_experts).to(self.custom_device)
                #print ("outputs, labels" , outputs.shape, labels.shape)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
                if epoch % loss_steps == 0:
                    loss_history.append(loss.item())
    
            all_loss_histories.append(loss_history)
            all_gating_layer_params.append(gating_layer.state_dict())
    
        self.plot_loss_histories(all_loss_histories, loss_steps)
        return all_gating_layer_params
    def set_gating_layer_params(self, gating_layer_params):
        for layer_idx, params in enumerate(gating_layer_params):
            self.model.model.layers[layer_idx].mlp.gating_layer.load_state_dict(params) 

def freeze_except_gating_layers(model):
    # freeze_except_gating_layers(moe_model)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze gating layer parameters
    for layer in model.model.model.layers:
        for name, param in layer.mlp.gating_layer.named_parameters():
            param.requires_grad = True


def un_freeze_all(model):
    # freeze_except_gating_layers(moe_model)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = True


from transformers import AutoConfig

AutoConfig.register("phi3_v_moe", Phi3VForCausalLMMoEConfig)

from transformers.models.auto.modeling_auto import MODEL_MAPPING

MODEL_MAPPING.update({"phi3_v_moe": Phi3VForCausalLMMoE})


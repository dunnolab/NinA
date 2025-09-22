import logging
import os
from typing import Any, Dict, Optional, Tuple, Collection, List
from functools import partial
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import einops
from einops import rearrange, repeat
from torch import einsum
from einops_exts import rearrange_many
import wandb
from timm.layers.mlp import Mlp
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig

from flower.models.networks.transformers import (
    TimestepEmbedder,
    SharedAdaLNController,
    RmsNorm,
    FreqEmbedder,
    ActionSpaceEmbedderParameter,
    ZeroEncoder,
    FlowBlock, 
    stateless_norm
)
from flower.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from flower.callbacks.ema import EMA
from flower.models.utils import ActionIndex, generate_policy_prompt

logger = logging.getLogger(__name__)

class FLOWERVLA(pl.LightningModule):
    def __init__(
        self,
        # VLM Configuration
        vlm_path: str = "microsoft/Florence-2-base",
        freeze_florence: bool = False,
        freeze_vision_tower: bool = False,
        vlm_prompt_style: str = "default",
        token_dropout: float = 0.2,
        
        # Model Structure
        multistep: int = 10,
        num_sampling_steps: int = 5,
        lowdim_obs_dim: int = 7,
        action_dim: int = 7,
        act_window_size: int = 10,
        
        # Model flags
        use_second_view: bool = False,
        second_view_key: str = 'image_wrist',
        action_type_adaln: bool = True,
        use_causal_attention: bool = True,
        use_cross_attn: bool = True,
        use_adaln_cond: bool = False,
        use_readout_token: bool = False,
        use_proprio: bool = False,
        return_act_chunk: bool = False,
        
        # DiT Configuration
        sampling_type: str = 'ln',
        dit_dim: int = 512,
        n_heads: int = 16,
        n_layers: int = 12,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_pdrop: float = 0.1,
        
        # RoPE Configuration
        use_rope: bool = False,
        use_nope: bool = False,
        query_seq_len: int = 128,
        rope_theta: float = 32.0,
        
        # Optimizer Configuration
        optimizer_type: str = "adamw",
        optimizer: DictConfig = None,
        lr_scheduler: DictConfig = None,

        load_pretrained: bool = False,
        pretrained_model_path: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        # self.automatic_optimization = False
        self.action_space_index = ActionIndex()
        # Initialize model flags and configurations
        self._init_flags(
            use_second_view=use_second_view,
            use_causal_attention=use_causal_attention,
            use_cross_attn=use_cross_attn,
            use_adaln_cond=use_adaln_cond,
            use_readout_token=use_readout_token,
            use_rope=use_rope,
            use_nope=use_nope,
            vlm_prompt_style=vlm_prompt_style,
            token_dropout=token_dropout,
            action_type_adaln=action_type_adaln,
            sampling_type=sampling_type,
            use_proprio=use_proprio,
            return_act_chunk=return_act_chunk,
            second_view_key=second_view_key,
        )
        self.obs_modalities = []
        # Initialize model dimensions
        self._init_dimensions(
            dit_dim=dit_dim,
            n_heads=n_heads,
            lowdim_obs_dim=lowdim_obs_dim,
            action_dim=action_dim,
            act_window_size=act_window_size,
            multistep=multistep,
            num_sampling_steps=num_sampling_steps,
        )
        self.target_modality = "actions"
        # Setup VLM and core components
        self._setup_vlm(vlm_path, freeze_vision_tower, freeze_florence)
        hidden_dim = self.vlm.config.text_config.d_model
        self.vlm_latent_dim = hidden_dim
        self.action_type_adaln = action_type_adaln
        self.use_proprio = use_proprio

        # Initialize state tracking
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self.modality_scope = "lang"
        # Save optimizer config
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.optimizer_type = optimizer_type

        if load_pretrained and pretrained_model_path is not None:
            self._load_pretrained_weights(pretrained_model_path)

            # Setup DiT components
        self._setup_dit_components(
            dit_dim=dit_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            lowdim_obs_dim=lowdim_obs_dim,
            action_dim=action_dim,
            act_window_size=act_window_size,
            hidden_dim=hidden_dim,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_pdrop=mlp_pdrop,
            use_cross_attn=use_cross_attn,
            use_rope=use_rope,
            use_nope=use_nope,
            query_seq_len=query_seq_len,
            rope_theta=rope_theta,
        )

    def _load_pretrained_weights(self, pretrained_model_path: str, mean_resizing: bool = False):
        """Loads pretrained weights, handling key mismatches (e.g., different prefixes)."""
        print(f"Loading pretrained weights from {pretrained_model_path}...")
        # Determine file type and load accordingly
        if pretrained_model_path.endswith('.safetensors'):
            # Load safetensors file
            from safetensors.torch import load_file
            state_dict = load_file(pretrained_model_path, device=str(self.device))
            checkpoint = {"state_dict": state_dict}  # Create checkpoint-like structure for compatibility
            print("Loaded safetensors file")
        else:
            # Load PyTorch checkpoint (.pt, .pth, .ckpt)
            checkpoint = torch.load(pretrained_model_path, map_location=self.device)
            # Extract the state dict (handle PyTorch Lightning or plain models)
            state_dict = checkpoint.get("state_dict", checkpoint)

        # Extract the state dict (handle PyTorch Lightning or plain models)
        state_dict = checkpoint.get("state_dict", checkpoint)

        if ("callbacks" in checkpoint and 
                "EMA" in checkpoint["callbacks"] and 
                "ema_weights" in checkpoint["callbacks"]["EMA"]):
                
                print("Found EMA weights in checkpoint, attempting to load them...")
                ema_weights_list = checkpoint['callbacks']['EMA']['ema_weights']
                
                # Get the original state dict to use as a reference for parameter names and shapes
                original_state_dict = checkpoint.get("state_dict", checkpoint)
                
                # Create a new state dict by matching EMA weights with original parameter names
                state_dict = {}
                ema_idx = 0
                
                for param_name, original_param in original_state_dict.items():
                    if ema_idx < len(ema_weights_list):
                        ema_weight = ema_weights_list[ema_idx]
                        
                        # Check if shapes match
                        if ema_weight.shape == original_param.shape:
                            state_dict[param_name] = ema_weight
                            ema_idx += 1
                        else:
                            # Shape mismatch - try to find the correct EMA weight by shape
                            found_match = False
                            for temp_idx in range(ema_idx, min(ema_idx + 20, len(ema_weights_list))):
                                if ema_weights_list[temp_idx].shape == original_param.shape:
                                    state_dict[param_name] = ema_weights_list[temp_idx]
                                    # Swap to maintain order
                                    ema_weights_list[temp_idx], ema_weights_list[ema_idx] = ema_weights_list[ema_idx], ema_weights_list[temp_idx]
                                    ema_idx += 1
                                    found_match = True
                                    break
                            
                            if not found_match:
                                # If no match found, use original parameter
                                print(f"Warning: No matching EMA weight found for {param_name}, using original")
                                state_dict[param_name] = original_param
                    else:
                        # No more EMA weights available, use original
                        print(f"Warning: Ran out of EMA weights at {param_name}, using original")
                        state_dict[param_name] = original_param
                
                print(f"Successfully matched {ema_idx} EMA weights out of {len(ema_weights_list)} total")

        # Fix key mismatches: remove 'agent.' prefix if it exists
        new_state_dict = {}
        # Handle language encoder/model naming mismatch
        for key, value in state_dict.items():
            new_key = key.replace("agent.", "")  # Remove 'agent.' if it exists
            
            # Handle language encoder/model naming mismatch
            if "vlm.language_encoder." in new_key:
                new_key = new_key.replace("vlm.language_encoder.", "vlm.language_model.model.encoder.")
            elif "vlm.language_model." in new_key and "vlm.language_model.model." not in new_key:
                # If it's already language_model but missing the nested structure, add it
                new_key = new_key.replace("vlm.language_model.", "vlm.language_model.model.encoder.")
                
            new_state_dict[new_key] = value

        # Load the state dict with strict=False to handle mismatches
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)

        # Log mismatches for debugging
        print(f"Pretrained weights loaded with the following issues:")
        if missing_keys:
            print(f"  ⚠️ Missing keys (not found in checkpoint, using default init): {len(missing_keys)}")
            print(f"    {missing_keys[:30]} ...")  # Show first 30 for brevity
        if unexpected_keys:
            print(f"  ⚠️ Unexpected keys (ignored): {len(unexpected_keys)}")
            print(f"    {unexpected_keys[:30]} ...")  # Show first 30 for brevity
        if not missing_keys and not unexpected_keys:
            print("  ✅ All keys matched successfully!")

        # Handle mean-resizing for missing embeddings if enabled
        if mean_resizing:
            self._initialize_new_embeddings(new_state_dict)

        return missing_keys, unexpected_keys

    def _init_flags(self, **kwargs):
        """Initialize model flags and configurations"""
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if self.vlm_prompt_style not in ["default", "feature_focused", "state_oriented"]:
            raise ValueError("Invalid VLM prompt style")
            
        if self.sampling_type not in ['ln', 'pi_zero', 'loglogistic', 'uniform', 'stratified']:
            raise ValueError(f"Invalid sampling type: {self.sampling_type}")
        
        self.format_instruction = functools.partial(
                             generate_policy_prompt,
                             robot_name="Franka Panda",
                             action_space="Delta End-Effector",
                             num_arms="1",
                             prompt_style='minimal')
        
        self.use_adaln_cond = self.use_adaln_cond 
        self.use_readout_token = self.use_readout_token and self.use_adaln_cond
        self.use_proprio = self.use_proprio 
        self.use_second_view = self.use_second_view and self.second_view_key is not None
        self.use_cross_attn = self.use_cross_attn
        self.use_rope = self.use_rope and not self.use_nope
        self.use_nope = self.use_nope and not self.use_rope
        self.vlm_prompt_style = self.vlm_prompt_style
        self.return_act_chunk = False

    def _init_dimensions(self, **kwargs):
        """Initialize model dimensions"""
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        if self.dit_dim % self.n_heads != 0:
            raise ValueError(f"dit_dim ({self.dit_dim}) must be divisible by n_heads ({self.n_heads})")


    def _setup_vlm(self, vlm_path: str, freeze_vision_tower: bool, freeze_florence: bool):
        """Initialize and configure the Florence-2 VLM"""
        print(f"Loading Florence-2 from {vlm_path}")
        
        self.vlm = AutoModelForCausalLM.from_pretrained(vlm_path, trust_remote_code=True)
        
        # Handle parameter freezing
        if freeze_florence:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif not freeze_vision_tower:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = True

        # Setup processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        
        # Create prompt embedding
        self.prompt_embeds = self._create_prompt_embed("<Flow>")
        
        # Remove unnecessary components
        del self.vlm.language_model.model.decoder
        del self.vlm.language_model.lm_head
        
        # Setup token dropout
        self.vlm_token_dropout = nn.Dropout(self.token_dropout)

    def _setup_dit_components(self, **kwargs):
        """Setup DiT model components"""
        # Extract parameters
        dit_dim = kwargs['dit_dim']
        n_heads = kwargs['n_heads']
        n_layers = kwargs['n_layers']
        hidden_dim = kwargs['hidden_dim']
        use_cross_attn = kwargs['use_cross_attn']
        use_rope = kwargs['use_rope']
        use_nope = kwargs['use_nope']

        self.action_encoders = nn.ModuleDict()
        self.action_decoders = nn.ModuleDict()
        if self.use_proprio:
            self.proprio_encoders = nn.ModuleDict()
            
        self.adaln = nn.ModuleDict() if self.action_type_adaln else None

        # Core components
        self.cond_linear = nn.Linear(hidden_dim, dit_dim, bias=False)
        self.t_embedder = TimestepEmbedder(dit_dim)
        self.cond_norm = RmsNorm(hidden_dim)
        self.frequency_embedder = FreqEmbedder(dit_dim)
        self.action_space_embedder = ActionSpaceEmbedderParameter(dit_dim, max_actions=len(self.action_space_index.action_spaces))


        # Positional encoding if not using ROPE/NOPE
        if not use_rope and not use_nope:
            self.positional_encoding = nn.Parameter(torch.randn(1, kwargs['act_window_size'], dit_dim) * 0.1)

        # DiT blocks
        self.dit = nn.ModuleList([
            FlowBlock(
                dit_dim, n_heads,
                attn_pdrop=kwargs['attn_pdrop'],
                resid_pdrop=kwargs['resid_pdrop'],
                mlp_pdrop=kwargs['mlp_pdrop'],
                use_cross_attn=use_cross_attn,
                use_rope=use_rope,
                query_seq_len=kwargs['query_seq_len'],
                rope_theta=kwargs['rope_theta'],

            ) for _ in range(n_layers)
        ])

        # Create components per action space
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            input_dim = self.action_space_index.get_action_dim(action_idx)
            
            # Add encoder/decoder for this action
            self.action_encoders[action_name] = Mlp(in_features=input_dim, hidden_features=dit_dim, out_features=dit_dim, bias=True)
            self.action_decoders[action_name] = nn.Linear(dit_dim, input_dim).to(self.device)
                
            if self.action_type_adaln:
                self.adaln[action_name] = SharedAdaLNController(dit_dim, global_conddim=dit_dim, use_cross_attn=use_cross_attn)

            if self.use_proprio:
                # Add proprio encoder if needed for bimanual nav variant otherwise use zero encoder
                self.proprio_encoders[action_name] = (Mlp(input_dim, dit_dim, out_features=dit_dim, drop=0.2).to(self.device) 
                    if action_name == 'bimanual_nav' else ZeroEncoder(self.dit_dim, device=self.device))

    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Get parameter groups
        optim_groups = self._get_param_groups()

        # Initialize optimizer
        optimizer = torch.optim.AdamW(
                optim_groups,
                lr=self.optimizer_config.learning_rate,
                betas=self.optimizer_config.betas
            )

        # Initialize scheduler
        scheduler = TriStageLRScheduler(
            optimizer,
            OmegaConf.create(self.lr_scheduler_config)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def _get_param_groups(self):
        """Get parameter groups for optimizer"""
        no_decay = ['bias', 'LayerNorm', 'layernorm', 'ln', 'norm']
        decay_group = []
        no_decay_group = []

        # Collect all parameters, excluding VLM if frozen
        for name, param in self.named_parameters():
            if param.requires_grad:
                if any(nd in name.lower() for nd in no_decay):
                    no_decay_group.append(param)
                else:
                    decay_group.append(param)

        return [
            {"params": decay_group, "weight_decay": self.optimizer_config.transformer_weight_decay},
            {"params": no_decay_group, "weight_decay": 0.0}
        ]

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        """Lightning training step"""
        # Get optimizer
        opt = self.optimizers()
        
        # Compute loss
        total_loss = torch.tensor(0.0, device=self.device)
        action_loss = torch.tensor(0.0, device=self.device) 
        total_bs = 0

        for modality_scope, dataset_batch in batch.items():
            self.modality_scope = modality_scope
            obs_features = self.encode_observations(dataset_batch)
            act_loss, losses_dict = self.rf_loss(obs_features, dataset_batch["actions"])
            action_loss = action_loss + act_loss
            total_loss = total_loss + act_loss
            total_bs = total_bs + len(dataset_batch["actions"])

        total_loss = total_loss / len(batch)

        # Log metrics
        self._log_training_metrics(total_loss, action_loss, total_bs)

        # Optimization step
        # opt.zero_grad()
        # self.manual_backward(action_loss)
        
        # Clip gradients
         #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Step optimizer
         #opt.step()

        # Update learning rate
         #sch = self.lr_schedulers()
         #if sch is not None:
        #     sch.step()

        return action_loss

    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Lightning validation step"""
        output = {}
        with torch.no_grad():
            obs_features = self.encode_observations(batch)
            target_actions = batch[self.target_modality]
            
            # Generate noise for sampling
            noise_actions = torch.randn_like(target_actions, device=self.device)

            # Sample actions
            action_pred = self.sample_actions(noise_actions, obs_features, inference=True)
            
            # Compute validation loss
            val_loss = F.mse_loss(action_pred, target_actions)
            
            # Log metrics
            self._log_validation_metrics(val_loss, val_loss)
            
            output["validation_loss"] = val_loss / len(batch)
            return output
            
    def rf_loss(self, cond, actions, dataset_idx=None):
        """
        Compute the rectified flow loss.
        """
        default_dtype = next(self.parameters()).dtype
        
        if len(actions.shape) == 4:
            actions = actions.squeeze(1)
        b = actions.size(0)
        device = actions.device
        actions = actions.to(default_dtype)

        # Sample time based on sampling strategy
        if self.sampling_type == "pi_zero":
            alpha, beta = 1.5, 1.0
            t = torch.distributions.Beta(alpha, beta).sample((b,)).to(device)
            t = t.clamp(max=0.999)
        elif self.sampling_type == "ln":
            t = torch.sigmoid(torch.randn((b,), device=device))
            t = t.clamp(max=0.999).to(default_dtype)
        elif self.sampling_type == "uniform":
            eps = 1e-5
            t = (torch.rand(1, device=device) + torch.arange(b, device=device) / b) % (1 - eps)
            t = t.to(default_dtype)
        else:
            raise NotImplementedError(f"Sampling type {self.sampling_type} not implemented")

        # Interpolate between actions and noise
        texp = t.view([b] + [1] * (actions.dim() - 1))
        z1 = torch.randn_like(actions, device=device).to(default_dtype)

        # Interpolate
        zt = (1 - texp) * actions + texp * z1

        # Forward pass
        vtheta = self.dit_forward(zt, t, cond)
        # Compute loss on valid dimensions only
        diff = (z1 - actions) - vtheta
        valid_diff = diff
        loss = (valid_diff ** 2).mean()

        # Collect debugging info
        losses_dict = {
            "diff_min": valid_diff.min().item(),
            "diff_max": valid_diff.max().item(),
            "diff_mean": valid_diff.mean().item(),
            "loss": loss.item(),
        }

        return loss, losses_dict

    def sample_actions(self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool=False):
        """
        Sample actions using Euler method.
        """
        steps = self.num_sampling_steps if inference else 5
        b = z.size(0)
        device = z.device

        # Integration
        dt = 1.0 / steps
        dt_tensor = torch.tensor([dt] * b, device=device).view([b] + [1]*(z.dim()-1))

        for i in range(steps, 0, -1):
            t_val = i / steps
            t_tensor = torch.full((b,), t_val, device=device)

            # Predict velocity field
            vc = self.dit_forward(z, t_tensor, cond)
            z = z - dt_tensor * vc

        return z.clamp(-1, 1)

    def dit_forward(self, z: torch.Tensor, t: torch.Tensor, cond_dict: dict) -> torch.Tensor:
        """
        Forward pass through the DiT blocks.
        """
        default_dtype = next(self.parameters()).dtype
        B, t_seq, d = z.shape
        
        # Get conditioning information
        cond = cond_dict['features'].to(default_dtype)
        frequency_embeds = cond_dict['frequency_embeds'].squeeze(1).to(default_dtype)
        action_type = cond_dict['action_type'].to(self.device)
        
        # Handle proprioception
        if self.use_proprio and cond_dict['proprio'] is not None:
            proprio = cond_dict['proprio'].to(default_dtype)
            proprio_embeds = self.encode_proprio(proprio, action_type, frequency_embeds.shape)
        else:
            proprio_embeds = torch.zeros_like(frequency_embeds)
        
        # Encode actions
        z, valid_dims = self.encode_actions(z, action_type)
        
        # Add positional encoding if not using ROPE/NOPE
        if not self.use_rope and not self.use_nope:
            z = z + self.positional_encoding
        
        # Process embeddings
        t_emb = stateless_norm(self.t_embedder(t)) + \
                stateless_norm(frequency_embeds).squeeze(1) + \
                stateless_norm(proprio_embeds).squeeze(1)
        
        cond = self.cond_linear(self.cond_norm(cond))
        
        # Set up conditioning
        if self.use_adaln_cond:
            vlm_token = cond[:, 0, :] if self.use_readout_token else cond.mean(dim=1)
            global_cond = vlm_token + t_emb
        else:
            global_cond = t_emb
        
        # Setup context
        cx = z
        context = cond if self.use_cross_attn else None
        
        # Get adaln signals
        if not self.action_type_adaln:
            global_adaln = self.adaln(global_cond)
        else:
            global_adaln = self.action_specific_adaln(global_cond, action_type)
        

        # Process through DiT blocks
        for layer in self.dit:
            cx = layer(
                cx, 
                global_cond, 
                context=context, 
                is_causal=True, 
                global_adaln=global_adaln
            )
            
        # Decode and return
        return self.decode_actions(cx, action_type, valid_dims)

    def encode_proprio(self, proprio: torch.Tensor, action_type: torch.Tensor, output_shape) -> torch.Tensor:
        """
        Encode proprioception based on action type.
        """
        batch_size, _ = output_shape
        default_dtype = next(self.parameters()).dtype
        
        if not self.use_proprio:
            return torch.zeros(batch_size, self.dit_dim, device=self.device)
        
        encoded_proprio = torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=default_dtype)
        
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded_proprio[mask] = self.proprio_encoders[action_name](proprio[mask]).squeeze(1)
        
        return encoded_proprio

    def action_specific_adaln(self, global_cond: torch.Tensor, action_type: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate action-specific AdaLN signals.
        """
        default_type = next(self.parameters()).dtype
        batch_size = global_cond.shape[0]
        num_chunks = 9 if self.use_cross_attn else 6
        device = global_cond.device
        
        mod_signals = [
            torch.zeros(batch_size, self.dit_dim, device=device, dtype=default_type) 
            for _ in range(num_chunks)
        ]
        
        for action_idx in range(len(self.action_space_index.action_spaces)):
            mask = (action_type == action_idx)
            if mask.any():
                action_name = self.action_space_index.get_action_name(action_idx)
                action_mod = self.adaln[action_name](global_cond)
                for i, signal in enumerate(action_mod):
                    mod_signals[i] = signal
        
        return mod_signals

    def _create_prompt_embed(self, prompt_text):
        """Create embeddings for prompt tokens"""
        # Add special token if not in vocabulary
        self.tokenizer.add_special_tokens({'additional_special_tokens': [prompt_text]})
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        
        # Get token ID and create embedding
        prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_text)
        prompt_embed = nn.Parameter(
            self.vlm.get_input_embeddings()(torch.tensor(prompt_token_id)), 
            requires_grad=False
        )
    
        return prompt_embed.unsqueeze(0).unsqueeze(0)

    def encode_observations(self, batch: Dict) -> torch.Tensor:
        """Encode observations using Florence-2"""
        device = self.device
        default_type = next(self.parameters()).dtype
        
        
        embed_tensor = torch.zeros(len(batch["rgb_obs"]['rgb_static']), 1, 1)
        action_type_tensor = torch.ones(len(batch["rgb_obs"]['rgb_static']), self.act_window_size, 7)
        # Process primary image
        image_tensor = batch["rgb_obs"]['rgb_static']
        B, T, C, H, W = image_tensor.shape
        
        # Extract visual features
        image_features = self.vlm._encode_image(
            image_tensor.view(-1, C, H, W).to(device).to(default_type)
        ).to(default_type)
        image_features = image_features.view(B, T * image_features.shape[1], -1)
        
        # Process second view if enabled
        if self.use_second_view:
            image2_tensor = batch["rgb_obs"]['rgb_gripper']
            image2_features = self.vlm._encode_image(
                image2_tensor.view(-1, C, H, W).to(device).to(default_type)
            ).to(default_type)
            image2_features = image2_features.view(B, T * image2_features.shape[1], -1)
            image_features = torch.cat([image_features, image2_features], dim=1)
        
        # Get text embeddings
        # Get text embeddings once to reuse
        constructed_prompts = self.construct_prompts(batch)
        text_embeds = self._get_text_embeddings(constructed_prompts, device)
        
        # Add task prompt and aggregation tokens
        task_prompt = self.prompt_embeds.expand(B, -1, -1).to(image_features.device)
        
        # Merge sequence
        merged_embeds = torch.cat([
            image_features,
            task_prompt,
            text_embeds.to(image_features.device)
        ], dim=1)
        
        # Create attention mask
        attention_mask = torch.ones(merged_embeds.shape[:2], device=merged_embeds.device)
        
        # Process through encoder
        features = self.vlm.get_encoder()(
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask
        ).last_hidden_state

        # Apply dropout 
        features = self.vlm_token_dropout(features)

        # Prepare frequency and action space embeddings
        frequency_embeds = self.frequency_embedder(
            torch.ones_like(embed_tensor).to(device) * 3
        )
        
        # Get proprioception if enabled
        proprio = None
        if self.use_proprio and 'proprio' in batch[self.obs_modalities]:
            proprio = batch[self.obs_modalities]['proprio'].to(device).to(default_type)

        return {
            'features': features,
            'frequency_embeds': frequency_embeds,
            'action_space_embeds': None,
            'action_type': torch.ones_like(action_type_tensor), # actiont ype is always 1
            'proprio': proprio,
            'attention_mask': attention_mask,
        }

    def encode_actions(self, z: torch.Tensor, action_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode actions using action-specific encoders."""
        default_dtype = next(self.parameters()).dtype
        action_type = action_type.to(self.device)
        batch_size = z.shape[0]
        encoded = torch.zeros(batch_size, z.shape[1], self.dit_dim, device=self.device).to(default_dtype)
        
        # Track valid dimensions per type
        valid_dims = torch.zeros_like(z).to(default_dtype)
        
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded = self.action_encoders[action_name](z)
        
        return encoded, valid_dims

    def decode_actions(self, z: torch.Tensor, action_type: torch.Tensor, valid_dims: torch.Tensor) -> torch.Tensor:
        """Decode actions using action-specific decoders."""
        default_dtype = next(self.parameters()).dtype
        batch_size = z.shape[0]
        max_action_dim = self.action_dim
        decoded = torch.zeros(batch_size, z.shape[1], max_action_dim, 
                        device=self.device).to(default_dtype)
        
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                action_dim = self.action_space_index.get_action_dim(action_idx)
                
                pred = self.action_decoders[action_name](z)
                # Only assign to valid dimensions
                decoded = pred
        return decoded

    def forward(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """
        Forward pass for inference.
        
        Args:
            obs: Dictionary of observations
            goal: Dictionary containing goal info
            
        Returns:
            Predicted action sequence
        """
        # batch = {'rgb_obs': obs, '"lang_text"': goal}
        rgb_static = obs["rgb_obs"]['rgb_static']
        rgb_gripper = obs["rgb_obs"]['rgb_gripper']

        # Create batch for observation encoding
        batch = {
            "rgb_obs": {
                "rgb_static": rgb_static,
                "rgb_gripper": rgb_gripper
            },
            "lang_text": [goal["lang_text"]]
        }
        features = self.encode_observations(batch)
        
        # Generate initial noise
        noise = torch.randn(
            len(features['features']),
            self.act_window_size,
            self.action_dim,
            device=features['features'].device
        )
        
        # Sample actions
        return self.sample_actions(noise, features, inference=True)

    def step(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """
        Do one step of inference, handling action chunking.
        
        Args:
            obs: Dictionary of observations
            goal: Dictionary containing goal info
            
        Returns:
            Current action prediction
        """
        if self.rollout_step_counter % self.multistep == 0:
            self.pred_action_seq = self(obs, goal)
        
        if not self.return_act_chunk:
            # Default: return current action
            current_action = self.pred_action_seq[0, self.rollout_step_counter]
            if len(current_action.shape) == 2:
                current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        else:
            # Return whole chunk for ALOHA setups
            current_action = self.pred_action_seq
            
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0
        
        return current_action

    def reset(self):
        """Reset model state for new rollout."""
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self.eval()

    def on_train_start(self):
        """Convert model to appropriate dtype on training start."""
        # Move core model components to appropriate device/dtype
        self.to(self.device)
        self.vlm.to(self.device)
        
    def on_validation_start(self):
        """Setup before validation starts."""
        self.eval()

    def on_validation_end(self):
        """Cleanup after validation ends."""
        self.train()

    def print_model_parameters(self):
        """Print model parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params}")
        
        for name, submodule in self.named_modules():
            if '.' not in name or name.count('.') <= 1:
                submodule_params = sum(p.numel() for p in submodule.parameters())
                if submodule_params > 0:
                    print(f"{name} - Total Params: {submodule_params}")
                    
    def print_encoded_texts(self, batch, device):
        """Print encoded text inputs for debugging."""
        text_embeds = self.vlm.get_input_embeddings()(
            batch[self.goal_modalities][self.lang_modalities[0]]['input_ids'].to(self.device)
        ).to(device).squeeze(1)
        
        input_ids = batch[self.goal_modalities][self.lang_modalities[0]]['input_ids'][0].squeeze(0).to(self.device)
        input_ids = input_ids.cpu()
        decoded_text = self.processor.tokenizer.decode(input_ids, skip_special_tokens=False)
        print("Original text:", decoded_text)

        decoded_texts = self.processor.tokenizer.batch_decode(text_embeds.cpu(), skip_special_tokens=True)
        print("Encoded texts:")
        for i, text in enumerate(decoded_texts):
            print(f"Sequence {i+1}: {text}")
    
    def construct_prompts(self, dataset_batch):
        """
        Constructs prompts for Florence-2's encoder to extract task-relevant visual features.
        
        Args:
            dataset_batch: Dictionary containing task information including language instructions
            
        Returns:
            text_prompts: List of formatted prompts for encoder conditioning
        """
        language_instruction = dataset_batch["lang_text"]
        text_prompts = []
        
        for instruction in language_instruction:
            if self.vlm_prompt_style == "default":
                # Original instruction only
                text_prompts.append(self.format_instruction(instruction))
                
            elif self.vlm_prompt_style == "feature_focused":
                # Focus on extracting visual features relevant for manipulation
                prompt = f"<od>{instruction}</od><grounding>identify objects and spatial relationships for robotic manipulation</grounding>"
                text_prompts.append(prompt)
                
            elif self.vlm_prompt_style == "state_oriented":
                # Focus on extracting state-relevant features
                prompt = f"<od>{instruction}</od><referring_expression_segmentation>locate objects and regions for manipulation</referring_expression_segmentation>"
                text_prompts.append(prompt)
                
            else:
                raise ValueError(f"Unknown prompt style: {self.vlm_prompt_style}")
        
        
        return text_prompts
    
    def _get_text_embeddings(self, text, device):
        """Get text embeddings to use with VLM"""
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)
        return self.vlm.get_input_embeddings()(text_inputs["input_ids"])
    
    def _log_training_metrics(self, total_loss, action_loss, total_bs):
        """
        Log training metrics
        Args:
            total_loss: Total loss value
            action_loss: Action-specific loss value
            total_bs: Total batch size
        """
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, 
                sync_dist=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, 
                sync_dist=True, batch_size=total_bs)
        
    def _log_validation_metrics(self, pred_loss, val_total_act_loss_pp):
        """
        Log validation metrics
        Args:
            pred_loss: Prediction loss value (scalar)
            val_total_act_loss_pp: Total validation action loss per prediction
        """
        # Log per-modality action loss
        self.log(
            f"val_act/{self.modality_scope}_act_loss_pp", 
            pred_loss, 
            sync_dist=True
        )
        
        # Log average action loss across modalities
        try:
            n_modalities = len(self.trainer.datamodule.modalities)
        except AttributeError:
            n_modalities = 1  # Default if modalities not available
            
        self.log(
            "val_act/action_loss",
            val_total_act_loss_pp / n_modalities,
            sync_dist=True
        )
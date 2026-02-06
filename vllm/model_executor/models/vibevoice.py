# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
VibeVoice ASR model implementation for vLLM.
Based on https://github.com/microsoft/VibeVoice
"""
import copy
import math
import os
import threading
from dataclasses import dataclass
from functools import partial
from subprocess import run
from typing import (Any, ClassVar, Dict, Iterable, List, Literal, Mapping,
                    Optional, Sequence, Tuple, Union)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoConfig, AutoModel, AutoProcessor, AutoTokenizer,
                          BatchFeature, PretrainedConfig, Qwen2Config,
                          Qwen2Tokenizer, Qwen2TokenizerFast)
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.utils import logging

from vllm.config import ModelConfig, VllmConfig
try:
    from vllm.config.speech_to_text import SpeechToTextConfig
except ImportError:
    # Fallback or dummy if not available in this version
    SpeechToTextConfig = Any

from vllm.inputs import PromptType
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (MultiModalEmbeddings,
                                                   SupportsMultiModal,
                                                   SupportsPP)
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper,
                                              init_vllm_registered_model,
                                              maybe_prefix)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (AudioItem, MultiModalFieldConfig,
                                    MultiModalKwargsItems)
from vllm.multimodal.parse import (AudioProcessorItems, MultiModalDataParser,
                                   ModalityData, ModalityDataItems)
from vllm.multimodal.processing import (BaseDummyInputsBuilder,
                                        BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.sequence import IntermediateTensors

logger = logging.get_logger(__name__)

# ============================================================================
# Audio Utilities (FFmpeg based)
# ============================================================================

def _get_ffmpeg_max_concurrency() -> int:
    """Get the maximum FFmpeg concurrency from environment variable."""
    v = os.getenv("VIBEVOICE_FFMPEG_MAX_CONCURRENCY", "")
    try:
        n = int(v) if v.strip() else 0
    except Exception:
        n = 0
    # 0/negative means no explicit limit.
    return n


_FFMPEG_MAX_CONCURRENCY = _get_ffmpeg_max_concurrency()
_FFMPEG_SEM = threading.Semaphore(_FFMPEG_MAX_CONCURRENCY) if _FFMPEG_MAX_CONCURRENCY > 0 else None


def _run_ffmpeg(cmd: list, *, stdin_bytes: bytes = None):
    """Run ffmpeg with optional global concurrency limiting."""
    if _FFMPEG_SEM is None:
        return run(cmd, capture_output=True, check=True, input=stdin_bytes)
    with _FFMPEG_SEM:
        return run(cmd, capture_output=True, check=True, input=stdin_bytes)


def load_audio_use_ffmpeg(file: str, resample: bool = False, target_sr: int = 24000):
    """Open an audio file and read as mono waveform, optionally resampling."""
    if not resample:
        cmd_probe = [
            "ffprobe", "-v", "quiet", "-show_entries", "stream=sample_rate",
            "-of", "default=noprint_wrappers=1:nokey=1", file
        ]
        original_sr = int(run(cmd_probe, capture_output=True, check=True).stdout.decode().strip())
    else:
        original_sr = None

    sr_to_use = target_sr if resample else original_sr

    cmd = [
        "ffmpeg", "-loglevel", "error", "-nostdin", "-threads", "0",
        "-i", file, "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", str(sr_to_use), "-"
    ]

    out = _run_ffmpeg(cmd).stdout
    audio_data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    return audio_data, sr_to_use


def load_audio_bytes_use_ffmpeg(data: bytes, *, resample: bool = False, target_sr: int = 24000):
    """Decode audio bytes via ffmpeg stdin pipe."""
    if not resample:
        raise ValueError("load_audio_bytes_use_ffmpeg requires resample=True")

    cmd = [
        "ffmpeg", "-loglevel", "error", "-threads", "0",
        "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", str(target_sr), "-"
    ]
    out = _run_ffmpeg(cmd, stdin_bytes=data).stdout
    audio_data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    return audio_data, target_sr


class AudioNormalizer:
    def __init__(self, target_dB_FS: float = -25, eps: float = 1e-6):
        self.target_dB_FS = target_dB_FS
        self.eps = eps

    def tailor_dB_FS(self, audio: np.ndarray) -> tuple:
        rms = np.sqrt(np.mean(audio**2))
        scalar = 10 ** (self.target_dB_FS / 20) / (rms + self.eps)
        normalized_audio = audio * scalar
        return normalized_audio, rms, scalar

    def avoid_clipping(self, audio: np.ndarray, scalar: Optional[float] = None) -> tuple:
        if scalar is None:
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                scalar = max_val + self.eps
            else:
                scalar = 1.0
        return audio / scalar, scalar

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        audio, _, _ = self.tailor_dB_FS(audio)
        audio, _ = self.avoid_clipping(audio)
        return audio

# ============================================================================
# Config Classes
# ============================================================================

class VibeVoiceAcousticTokenizerConfig(PretrainedConfig):
    model_type = "vibevoice_acoustic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0.5,
        std_dist_type: str = 'gaussian',
        mixer_layer: str = 'depthwise_conv',
        conv_norm: str = 'none',
        pad_mode: str = 'constant',
        disable_last_norm: bool = True,
        layernorm: str = 'RMSNorm',
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        encoder_n_filters: int = 32,
        encoder_ratios: Optional[List[int]] = None,
        encoder_depths: str = "3-3-3-3-3-3-8",
        decoder_n_filters: int = 32,
        decoder_ratios: Optional[List[int]] = None,
        decoder_depths: Optional[str] = None,
        **kwargs
    ):
        if encoder_ratios is None:
            encoder_ratios = [8, 5, 5, 4, 2, 2]
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.layernorm_eps = layernorm_eps
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.mixer_layer = mixer_layer
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios
        self.encoder_depths = encoder_depths
        self.decoder_ratios = decoder_ratios if decoder_ratios is not None else encoder_ratios
        self.decoder_n_filters = decoder_n_filters
        self.decoder_depths = decoder_depths

class VibeVoiceSemanticTokenizerConfig(PretrainedConfig):
    model_type = "vibevoice_semantic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0,
        std_dist_type: str = 'none',
        mixer_layer: str = 'depthwise_conv',
        conv_norm: str = 'none',
        pad_mode: str = 'constant',
        disable_last_norm: bool = True,
        layernorm: str = 'RMSNorm',
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        encoder_n_filters: int = 32,
        encoder_ratios: Optional[List[int]] = None,
        encoder_depths: str = "3-3-3-3-3-3-8",
        **kwargs
    ):
        if encoder_ratios is None:
            encoder_ratios = [8, 5, 5, 4, 2, 2]
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.layernorm_eps = layernorm_eps
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.mixer_layer = mixer_layer
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios
        self.encoder_depths = encoder_depths

class VibeVoiceConfig(PretrainedConfig):
    model_type = "vibevoice"
    is_composition = True
    sub_configs = {
        "acoustic_tokenizer_config": VibeVoiceAcousticTokenizerConfig,
        "semantic_tokenizer_config": VibeVoiceSemanticTokenizerConfig,
        "decoder_config": Qwen2Config,
    }

    def __init__(
        self,
        acoustic_tokenizer_config=None,
        semantic_tokenizer_config=None,
        decoder_config=None,
        **kwargs
    ):
        kwargs["_attn_implementation_autoset"] = False
        super().__init__(**kwargs)

        if acoustic_tokenizer_config is None:
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"]()
        elif isinstance(acoustic_tokenizer_config, dict):
            acoustic_tokenizer_config["model_type"] = "vibevoice_acoustic_tokenizer"
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"](**acoustic_tokenizer_config)
        else:
            self.acoustic_tokenizer_config = acoustic_tokenizer_config

        if semantic_tokenizer_config is None:
            self.semantic_tokenizer_config = self.sub_configs["semantic_tokenizer_config"]()
        elif isinstance(semantic_tokenizer_config, dict):
            semantic_tokenizer_config["model_type"] = "vibevoice_semantic_tokenizer"
            self.semantic_tokenizer_config = self.sub_configs["semantic_tokenizer_config"](**semantic_tokenizer_config)
        else:
            self.semantic_tokenizer_config = semantic_tokenizer_config

        if decoder_config is None:
            self.decoder_config = self.sub_configs["decoder_config"]()
        elif isinstance(decoder_config, dict):
            if decoder_config.get("model_type", '') == "qwen2":
                self.decoder_config = Qwen2Config(**decoder_config)
            else:
                # Default to Qwen2Config if type is unknown or matches generic
                self.decoder_config = Qwen2Config(**decoder_config)
        else:
            self.decoder_config = decoder_config

        self.acoustic_vae_dim = getattr(self.acoustic_tokenizer_config, 'vae_dim', 64)
        self.semantic_vae_dim = getattr(self.semantic_tokenizer_config, 'vae_dim', 128)

    def get_text_config(self, decoder=False):
        return self.decoder_config

    def to_dict(self):
        output = super().to_dict()
        if "torch_dtype" in output and output["torch_dtype"] is not None:
             # Fix torch.dtype not serializable
             dtype = output["torch_dtype"]
             if isinstance(dtype, torch.dtype):
                 output["torch_dtype"] = str(dtype).replace("torch.", "")
        return output

# Alias for compatibility
VibeVoiceASRConfig = VibeVoiceConfig

# ============================================================================
# Text Tokenizer
# ============================================================================

class VibeVoiceASRTextTokenizerFast(Qwen2TokenizerFast):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        self._add_vibevoice_special_tokens()
        self.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    def _add_vibevoice_special_tokens(self):
        special_tokens = {
            "additional_special_tokens": [
                "<|object_ref_start|>",
                "<|object_ref_end|>",
                "<|box_start|>",
            ]
        }
        self.add_special_tokens(special_tokens)
        self._speech_start_id = self.convert_tokens_to_ids("<|object_ref_start|>")
        self._speech_end_id = self.convert_tokens_to_ids("<|object_ref_end|>")
        self._speech_pad_id = self.convert_tokens_to_ids("<|box_start|>")
        self._eos_id = self.eos_token_id
        self._pad_id = self.convert_tokens_to_ids('<|image_pad|>')

    @property
    def speech_start_id(self) -> int: return self._speech_start_id
    @property
    def speech_end_id(self) -> int: return self._speech_end_id
    @property
    def speech_pad_id(self) -> int: return self._speech_pad_id
    @property
    def pad_id(self) -> int: return self._pad_id

# ============================================================================
# Acoustic/Semantic Tokenizer Models (Audio Encoder)
# ============================================================================

class ConvLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = nn.functional.layer_norm(x.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).type_as(x)
        x = x.transpose(1, 2)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, weight_shape=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            weight_shape = (dim,) if weight_shape is None else weight_shape
            self.weight = nn.Parameter(torch.ones(weight_shape))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

class ConvRMSNorm(RMSNorm):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, weight_shape=None):
        super().__init__(dim, eps, elementwise_affine, weight_shape)

    def forward(self, x):
        x = x.transpose(1, 2)
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        output = output.transpose(1, 2)
        return output

def get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', **norm_kwargs) -> nn.Module:
    if norm == 'layer_norm':
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()

def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length

def pad1d(x: torch.Tensor, paddings: Tuple[int, int], mode: str = 'zero', value: float = 0.):
    padding_left, padding_right = paddings
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        length = x.shape[-1]
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)

class VibeVoiceTokenizerStreamingCache:
    def __init__(self):
        self.cache = {}

    def get(self, layer_id: str, sample_indices: torch.Tensor) -> Optional[torch.Tensor]:
        states = []
        max_length = 0
        for idx in sample_indices.tolist():
            key = (layer_id, idx)
            if key not in self.cache:
                return None
            state = self.cache[key]
            states.append(state)
            max_length = max(max_length, state.shape[-1])

        if len(states) > 0 and states[0].dim() >= 2:
            padded_states = []
            for state in states:
                if state.shape[-1] < max_length:
                    pad_size = max_length - state.shape[-1]
                    padded_state = F.pad(state, (pad_size, 0), mode='constant', value=0)
                    padded_states.append(padded_state)
                else:
                    padded_states.append(state)
            return torch.stack(padded_states, dim=0)
        return torch.stack(states, dim=0)

    def set(self, layer_id: str, sample_indices: torch.Tensor, states: torch.Tensor):
        for i, idx in enumerate(sample_indices.tolist()):
            key = (layer_id, idx)
            self.cache[key] = states[i].detach()

class SConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1,
                groups: int = 1, bias: bool = True, causal: bool = False, norm: str = 'none',
                norm_kwargs: Dict[str, Any] = {}, pad_mode: str = 'reflect'):
        super().__init__()
        # Use nn.Conv1d directly but wrapped logic
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding_total = (kernel_size - 1) * dilation - (stride - 1)
        self.context_size = (kernel_size - 1) * dilation - (stride - 1)
        self._layer_id = None

    @property
    def layer_id(self):
        if self._layer_id is None:
            self._layer_id = f"sconv1d_{id(self)}"
        return self._layer_id

    def forward(self, x: torch.Tensor, cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
                sample_indices: Optional[torch.Tensor] = None, use_cache: bool = False,
                debug: bool = False, is_final_chunk: bool = False) -> torch.Tensor:
        if not use_cache or cache is None:
            return self._forward_non_streaming(x)
        return self._forward_streaming(x, cache, sample_indices, debug, is_final_chunk)

    def _forward_streaming(self, x, cache, sample_indices, debug, is_final_chunk):
        B, C, T = x.shape
        cached_states = cache.get(self.layer_id, sample_indices)
        if cached_states is None:
            if self.context_size > 0:
                cached_states = torch.zeros(B, C, self.context_size, device=x.device, dtype=x.dtype)
            else:
                cached_states = torch.zeros(B, C, 0, device=x.device, dtype=x.dtype)

        if cached_states.shape[2] > 0:
            input_with_context = torch.cat([cached_states, x], dim=2)
        else:
            input_with_context = x

        if is_final_chunk:
            extra_padding = get_extra_padding_for_conv1d(input_with_context, self.kernel_size, self.stride, self.padding_total)
            if extra_padding > 0:
                input_with_context = pad1d(input_with_context, (0, extra_padding), mode=self.pad_mode)

        output = self.conv(input_with_context)
        output = self.norm(output)

        if self.context_size > 0:
            total_input_length = input_with_context.shape[2]
            if total_input_length >= self.context_size:
                new_cache = input_with_context[:, :, -self.context_size:]
            else:
                new_cache = input_with_context
            cache.set(self.layer_id, sample_indices, new_cache)

        return output

    def _forward_non_streaming(self, x):
        extra_padding = get_extra_padding_for_conv1d(x, self.kernel_size, self.stride, self.padding_total)
        if self.causal:
            if self.pad_mode == 'constant':
                x = pad1d(x, (self.padding_total, extra_padding), mode=self.pad_mode, value=0)
            else:
                x = pad1d(x, (self.padding_total, extra_padding), mode=self.pad_mode)
        else:
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)

        x = self.conv(x)
        x = self.norm(x)
        return x

class FFN(nn.Module):
    def __init__(self, embed_dim, ffn_dim, bias=False):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=bias)
        self.gelu = ACT2FN["gelu"]
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x

class Block1D(nn.Module):
    def __init__(self, dim, kernel_size=7, drop_path=0., mixer_layer='conv', layer_scale_init_value=1e-6, **kwargs):
        super().__init__()
        eps = kwargs.get('eps', 1e-6)
        layernorm = kwargs.get('layernorm', 'RMSNorm')
        if layernorm == 'LN':
            self.norm = ConvLayerNorm(dim, eps=eps)
            self.ffn_norm = ConvLayerNorm(dim, eps=eps)
        else:
            self.norm = ConvRMSNorm(dim, eps=eps)
            self.ffn_norm = ConvRMSNorm(dim, eps=eps)

        self.mixer = SConv1d(dim, dim, kernel_size=kernel_size, groups=dim, pad_mode=kwargs.get('pad_mode', 'reflect'),
                             norm=kwargs.get('norm', 'none'), causal=kwargs.get('causal', True), bias=kwargs.get('bias', True))

        self.ffn = FFN(dim, kwargs.get('ffn_expansion', 4) * dim, bias=kwargs.get('bias', False))

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.ffn_gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma = None
            self.ffn_gamma = None

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mixer(x) # SConv1d call
        if self.gamma is not None:
            x = x * self.gamma.unsqueeze(-1)
        x = residual + x

        residual = x
        x = self.ffn_norm(x)
        x = x.permute(0, 2, 1)
        x = self.ffn(x)
        x = x.permute(0, 2, 1)
        if self.ffn_gamma is not None:
            x = x * self.ffn_gamma.unsqueeze(-1)
        x = residual + x
        return x

class TokenizerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.channels = config.channels
        self.dimension = config.dimension
        self.n_filters = config.n_filters
        self.ratios = list(reversed(config.ratios))
        self.depths = config.depths
        self.causal = config.causal
        kernel_size = getattr(config, "kernel_size", 7)
        last_kernel_size = getattr(config, "last_kernel_size", 7)
        norm = getattr(config, "norm", "none")
        pad_mode = getattr(config, "pad_mode", "reflect")
        bias = getattr(config, "bias", True)
        layernorm = getattr(config, "layernorm", "LN")
        layernorm_eps = getattr(config, "layernorm_eps", 1e-6)

        norm_type = ConvLayerNorm if layernorm == 'LN' else ConvRMSNorm

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(SConv1d(self.channels, self.n_filters, kernel_size, norm=norm, causal=self.causal, pad_mode=pad_mode, bias=bias))
        self.downsample_layers.append(stem)

        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2 ** i)
            out_ch = self.n_filters * (2 ** (i + 1))
            self.downsample_layers.append(nn.Sequential(
                SConv1d(in_ch, out_ch, kernel_size=self.ratios[i] * 2, stride=self.ratios[i], causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias)
            ))

        self.stages = nn.ModuleList()
        cur = 0
        layer_type = partial(Block1D, layernorm=layernorm, eps=layernorm_eps, causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias)

        for i in range(len(self.depths)):
            in_ch = self.n_filters * (2 ** i)
            self.stages.append(nn.Sequential(*[layer_type(dim=in_ch) for j in range(self.depths[i])]))

        if not getattr(config, "disable_last_norm", False):
            self.norm = norm_type(in_ch, eps=layernorm_eps)
        else:
            self.norm = nn.Identity()

        self.head = SConv1d(in_ch, self.dimension, kernel_size=last_kernel_size, causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias)

    def forward(self, x, cache=None, sample_indices=None, use_cache=False, debug=False, is_final_chunk=False):
        for i in range(len(self.depths)):
            for layer in self.downsample_layers[i]:
                if isinstance(layer, SConv1d):
                    x = layer(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug, is_final_chunk=is_final_chunk)
                else:
                    x = layer(x)

            for block in self.stages[i]:
                # Block1D unrolling for streaming
                residual = x
                x = block.norm(x)
                x = block.mixer(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug, is_final_chunk=is_final_chunk)
                if block.gamma is not None:
                    x = x * block.gamma.unsqueeze(-1)
                x = residual + x

                residual = x
                x = block.ffn_norm(x)
                x = x.permute(0, 2, 1)
                x = block.ffn(x)
                x = x.permute(0, 2, 1)
                if block.ffn_gamma is not None:
                    x = x * block.ffn_gamma.unsqueeze(-1)
                x = residual + x

        x = self.norm(x)
        x = self.head(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug, is_final_chunk=is_final_chunk)
        return x

@dataclass
class VibeVoiceTokenizerEncoderOutput:
    mean: torch.Tensor
    std: Optional[Union[float, torch.Tensor]] = None

    def sample(self, dist_type='fix'):
        if dist_type == 'fix':
            x = self.mean + self.std * torch.randn_like(self.mean)
            return x, self.std
        return self.mean, self.std

class VibeVoiceAcousticTokenizerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer('fix_std', torch.tensor(config.fix_std), persistent=False)
        self.std_dist_type = getattr(config, "std_dist_type", "fix")

        if isinstance(config.encoder_depths, str):
            encoder_depths = [int(d) for d in config.encoder_depths.split('-')]
        else:
            encoder_depths = config.encoder_depths

        encoder_config = copy.deepcopy(config)
        encoder_config.dimension = config.vae_dim
        encoder_config.n_filters = config.encoder_n_filters
        encoder_config.ratios = config.encoder_ratios
        encoder_config.depths = encoder_depths
        self.encoder = TokenizerEncoder(encoder_config)

    def encode(self, audio, cache=None, sample_indices=None, use_cache=False, debug=False, is_final_chunk=False):
        latents = self.encoder(audio, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug, is_final_chunk=is_final_chunk)
        return VibeVoiceTokenizerEncoderOutput(mean=latents.permute(0, 2, 1), std=self.fix_std)

class VibeVoiceSemanticTokenizerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if isinstance(config.encoder_depths, str):
            encoder_depths = [int(d) for d in config.encoder_depths.split('-')]
        else:
            encoder_depths = config.encoder_depths

        encoder_config = copy.deepcopy(config)
        encoder_config.dimension = config.vae_dim
        encoder_config.n_filters = config.encoder_n_filters
        encoder_config.ratios = config.encoder_ratios
        encoder_config.depths = encoder_depths
        self.encoder = TokenizerEncoder(encoder_config)

    def encode(self, audio, cache=None, sample_indices=None, use_cache=False, debug=False, is_final_chunk=False):
        latents = self.encoder(audio, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug, is_final_chunk=is_final_chunk)
        return VibeVoiceTokenizerEncoderOutput(mean=latents.permute(0, 2, 1))

# ============================================================================
# VibeVoice Audio Encoder
# ============================================================================

class SpeechConnector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = RMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x

class VibeVoiceAudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.acoustic_vae_dim = getattr(config, "acoustic_vae_dim", 64)
        self.semantic_vae_dim = getattr(config, "semantic_vae_dim", 128)

        decoder_config = getattr(config, "decoder_config", None)
        target_hidden_size = None
        if decoder_config:
            target_hidden_size = getattr(decoder_config, "hidden_size", None)
        if target_hidden_size is None:
            target_hidden_size = getattr(config, "hidden_size", 3584)
        self.hidden_size = target_hidden_size

        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(config.acoustic_tokenizer_config)
        self.semantic_tokenizer = VibeVoiceSemanticTokenizerModel(config.semantic_tokenizer_config)

        self.acoustic_connector = SpeechConnector(self.acoustic_vae_dim, self.hidden_size)
        self.semantic_connector = SpeechConnector(self.semantic_vae_dim, self.hidden_size)

        self._audio_encoder_dtype = torch.float32 # Internal computation in float32
        self.compress_ratio = getattr(config, "speech_tok_compress_ratio", 3200)
        self.sample_rate = getattr(config, "target_sample_rate", 24000)
        self.enable_streaming = getattr(config, "enable_streaming", True)
        self.streaming_segment_duration = getattr(config, "streaming_segment_duration", 60.0)

        use_mean_env = os.getenv("VIBEVOICE_USE_MEAN", "").strip().lower()
        self.use_sample = use_mean_env not in ("1", "true", "yes")
        self._lm_dtype = torch.bfloat16

    def _ensure_audio_encoder_dtype(self):
        target_dtype = self._audio_encoder_dtype
        for module in [self.acoustic_tokenizer, self.semantic_tokenizer, self.acoustic_connector, self.semantic_connector]:
            try:
                if next(module.parameters()).dtype != target_dtype:
                    module.to(dtype=target_dtype)
            except StopIteration: pass

    def forward(self, audio: torch.Tensor, *, use_streaming: bool = True, segment_duration_s: Optional[float] = None, use_sample: Optional[bool] = None) -> torch.Tensor:
        self._ensure_audio_encoder_dtype()
        audio = audio.to(dtype=self._audio_encoder_dtype)
        if audio.ndim == 1: audio = audio.unsqueeze(0)

        segment_duration = segment_duration_s or self.streaming_segment_duration
        sample_rate = self.sample_rate
        total_samples = audio.shape[-1]
        segment_samples = int(segment_duration * sample_rate)

        use_streaming = use_streaming and self.enable_streaming and total_samples > segment_samples
        if use_sample is None: use_sample = self.use_sample

        with torch.no_grad():
            if not use_streaming:
                acoustic_input = audio.unsqueeze(1)
                acoustic_out = self.acoustic_tokenizer.encode(acoustic_input)
                acoustic_tokens = acoustic_out.sample(dist_type='gaussian')[0] if use_sample else acoustic_out.mean
                acoustic_embeds = self.acoustic_connector(acoustic_tokens)

                semantic_out = self.semantic_tokenizer.encode(acoustic_input)
                semantic_embeds = self.semantic_connector(semantic_out.mean)
            else:
                acoustic_cache = VibeVoiceTokenizerStreamingCache()
                semantic_cache = VibeVoiceTokenizerStreamingCache()
                acoustic_mean_segments = []
                semantic_mean_segments = []
                batch_size = audio.shape[0]
                sample_indices = torch.arange(batch_size, device=audio.device)

                for start in range(0, total_samples, segment_samples):
                    end = min(start + segment_samples, total_samples)
                    chunk = audio[:, start:end].contiguous()
                    is_final = (end == total_samples)

                    ac_out = self.acoustic_tokenizer.encode(chunk.unsqueeze(1), cache=acoustic_cache, sample_indices=sample_indices, use_cache=True, is_final_chunk=is_final)
                    acoustic_mean_segments.append(ac_out.mean)

                    sc_out = self.semantic_tokenizer.encode(chunk.unsqueeze(1), cache=semantic_cache, sample_indices=sample_indices, use_cache=True, is_final_chunk=is_final)
                    semantic_mean_segments.append(sc_out.mean)

                acoustic_mean_full = torch.cat(acoustic_mean_segments, dim=1) if acoustic_mean_segments else torch.zeros((batch_size, 0, self.acoustic_vae_dim), device=audio.device, dtype=self._audio_encoder_dtype)
                ac_enc_full = VibeVoiceTokenizerEncoderOutput(mean=acoustic_mean_full, std=self.acoustic_tokenizer.fix_std)
                acoustic_tokens = ac_enc_full.sample(dist_type='gaussian')[0] if use_sample else ac_enc_full.mean
                acoustic_embeds = self.acoustic_connector(acoustic_tokens)

                semantic_mean_full = torch.cat(semantic_mean_segments, dim=1) if semantic_mean_segments else torch.zeros((batch_size, 0, self.semantic_vae_dim), device=audio.device, dtype=self._audio_encoder_dtype)
                semantic_embeds = self.semantic_connector(semantic_mean_full)

        combined_embeds = acoustic_embeds + semantic_embeds
        return combined_embeds.to(dtype=self._lm_dtype)

# ============================================================================
# Multimodal Processing
# ============================================================================

class VibeVoiceMultiModalDataParser(MultiModalDataParser):
    def __init__(self, target_sr=24000, **kwargs):
        super().__init__(target_sr=target_sr, **kwargs)
        self.normalizer = AudioNormalizer()

    def _parse_audio_data(self, data: ModalityData[AudioItem]) -> ModalityDataItems[Any, Any] | None:
        if data is None:
            return AudioProcessorItems(None)

        # Check if we should use ffmpeg for string/bytes inputs
        is_list = isinstance(data, list)
        items = data if is_list else [data]
        new_audios = []

        for item in items:
            audio_arr = None
            if isinstance(item, str):
                # Use ffmpeg to load file
                try:
                    audio_arr, _ = load_audio_use_ffmpeg(item, resample=True, target_sr=self.audio_resampler.target_sr)
                except Exception:
                    pass
            elif isinstance(item, bytes):
                try:
                    audio_arr, _ = load_audio_bytes_use_ffmpeg(item, resample=True, target_sr=self.audio_resampler.target_sr)
                except Exception:
                    pass

            if audio_arr is None:
                if isinstance(item, (np.ndarray, torch.Tensor, list, tuple)):
                    audio_arr, orig_sr = self._get_audio_with_sr(item)
                    if orig_sr is not None:
                        audio_arr = self.audio_resampler.resample(audio_arr, orig_sr=orig_sr)
                else:
                    # Try librosa if ffmpeg failed or item is str/bytes and we didn't use ffmpeg
                    try:
                        import librosa
                        from io import BytesIO
                        source = BytesIO(item) if isinstance(item, bytes) else item
                        audio_arr, _ = librosa.load(source, sr=self.audio_resampler.target_sr)
                    except Exception as e:
                        raise ValueError(f"Failed to load audio: {e}")

            # Normalize
            if audio_arr is not None:
                 audio_arr = self.normalizer(audio_arr)
                 new_audios.append(audio_arr)

        return AudioProcessorItems(new_audios)

class VibeVoiceProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_feature_extractor(self, **kwargs) -> WhisperFeatureExtractor:
        # Return dummy extractor for profiling
        return WhisperFeatureExtractor(feature_size=128, sampling_rate=24000, hop_length=240, chunk_length=30, n_fft=400)

    def get_audio_token_info(self) -> dict:
        tokenizer = self.get_tokenizer()
        vocab = tokenizer.get_vocab()
        tokens = {"audio_token": "<|AUDIO|>", "audio_bos_token": "<|audio_bos|>", "audio_eos_token": "<|audio_eos|>"}
        tokens["audio_token_id"] = vocab.get(tokens["audio_token"])
        return tokens

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_data_parser(self) -> MultiModalDataParser:
        return VibeVoiceMultiModalDataParser(target_sr=24000)

class VibeVoiceDummyInputsBuilder(BaseDummyInputsBuilder[VibeVoiceProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<|AUDIO|>" * mm_counts.get("audio", 0)

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        sampling_rate = 24000
        audio_len = 30 * sampling_rate
        return {"audio": [np.zeros(audio_len, dtype=np.float32) for _ in range(mm_counts.get("audio", 0))]}

    def get_dummy_processor_inputs(self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, Any] | None = None):
        return super().get_dummy_processor_inputs(seq_len, mm_counts, mm_options)

def _vibevoice_field_config(hf_inputs):
    return {
        "raw_audio": MultiModalFieldConfig.batched("audio"),
        "raw_audio_lengths": MultiModalFieldConfig.batched("audio"),
        "salt": MultiModalFieldConfig.batched("audio"),
    }

class VibeVoiceMultiModalProcessor(BaseMultiModalProcessor[VibeVoiceProcessingInfo]):
    def _call_hf_processor(self, prompt: str, mm_data: Mapping[str, object], mm_kwargs: Mapping[str, object], tok_kwargs: Mapping[str, object]) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", None)
        if audios is not None and "audio" not in mm_data:
            mm_data["audio"] = audios

        raw_audio_list = mm_data.get("audio")

        # Check if audio input is present and not empty
        has_audio = raw_audio_list is not None
        if has_audio and isinstance(raw_audio_list, (list, tuple)) and len(raw_audio_list) == 0:
            has_audio = False

        if not has_audio:
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        if isinstance(raw_audio_list, np.ndarray): raw_audio_list = [raw_audio_list]
        elif not isinstance(raw_audio_list, list): raw_audio_list = list(raw_audio_list)

        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
        result = BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        max_len = max(len(a) for a in raw_audio_list)
        raw_audio_tensors = []
        audio_lengths = []
        for audio in raw_audio_list:
            audio_len = len(audio)
            audio_lengths.append(audio_len)
            if audio_len < max_len:
                audio = np.pad(audio, (0, max_len - audio_len), mode='constant')
            raw_audio_tensors.append(torch.from_numpy(audio).float())

        result["raw_audio"] = torch.stack(raw_audio_tensors, dim=0)
        result["raw_audio_lengths"] = torch.tensor(audio_lengths, dtype=torch.long)
        import uuid
        result["salt"] = torch.tensor([hash(str(uuid.uuid4())) % 100000], dtype=torch.long).expand(len(raw_audio_list))
        return result

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return _vibevoice_field_config(hf_inputs)

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        audio_token = "<|AUDIO|>"
        audio_token_id = vocab.get(audio_token)
        if audio_token_id is None: return []

        speech_start_id = vocab.get("<|object_ref_start|>")
        speech_end_id = vocab.get("<|object_ref_end|>")
        speech_pad_id = vocab.get("<|box_start|>")

        out_mm_data = out_mm_kwargs.get_data()
        raw_audio_lengths = out_mm_data.get("raw_audio_lengths", [])
        hf_config = self.info.get_hf_config()
        compress_ratio = int(getattr(hf_config, "speech_tok_compress_ratio", 3200))

        def _to_int_len(x):
            if x is None: return 0
            if isinstance(x, torch.Tensor): return int(x.item()) if x.numel() == 1 else int(x.shape[0])
            return int(x)

        def get_replacement(item_idx):
            if raw_audio_lengths is not None and item_idx < len(raw_audio_lengths):
                audio_len = _to_int_len(raw_audio_lengths[item_idx])
                num_features = max(1, int(np.ceil(audio_len / compress_ratio)))
            else:
                num_features = 1 # Fallback

            newline_id = 198
            if speech_start_id and speech_pad_id and speech_end_id:
                replacement_ids = [speech_start_id] + [speech_pad_id] * num_features + [speech_end_id, newline_id]
            else:
                replacement_ids = [audio_token_id] * num_features

            return PromptUpdateDetails.select_token_id(replacement_ids, embed_token_id=int(speech_pad_id or audio_token_id))

        return [PromptReplacement(modality="audio", target=audio_token, replacement=get_replacement)]

# ============================================================================
# Main Model
# ============================================================================

@MULTIMODAL_REGISTRY.register_processor(VibeVoiceMultiModalProcessor, info=VibeVoiceProcessingInfo, dummy_inputs=VibeVoiceDummyInputsBuilder)
class VibeVoiceForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    supports_transcription: ClassVar[Literal[True]] = True
    supports_transcription_only: ClassVar[bool] = False
    supported_languages: ClassVar[Mapping[str, str]] = {"zh": "Chinese", "en": "English"}

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"): return "<|AUDIO|>"
        raise ValueError(f"Unsupported modality: {modality}")

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        if request_prompt:
            return request_prompt
        duration = len(audio) / 24000 if audio is not None else 10.0
        system_prompt = "You are a helpful assistant that transcribes audio input into text output in JSON format."
        show_keys = ["Start time", "End time", "Speaker ID", "Content"]
        user_suffix = (
            f"This is a {duration:.2f} seconds audio, please transcribe it with these keys: "
            + ", ".join(show_keys)
        )
        user_content = "<|AUDIO|>\n" + user_suffix
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return prompt

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: Literal["transcribe", "translate"]
    ) -> SpeechToTextConfig:
        return SpeechToTextConfig(
            language=None,
            task_type=task_type,
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        sampling_rate = 24000
        compress_ratio = 3200
        samples = int(audio_duration_s * sampling_rate)
        num_tokens = int(np.ceil(samples / compress_ratio))
        return num_tokens

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self._model_path = vllm_config.model_config.model
        self.audio_encoder = VibeVoiceAudioEncoder(config)

        decoder_config = getattr(config, "decoder_config", config)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=decoder_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        lm_dtype = vllm_config.model_config.dtype
        if lm_dtype is not None:
            self.audio_encoder._lm_dtype = lm_dtype
        try:
            self.audio_encoder._ensure_audio_encoder_dtype()
        except Exception: pass

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        raw_audio = kwargs.get("raw_audio")
        raw_audio_lengths = kwargs.get("raw_audio_lengths")
        if raw_audio is None: return []

        def flatten_lengths(lengths):
            if lengths is None: return []
            result = []
            if isinstance(lengths, torch.Tensor): lengths = lengths.tolist()
            if isinstance(lengths, (list, tuple)):
                for item in lengths:
                    if isinstance(item, (list, tuple)): result.extend(flatten_lengths(item))
                    elif isinstance(item, torch.Tensor):
                        result.append(item.item() if item.dim()==0 else item.tolist())
                    else: result.append(item)
            else: result.append(lengths)
            return result

        raw_audio_lengths = flatten_lengths(raw_audio_lengths)
        embeddings = []

        # Determine device/dtype from audio_encoder parameters
        try:
            p = next(self.audio_encoder.parameters())
            device, dtype = p.device, p.dtype
        except:
             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
             dtype = torch.float32

        if isinstance(raw_audio, torch.Tensor):
            if raw_audio.dim() == 3: audio_list = [raw_audio[i].squeeze(0) for i in range(raw_audio.shape[0])]
            elif raw_audio.dim() == 2: audio_list = [raw_audio[i] for i in range(raw_audio.shape[0])]
            else: audio_list = [raw_audio]
        elif isinstance(raw_audio, (list, tuple)):
            audio_list = list(raw_audio)
        else:
             audio_list = [raw_audio]

        for i, audio_tensor in enumerate(audio_list):
            if isinstance(audio_tensor, list): audio_tensor = torch.stack(audio_tensor)
            if not isinstance(audio_tensor, torch.Tensor): audio_tensor = torch.tensor(audio_tensor)

            # Use float32 for audio encoder input
            audio_tensor = audio_tensor.to(device=device, dtype=torch.float32)

            if raw_audio_lengths and i < len(raw_audio_lengths):
                actual_len = int(raw_audio_lengths[i])
                if 0 < actual_len <= audio_tensor.shape[-1]:
                    audio_tensor = audio_tensor[..., :actual_len]

            if audio_tensor.numel() < 160: continue

            audio_embeds = self.audio_encoder(audio_tensor)
            final_embed = audio_embeds.squeeze(0)
            embeddings.append(final_embed)

        return tuple(embeddings)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.language_model.get_input_embeddings()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.acoustic_tokenizer.": "audio_encoder.acoustic_tokenizer.",
                "model.semantic_tokenizer.": "audio_encoder.semantic_tokenizer.",
                "model.acoustic_connector.": "audio_encoder.acoustic_connector.",
                "model.semantic_connector.": "audio_encoder.semantic_connector.",
                "model.language_model.": "language_model.model.",
                "lm_head.": "language_model.lm_head.",
            }
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if intermediate_tensors is not None:
            inputs_embeds = None

        language_model = self.language_model
        if hasattr(language_model, "language_model"): language_model = language_model.language_model

        return language_model.model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

# Register Tokenizer
try:
    AutoTokenizer.register(VibeVoiceConfig, slow_tokenizer_class=Qwen2Tokenizer, fast_tokenizer_class=VibeVoiceASRTextTokenizerFast)
except Exception: pass
try:
    AutoProcessor.register(VibeVoiceConfig, processor_class=None) # vLLM uses own processor
except Exception: pass

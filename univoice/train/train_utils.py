from __future__ import annotations
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from collections import OrderedDict
from univoice.constants import *
import transformers
from univoice.tensor_util import spec_to_figure, spec_to_figure_single

def set_all_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d




class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_last: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0
        
        if not drop_last and len(batch) > 0:
            batches.append(batch)

        del indices

        self.batches = batches

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


def get_train_sampler(args, train_dataset) -> Optional[torch.utils.data.Sampler]:
    sampler = SequentialSampler(train_dataset)
    # accelerator.even_batches = False 
    batch_sampler = DynamicBatchSampler( 
        sampler, args.max_frames, max_samples=args.max_samples, random_seed=args.seed, drop_last=False
    )
    batch_sampler.batch_size=None
    return batch_sampler


def get_train_dataloader(args, train_dataset, data_collator) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        dataloader_params = {
            # "batch_size": self._train_batch_size, #
            "collate_fn": data_collator,
            "num_workers": args.dataloader_num_workers,
            "pin_memory": True,  # self.args.dataloader_pin_memory,
            "persistent_workers": args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            # "batch_sampler" and "sampler" difference sampler->batch_sampler
            dataloader_params["batch_sampler"] = get_train_sampler(args, train_dataset)
            # dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = args.dataloader_drop_last
            # dataloader_params["worker_init_fn"] = args.seed
        return DataLoader(train_dataset, **dataloader_params)


def get_eval_dataloader(args, dataset, data_collator) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        dataloader_params = {
            "batch_size": args.eval_batch_size, #
            "collate_fn": data_collator,
            "num_workers": args.dataloader_num_workers,
            "pin_memory": True,  # self.args.dataloader_pin_memory,
            "persistent_workers": args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            # "batch_sampler" and "sampler" difference sampler->batch_sampler
            # dataloader_params["batch_sampler"] = get_train_sampler()
            dataloader_params["sampler"] = SequentialSampler(dataset)
            dataloader_params["drop_last"] = args.dataloader_drop_last
            # dataloader_params["worker_init_fn"] = args.seed
        return DataLoader(dataset, **dataloader_params)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)



def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def setup_fsdp(model, fsdp_strategy: str = 'fsdp', mixed_precision: str = 'fp32', grad_precision: str = 'fp32'):
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "hsdp": ShardingStrategy.HYBRID_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[fsdp_strategy],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[mixed_precision],
            reduce_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[grad_precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    return model


def plot_mel(
        spec_out,
        spec_gt=None,
        name=None,
        title="",
        logger=None,
        step=None,
    ):
        if len(spec_out.shape) == 3:
            spec_out = spec_out[0]
        if isinstance(spec_out, torch.Tensor):
            spec_out = spec_out.cpu().numpy()
        if spec_gt is not None:
            if len(spec_gt.shape) == 3:
                spec_gt = spec_gt[0]
            if isinstance(spec_gt, torch.Tensor):
                spec_gt = spec_gt.cpu().numpy()
           
            spec_out = np.concatenate([spec_out, spec_gt], -1)
        logger.add_figure(
            name,
            spec_to_figure(
                spec_out, title=title
            ),
            step,
        )



def setup_tokenizer(model, tokenizer):
    """
    Add speech generation tokens to the tokenizer. And resize the embedding layer of the model to match the tokenizer vocab size.
    """
    vocab = tokenizer.get_vocab()
    is_new_tokens_added = False
    for token in [DEFAULT_SPEECH_TOKEN, DEFAULT_SPEECH_START_TOKEN, DEFAULT_SPEECH_END_TOKEN, DEFAULT_PAD_TOKEN, '<|asr|>']:
        if token not in vocab:
            is_new_tokens_added = True
    if is_new_tokens_added is False:
        print('all special tokens are already added to tokenizer')
    else:
        tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
        tokenizer.add_tokens([
            DEFAULT_SPEECH_TOKEN, DEFAULT_SPEECH_START_TOKEN, DEFAULT_SPEECH_END_TOKEN, '<|asr|>'], special_tokens=True)
    
    tokenizer.bos_token = "<|im_start|>"
    tokenizer.eos_token = "<|im_end|>"
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer


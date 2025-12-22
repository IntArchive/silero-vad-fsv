# Dimension Flow Analysis for `validate()` Function

This document describes the flow of tensor dimensions through each step of the `validate()` function in `tuning/utils.py`.

## Overview

The `validate()` function processes batched audio data through the Silero VAD model pipeline:
1. **Input**: Batched audio waveforms, ground truth labels, and masks from DataLoader
2. **Processing**: Audio is chunked, processed through STFT → Encoder → Decoder
3. **Output**: Predictions and ground truth values for ROC-AUC calculation

## Key Constants

- `batch_size`: Number of samples per batch (from config, typically 128)
- `context_size`: 64 for 16kHz, 32 for 8kHz
- `num_samples`: 512 for 16kHz, 256 for 8kHz (processing window size)
- `sr`: Sampling rate (16000 Hz for 16k model, 8000 Hz for 8k model)

## Dimension Flow

### Step 1: DataLoader Input

```python fold title:DataLoader Input
for _, (x, targets, masks) in tqdm(enumerate(loader), total=len(loader)):
    # x: (batch_size, max_audio_length)
    #    - Variable length audios padded to same length in batch
    #    - Example: (128, 8192) for batch_size=128, max_audio_length=8192 samples
    # targets: (batch_size, max_gt_length)
    #    - Ground truth labels, one per num_samples chunk
    #    - max_gt_length = max_audio_length / num_samples
    #    - Example: (128, 16) for max_audio_length=8192, num_samples=512
    # masks: (batch_size, max_gt_length)
    #    - Mask values (1.0 for speech, noise_loss for non-speech)
    #    - Same shape as targets: (128, 16)
```

### Step 2: Device Transfer and Padding

```python fold title:Device Transfer and Padding
targets = targets.to(device)  # (batch_size, max_gt_length) - unchanged
x = x.to(device)              # (batch_size, max_audio_length) - unchanged
masks = masks.to(device)      # (batch_size, max_gt_length) - unchanged
x = torch.nn.functional.pad(x, (context_size, 0))
# x: (batch_size, max_audio_length + context_size)
#    - Padded on the left with context_size zeros
#    - Example: (128, 8192 + 64) = (128, 8256)
#    - This padding provides context for the first chunk
```

### Step 3: Chunk Processing Loop Initialization

```python fold title:Chunk Processing Loop
outs = []
state = torch.zeros(0)  # Empty tensor, will be initialized by decoder
# outs: List to collect decoder outputs for each chunk
# state: RNN state tensor, shape (2, batch_size, 128) after first iteration

for i in range(context_size, x.shape[1], num_samples):
    # i starts at context_size (64) and increments by num_samples (512)
    # Processes audio in overlapping chunks
    # Each chunk uses previous context_size samples as context
```

### Step 4: Extract Audio Chunk

```python fold title:Extract Audio Chunk
input_ = x[:, i-context_size:i+num_samples]
# input_: (batch_size, context_size + num_samples)
#    - Extracts chunk with context
#    - Example: (128, 64 + 512) = (128, 576)
#    - Contains: [context (64 samples) | new chunk (512 samples)]
```

### Step 5: STFT Layer

```python fold title:STFT Layer Processing
out = stft_layer(input_)
# input_: (batch_size, context_size + num_samples) = (batch_size, 576)
# out: (batch_size, freq_bins, time_frames)
#    - STFT converts time-domain to frequency-domain
#    - freq_bins: Typically 129 (n_fft//2 + 1) for n_fft=256
#    - time_frames: Depends on STFT stride, typically ~5 frames for 576 samples
#    - Example: (128, 129, 5) or similar
```

### Step 6: Encoder Layer

```python fold title:Encoder Layer Processing
out = encoder_layer(out)
# Input: (batch_size, freq_bins, time_frames) = (128, 129, 5)
# out: (batch_size, 128, time_frames)
#    - Encoder reduces frequency dimension to 128 features
#    - Time dimension preserved
#    - Example: (128, 128, 5)
#    - Note: For typical chunk processing, time_frames may be 1 after pooling
```

### Step 7: Decoder Processing

```python fold title:Decoder Forward Pass
out, state = decoder(out, state)
# Input out: (batch_size, 128, time_frames) - typically (128, 128, 1) after encoder
# Input state: (2, batch_size, 128) after first iteration, or empty tensor initially

# Inside decoder.forward():
#   x = x.squeeze(-1)  # (batch_size, 128, 1) -> (batch_size, 128)
#   h, c = self.rnn(x, (state[0], state[1]))  # LSTMCell processes (batch_size, 128)
#   h: (batch_size, 128), c: (batch_size, 128)
#   x = h.unsqueeze(-1).float()  # (batch_size, 128) -> (batch_size, 128, 1)
#   x = self.decoder(x)  # Conv1d(128->1): (batch_size, 128, 1) -> (batch_size, 1, 1)
#   state = torch.stack([h, c])  # (2, batch_size, 128)

# Output out: (batch_size, 1, 1)
#    - Single probability value per sample in batch
#    - Example: (128, 1, 1)
# Output state: (2, batch_size, 128)
#    - RNN hidden and cell states for next iteration
```

### Step 8: Collect and Stack Outputs

```python fold title:Stack Decoder Outputs
outs.append(out)
# outs: List of tensors, each (batch_size, 1, 1)
#    - One tensor per processed chunk
#    - Example: If 16 chunks processed, list with 16 tensors of shape (128, 1, 1)

stacked = torch.cat(outs, dim=2).squeeze(1)
# torch.cat(outs, dim=2): Concatenate along dimension 2
#    - Input: List of (batch_size, 1, 1) tensors
#    - Output: (batch_size, 1, num_chunks)
#    - Example: (128, 1, 16) for 16 chunks
# .squeeze(1): Remove dimension 1
#    - Input: (batch_size, 1, num_chunks)
#    - Output: (batch_size, num_chunks)
#    - Example: (128, 16)
#    - This matches the shape of targets: (batch_size, max_gt_length)
```

### Step 9: Extract Predictions and Ground Truth

```python fold title:Extract Valid Predictions
predicts.extend(stacked[masks != 0].tolist())
# stacked: (batch_size, num_chunks) = (128, 16)
# masks: (batch_size, num_chunks) = (128, 16)
# masks != 0: Boolean mask, same shape (128, 16)
# stacked[masks != 0]: 1D tensor of valid predictions
#    - Flattened tensor with only non-zero mask positions
#    - Shape: (num_valid_samples,) where num_valid_samples <= batch_size * num_chunks
#    - Example: (2048,) if all positions are valid
# .tolist(): Converts to Python list
# predicts: List of floats, accumulates across batches

gts.extend(targets[masks != 0].tolist())
# targets: (batch_size, num_chunks) = (128, 16)
# targets[masks != 0]: 1D tensor of valid ground truth values
#    - Same indexing as predictions
#    - Shape: (num_valid_samples,)
#    - Example: (2048,)
# .tolist(): Converts to Python list
# gts: List of floats, accumulates across batches
```

### Step 10: Loss Calculation

```python fold title:Loss Calculation
loss = criterion(stacked, targets)
# stacked: (batch_size, num_chunks) = (128, 16)
# targets: (batch_size, num_chunks) = (128, 16)
# criterion: nn.BCELoss(reduction='none')
# loss: (batch_size, num_chunks) - per-element losses
#    - Example: (128, 16)

loss = (loss * masks).mean()
# loss: (batch_size, num_chunks) = (128, 16)
# masks: (batch_size, num_chunks) = (128, 16)
# loss * masks: Element-wise multiplication, same shape (128, 16)
# .mean(): Scalar loss value
#    - Averages all valid (non-zero mask) positions
#    - Returns: torch.Tensor with single float value
```

### Step 11: ROC-AUC Calculation

```python fold title:ROC-AUC Calculation
predicts_np = np.asarray(predicts, dtype=float)
# predicts: List of floats (accumulated across all batches)
# predicts_np: 1D numpy array
#    - Shape: (total_valid_samples,)
#    - Example: (25600,) for 10 batches with 2560 valid samples each

gts_np = np.asarray(gts, dtype=float)
# gts: List of floats (accumulated across all batches)
# gts_np: 1D numpy array
#    - Shape: (total_valid_samples,)
#    - Example: (25600,)
#    - Must contain both 0 and 1 for ROC-AUC to be defined

score = roc_auc_score(gts_np, predicts_np)
# Input: Two 1D arrays of same length
#    - gts_np: (total_valid_samples,)
#    - predicts_np: (total_valid_samples,)
# Output: Scalar float (ROC-AUC score between 0 and 1)
#    - Returns NaN if gts_np contains only one unique class
```

## Summary Table

| Step | Tensor Name | Shape | Description |
|------|-------------|-------|-------------|
| Input | `x` | `(batch_size, max_audio_length)` | Raw audio waveforms |
| Input | `targets` | `(batch_size, max_gt_length)` | Ground truth labels |
| Input | `masks` | `(batch_size, max_gt_length)` | Valid sample masks |
| After padding | `x` | `(batch_size, max_audio_length + context_size)` | Padded audio |
| Chunk | `input_` | `(batch_size, context_size + num_samples)` | Single chunk with context |
| STFT | `out` | `(batch_size, freq_bins, time_frames)` | Frequency domain |
| Encoder | `out` | `(batch_size, 128, time_frames)` | Encoded features |
| Decoder | `out` | `(batch_size, 1, 1)` | Per-chunk prediction |
| Stacked | `stacked` | `(batch_size, num_chunks)` | All predictions |
| Final | `predicts_np` | `(total_valid_samples,)` | Flattened predictions |
| Final | `gts_np` | `(total_valid_samples,)` | Flattened ground truth |

## Important Notes

1. **Variable Length Handling**: The DataLoader pads sequences to the same length within each batch, but different batches may have different `max_audio_length` values.

2. **Chunk Overlap**: Each chunk uses `context_size` previous samples as context, creating overlapping processing windows.

3. **State Persistence**: The decoder RNN state is maintained across chunks within the same audio sample, allowing temporal context.

4. **Mask Usage**: The mask ensures that only valid positions (where ground truth exists) are used for loss calculation and ROC-AUC computation.

5. **ROC-AUC Requirement**: For ROC-AUC to be defined, `gts_np` must contain at least two unique classes (both 0 and 1). If all samples are the same class, ROC-AUC returns NaN.



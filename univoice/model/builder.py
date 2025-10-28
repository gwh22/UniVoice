from torch import nn
from transformers import WhisperModel, WhisperFeatureExtractor


class WhisperProjection(nn.Module):
    def __init__(self, input_embedding_size=1280, output_embedding_size=960):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(250)
        self.proj = nn.Linear(input_embedding_size, output_embedding_size, bias=False)
        self.ln1 = nn.LayerNorm(input_embedding_size)

    def forward(self, whisper_output):
        pooled = self.pool(whisper_output.transpose(-2, -1))
        normalized = self.ln1(pooled.transpose(-2, -1))
        projected = self.proj(normalized)
        return projected



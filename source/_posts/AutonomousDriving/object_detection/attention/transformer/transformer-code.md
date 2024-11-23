---
title: transformer code
top: false
cover: false
toc: true
mathjax: true
summary: Transformer模型pytorch的实现，包含多头注意力、前馈网络、编码器和解码器。
tags:
  - transformer
  - code
categories:
  - 注意力
abbrlink: fae766f4
date: 2024-07-11 11:34:56
password:
---

### Utils.py

`create_pad_mask`函数生成一个用于标记输入序列中填充位置的掩码，

`create_target_self_mask`函数则创建一个用于目标序列的自注意力掩码，屏蔽未来的词。

这两个函数主要用于处理序列数据中的填充和自注意力机制中的遮挡问题。

```python
import torch

def create_pad_mask(t, pad):
    """
    Creates a padding mask.

    Parameters:
    - t (Tensor): The input tensor, typically the input sequence or target sequence.
    - pad (int): The index of the padding token.

    Returns:
    - mask (Tensor): A padding mask tensor with shape (batch_size, 1, seq_len), where each position is a boolean indicating if the position is a padding token.
    """
    mask = (t == pad).unsqueeze(-2)
    return mask


def create_target_self_mask(target_len, device=None):
    """
    Creates a self-attention mask for the target sequence.

    Parameters:
    - target_len (int): The length of the target sequence.
    - device (torch.device, optional): The device on which the tensor is located (e.g., 'cpu' or 'cuda').

    Returns:
    - target_self_mask (Tensor): A self-attention mask tensor with shape (1, target_len, target_len), where the upper triangular part (excluding the diagonal) is True, indicating that these positions are masked.
    """
    ones = torch.ones(target_len, target_len, dtype=torch.bool, device=device)
    # torch.triu返回输入张量的上三角部分, diagonal表示向右上偏移量
    target_self_mask = torch.triu(ones, diagonal=1).unsqueeze(0)
    return target_self_mask
```

### Transformer.py

- **initialize_weight函数**：使用Xavier初始化方法初始化线性层的权重，如果存在偏置项，则将其初始化为零。

- **MutilHeadAttention类**：实现多头注意力机制。它将查询、键和值投影到多个注意力头上进行计算，然后将结果连接并通过线性层输出。

- **FeedForwardNetwork类**：实现带有两个线性层和ReLU激活的前馈神经网络。

- **Encoder_Layer类**：包含多头注意力机制和前馈网络的编码器层，并应用层归一化和Dropout。

- **Encoder类**：包含多个编码器层的编码器，并应用最终的层归一化。

- **Decoder_layer类**：实现解码器层，包含掩码多头注意力、编码器-解码器多头注意力和前馈网络。

- **Decoder类**：包含多个解码器层的解码器，并应用最终的层归一化。

- **transformer类**：实现完整的Transformer模型，包含嵌入层、位置编码、编码器和解码器。它可以处理输入和目标序列，通过编码器生成隐藏表示，并通过解码器生成最终输出。


```python
import math
import time
import torch
from torch import nn
import utils
import torch.nn.functional as F
```
```python
def initialize_weight(x):
    """
    Initializes the weights of the given linear layer using Xavier uniform distribution.
    If the layer has a bias term, it initializes the bias to zero.
    """
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)
```
```python
class MutilHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, num_heads=8) -> None:
        """
        Multi-head attention mechanism as described in the Transformer architecture.

        Args:
            hidden_size (int): Dimensionality of the input and output.
            dropout_rate (float): Dropout rate for attention scores.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.dropout_rate = dropout_rate
        self.scale = att_size ** -0.5

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        initialize_weight(self.q_linear)
        initialize_weight(self.k_linear)
        initialize_weight(self.v_linear)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, mask, cache=None):
        """
        Forward pass for multi-head attention.

        Args:
            q (Tensor): Query tensor of shape (batch_size, seq_len, hidden_size).
            k (Tensor): Key tensor of shape (batch_size, seq_len, hidden_size).
            v (Tensor): Value tensor of shape (batch_size, seq_len, hidden_size).
            mask (Tensor): Attention mask tensor of shape (batch_size, seq_len, seq_len).
            cache (dict): Cache for key and value tensors to reuse during decoding.

        Returns:
            Tensor: Output tensor after applying multi-head attention.
        """
        orig_q_size = q.shape

        d_k = self.att_size
        d_v = self.att_size

        batch_size = q.shape[0]

        q = self.q_linear(q).view(batch_size, -1, self.num_heads, d_k)

        if cache is not None and 'encoder_key' in cache:
            k, v = cache['encoder_key'], cache['encoder_value']
        else:
            k = self.k_linear(k).view(batch_size, -1, self.num_heads, d_k)
            v = self.v_linear(v).view(batch_size, -1, self.num_heads, d_v)
            if cache is not None:
                cache['encoder_key'], cache['encoder_value'] = k, v

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        x.masked_fill_(mask=mask.unsqueeze(1), value=-1e9)
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.shape == orig_q_size
        return x
```
```python
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate) -> None:
        """
        Feed-forward network with two linear layers and ReLU activation.

        Args:
            hidden_size (int): Dimensionality of the input and output.
            filter_size (int): Dimensionality of the hidden layer.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        """
        Forward pass for the feed-forward network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Tensor: Output tensor after applying the feed-forward network.
        """
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x
```
```python
class Encoder_Layer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate) -> None:
        """
        Encoder layer consisting of multi-head attention and feed-forward network.

        Args:
            hidden_size (int): Dimensionality of the input and output.
            filter_size (int): Dimensionality of the feed-forward network's hidden layer.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()

        # Multi-head attention sub-layer with normalization and dropout
        self.MutilHeadAttention_norm = nn.LayerNorm(hidden_size, 1e-6)
        self.MutilHeadAttention = MutilHeadAttention(hidden_size, dropout_rate)
        self.MutilHeadAttention_dropout = nn.Dropout(dropout_rate)

        # Feed-forward network sub-layer with normalization and dropout
        self.FeedForwardNetwork_nrom = nn.LayerNorm(hidden_size, 1e-6)
        self.FeedForwardNetwork = FeedForwardNetwork(
            hidden_size, filter_size, dropout_rate)
        self.FeedForwardNetwork_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        """
        Forward pass for the encoder layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            mask (Tensor): Attention mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tensor: Output tensor after applying the encoder layer.
        """
        # Apply multi-head attention
        y = self.MutilHeadAttention_norm(x)
        y = self.MutilHeadAttention(y, y, y, mask)
        y = self.MutilHeadAttention_dropout(y)
        x = y + x

        # Apply feed-forward network
        y = self.FeedForwardNetwork_nrom(x)
        y = self.FeedForwardNetwork(y)
        y = self.FeedForwardNetwork_dropout(y)
        x = y + x

        return x
```
```python
class Encoder(nn.Module):
    def __init__(self, num_encoder_layer, hidden_size, filter_size, dropout_rate):
        """
        Encoder consisting of multiple encoder layers.

        Args:
            num_encoder_layer (int): Number of encoder layers.
            hidden_size (int): Dimensionality of the input and output.
            filter_size (int): Dimensionality of the feed-forward network's hidden layer.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()

        self.Encoder_layers = nn.ModuleList([Encoder_Layer(
            hidden_size, filter_size, dropout_rate) for _ in range(num_encoder_layer)])
        self.Last_norm = nn.LayerNorm(hidden_size, 1e-6)

    def forward(self, input, mask):
        """
        Forward pass for the encoder.

        Args:
            input (Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            mask (Tensor): Attention mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tensor: Output tensor after applying the encoder.
        """
        output = input
        # self.Encoder_layers(input) 不会逐层调用 Encoder_Layer 的 forward 方法，这里需要遍历每一层。
        for encoder_layer in self.Encoder_layers:
            output = encoder_layer(output, mask)
        output = self.Last_norm(output)
        return output
```
```python
class Decoder_layer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        """
        Decoder layer consisting of masked multi-head attention, encoder-decoder attention,
        and feed-forward network.

        Args:
            hidden_size (int): Dimensionality of the input and output.
            filter_size (int): Dimensionality of the feed-forward network's hidden layer.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.MutilHeadAttention_norm = nn.LayerNorm(hidden_size, 1e-6)
        self.MutilHeadAttention = MutilHeadAttention(hidden_size, dropout_rate)
        self.MutilHeadAttention_dropout = nn.Dropout(dropout_rate)

        self.Encoder_Decoder_MutilHeadAttention_norm = nn.LayerNorm(
            hidden_size, 1e-6)
        self.Encoder_Decoder_MutilHeadAttention = MutilHeadAttention(
            hidden_size, dropout_rate)
        self.Encoder_Decoder_MutilHeadAttention_dropout = nn.Dropout(
            dropout_rate)

        self.FeedForwardNetwork_norm = nn.LayerNorm(hidden_size, 1e-6)
        self.FeedForwardNetwork = FeedForwardNetwork(
            hidden_size, filter_size, dropout_rate)
        self.FeedForwardNetwork_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, encoder_output, cross_mask, self_mask, cache=None):
        """
        Forward pass for the decoder layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            encoder_output (Tensor): Encoder output tensor of shape (batch_size, seq_len, hidden_size).
            cross_mask (Tensor): Cross-attention mask tensor of shape (batch_size, seq_len, seq_len).
            self_mask (Tensor): Self-attention mask tensor of shape (batch_size, seq_len, seq_len).
            cache (dict): Cache for key and value tensors to reuse during decoding.

        Returns:
            Tensor: Output tensor after applying the decoder layer.
        """
        # Masked multi-head attention
        y = self.MutilHeadAttention_norm(x)
        y = self.MutilHeadAttention(y, y, y, self_mask, cache)
        y = self.MutilHeadAttention_dropout(y)
        x = y + x

        # Encoder-decoder multi-head attention
        if encoder_output is not None:
            y = self.Encoder_Decoder_MutilHeadAttention_norm(x)
            y = self.Encoder_Decoder_MutilHeadAttention(
                y, encoder_output, encoder_output, cross_mask)
            y = self.Encoder_Decoder_MutilHeadAttention_dropout(y)
            x = y + x

        # Feed-forward network
        y = self.FeedForwardNetwork_norm(x)
        y = self.FeedForwardNetwork(y)
        y = self.FeedForwardNetwork_dropout(y)
        x = y + x

        return x
```
```python
class Decoder(nn.Module):
    def __init__(self, num_decoder_layer, hidden_size, filter_size, dropout_rate):
        """
        Decoder consisting of multiple decoder layers.

        Args:
            num_decoder_layer (int): Number of decoder layers.
            hidden_size (int): Dimensionality of the input and output.
            filter_size (int): Dimensionality of the feed-forward network's hidden layer.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.Decoder_layers = nn.ModuleList([Decoder_layer(
            hidden_size, filter_size, dropout_rate) for _ in range(num_decoder_layer)])
        self.Last_norm = nn.LayerNorm(hidden_size, 1e-6)

    def forward(self, targets, encoder_output, cross_mask, self_mask, cache=None):
        """
        Forward pass for the decoder.

        Args:
            input (Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            encoder_output (Tensor): Encoder output tensor of shape (batch_size, seq_len, hidden_size).
            cross_mask (Tensor): Cross-attention mask tensor of shape (batch_size, seq_len, seq_len).
            self_mask (Tensor): Self-attention mask tensor of shape (batch_size, seq_len, seq_len).
            cache (dict): Cache for key and value tensors to reuse during decoding.

        Returns:
            Tensor: Output tensor after applying the decoder.
        """
        output = targets
        for i, decoder_layer in enumerate(self.Decoder_layers):
            layer_cache = None
            if cache is not None:
                if i not in cache:
                    cache[i] = {}  # 初始化该层的缓存字典
                layer_cache = cache[i]  # 读取该层缓存数据
            output = decoder_layer(
                output, encoder_output, cross_mask, self_mask, layer_cache)
        return self.Last_norm(output)
```
```python
class transformer(nn.Module):
    def __init__(self,
                 input_vocab_size,  # 输入序列的词汇表大小
                 target_vocab_size,  # 目标序列的词汇表大小
                 num_layers=6,  # 编码器和解码器的层数
                 hidden_size=512,  # 嵌入和隐藏层的维度大小
                 filter_size=2048,  # FFN的隐藏层大小
                 dropout_rate=0.1,  # Dropout的概率
                 share_target_embedding=True,  # 是否共享输入和目标词汇表的嵌入
                 has_inputs=True,  # 模型是否有输入（模型是否包含编码器部分）
                 src_pad_idx=None,  # 输入序列中的填充索引
                 trg_pad_idx=None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_scale = hidden_size ** 0.5  # 嵌入的缩放因子，用于缩放嵌入向量
        self.has_inputs = has_inputs
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.target_vocab_embedding = nn.Embedding(
            target_vocab_size, hidden_size)
        nn.init.normal_(self.target_vocab_embedding.weight,
                        mean=0, std=hidden_size**-0.5)

        self.target_embedding_dropout = nn.Dropout(dropout_rate)

        self.decoder = Decoder(num_decoder_layer=num_layers, hidden_size=hidden_size,
                               filter_size=filter_size, dropout_rate=dropout_rate)

        if has_inputs:
            if not share_target_embedding:
                self.input_vocab_embedding = nn.Embedding(
                    input_vocab_size, hidden_size)
                nn.init.normal_(self.input_vocab_embedding.weight,
                                mean=0, std=hidden_size**-0.5)
            else:
                self.input_vocab_embedding = self.target_vocab_embedding

            self.input_embedding_dropout = nn.Dropout(dropout_rate)
            self.encoder = Encoder(num_encoder_layer=num_layers, hidden_size=hidden_size,
                                   filter_size=filter_size, dropout_rate=dropout_rate)

        # positional encoding

        num_timescales = self.hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (math.log(
            float(max_timescale) / float(min_timescale)) / max(num_timescales-1, 1))
        # 位置编码的逆时间尺度
        inv_timescales = min_timescale * \
            torch.exp(torch.arange(num_timescales, dtype=torch.float32)
                      * -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

    def forward(self, inputs, targets):
        cache = {}
        encoder_output, input_mask = None, None
        if self.has_inputs:
            input_mask = utils.create_pad_mask(inputs, self.src_pad_idx)
            encoder_output = self.encode(inputs, input_mask)

        # 目标序列的填充掩码
        target_mask = utils.create_pad_mask(targets, self.trg_pad_idx)
        target_size = targets.shape[1]
        # 目标序列的自注意力掩码
        target_self_mask = utils.create_target_self_mask(
            target_size, device=targets.device)
        return self.decode(targets, encoder_output, input_mask, target_self_mask, target_mask, cache)

    def encode(self, input, input_mask):
        input_embedded = self.input_vocab_embedding(input)
        input_embedded.masked_fill_(input_mask.squeeze(1).unsqueeze(-1), 0)
        input_embedded *= self.embedding_scale
        input_embedded += self.get_position_encoding(input)
        input_embedded = self.input_embedding_dropout(input_embedded)
        return input_embedded

    def decode(self, targets, encoder_output, input_mask, target_self_mask, target_mask, cache=None):
        # target embedding
        target_embedded = self.target_vocab_embedding(targets)
        target_embedded.masked_fill_(target_mask.squeeze(1).unsqueeze(-1), 0)

        # shifting
        target_embedded = target_embedded[:, :-1]
        target_embedded = F.pad(target_embedded, (0, 0, 1, 0))

        target_embedded *= self.embedding_scale
        target_embedded += self.get_position_encoding(targets)
        target_embedded = self.target_embedding_dropout(target_embedded)

        # decoder
        decoder_output = self.decoder(
            target_embedded, encoder_output, input_mask, target_self_mask, cache)

        output = torch.matmul(
            decoder_output, self.target_vocab_embedding.weight.transpose(0, 1))
        return output

    def get_position_encoding(self, x):
        max_length = x.shape[1]
        position = torch.arange(
            max_length, dtype=torch.float32, device=x.device)

        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
        signal = signal.view(1, max_length, self.hidden_size)
        return signal
```
```python
if __name__ == "__main__":
    # MHAttention = MutilHeadAttention(512,0.5,8)
    # data = torch.rand((16, 256, 512))
    # mask = torch.ones((16, 256, 256), dtype=torch.bool)  # Example mask, change as needed
    # out = MHAttention(data, data, data, mask)
    # print(data.shape)
    # print(out.shape)

    # FFN = FeedForwardNetwork(512, 2048, 0.5)
    # data = torch.rand((16,64,512))
    # out = FFN(data)
    # print(data.shape)
    # print(data.shape)

    # encoder = Encoder(4, 512, 2048, 0.5)
    # data = torch.rand((16, 64, 512))  # Example input
    # mask = torch.ones((16, 64, 64), dtype=torch.bool)
    # output = encoder(data, mask)
    # print(data.shape)
    # print(output.shape)

    # # Instantiate the Encoder
    # encoder = Encoder(3, 512, 2048, 0.1)

    # # Create random input and mask tensors
    # input_data = torch.rand((16, 64, 512))  # Example input data
    # mask = torch.zeros((16, 64), dtype=torch.bool)  # Example mask (all positions are unmasked)

    # # Forward pass through the encoder
    # output = encoder(input_data, mask)

    # # Print shapes for input and output tensors
    # print("Input shape:", input_data.shape)
    # print("Output shape:", output.shape)

    # 假设输入序列的词汇表大小为1000，目标序列的词汇表大小为1000
    input_vocab_size = 1000
    target_vocab_size = 1000

    # 创建Transformer模型
    model = transformer(input_vocab_size, target_vocab_size,
                        src_pad_idx=0, trg_pad_idx=0)

    # 假设输入序列和目标序列的batch_size为32，序列长度为10
    inputs = torch.randint(0, input_vocab_size, (32, 10))
    targets = torch.randint(0, target_vocab_size, (32, 10))

    # 调用模型
    output = model(inputs, targets)

    print("输入维度:", inputs.shape)  # (32, 10)
    print("目标维度:", targets.shape)  # (32, 10)
    print("输出维度:", output.shape)  # (32, 10, 1000)
```


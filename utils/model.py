import torch


class SelfAttention(torch.nn.Module):
    """
    Computes Query=Key-Value attention by weighting the value vectors by the self-attention.
    """

    def __init__(self, embedding_size, attention_value_size):
        """
        W_Q,W_K,W_V are all (embedding_size,attention_matrix_size)
        Q,K,V are all (n_tokens,attention_value_size)
        """

        super().__init__()
        self.attention_value_size = torch.tensor(attention_value_size)
        self.W_Q = torch.rand(size=(embedding_size, attention_value_size))
        self.W_K = torch.rand(size=(embedding_size, attention_value_size))
        self.W_V = torch.rand(size=(embedding_size, attention_value_size))

    def forward(self, token_embeddings, mask):
        """
        Receives:
        token_embeddings: (n_tokens,embedding_size)
        mask: (1,n_tokens) with 1 in positions to attend, -inf otherwise

        Returns a matrix (n_tokens,attention_value_size) which is the result of the Q,K,V attention.
        """

        Q = torch.matmul(token_embeddings, self.W_Q)
        K = torch.matmul(token_embeddings, self.W_K)
        V = torch.matmul(token_embeddings, self.W_V)

        normalize_value = torch.sqrt(self.attention_value_size)

        Q_mm_K = torch.matmul(Q, torch.transpose(K, 0, 1))

        # mask out due to causal attention
        Q_mm_K *= mask

        softmax_val = torch.nn.functional.softmax(Q_mm_K / normalize_value, dim=1)
        self_attention_matrix = torch.matmul(softmax_val, V)

        return self_attention_matrix


class FeedForward(torch.nn.Module):
    """
    Forward layer as described in Attention is all you need, equation (2)
    """

    def __init__(self, embedding_size, ffn_inner_layer):

        super().__init__()
        self.W_1 = torch.rand(size=(embedding_size, ffn_inner_layer))
        self.W_2 = torch.rand(size=(ffn_inner_layer, embedding_size))
        self.b_1 = torch.rand(size=(1, ffn_inner_layer))
        self.b_2 = torch.rand(size=(1, embedding_size))

    def forward(self, token_embeddings):
        """
        Receives:
        token_embeddings: (n_tokens, embedding_size)

        Returns (n_tokens, embedding_size) after applying ReLU non linearity
        """

        layer_1 = torch.matmul(token_embeddings, self.W_1) + self.b_1
        relu = torch.relu(layer_1)
        layer_2 = torch.matmul(relu, self.W_2) + self.b_2

        return layer_2


class MultiHeadAttention(torch.nn.Module):
    """
    Collects the concatenated attention heads and pools them back to the correct embedding_size.
    """

    def __init__(self, n_attention_heads, attention_value_size, embedding_size):
        super().__init__()

        self.W_Pool = torch.rand(
            size=(n_attention_heads * attention_value_size, embedding_size)
        )

    def forward(self, concatenated_heads):
        """
        Receives concatenated_heads: (n_tokens,n_attention_heads*attention_value_size)
        Returns (n_tokens, embedding_size)
        """

        return torch.matmul(concatenated_heads, self.W_Pool)


class DecoderModule(torch.nn.Module):
    """
    Unified decoder block. Computes multi-head attention, residual layer, feedforward layer and once again residual layer.
    """

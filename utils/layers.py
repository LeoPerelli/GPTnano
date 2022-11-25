import torch

#! Add batch size, probably the only problem is in the softmax transpose etc that are indexed with 1. Instead, I should put relative positions with -1


class SelfAttention(torch.nn.Module):
    """
    Computes Query-Key-Value attention by weighting the value vectors by the self-attention.
    """

    def __init__(self, embedding_size, attention_value_size, n_tokens):
        """
        W_Q,W_K,W_V are all (embedding_size,attention_matrix_size)
        mask: (n_tokens,n_tokens) with 1 in positions to attend, -inf otherwise
        """

        super().__init__()
        self.attention_value_size = torch.tensor(attention_value_size)
        self.W_Q = torch.rand(size=(embedding_size, attention_value_size))
        self.W_K = torch.rand(size=(embedding_size, attention_value_size))
        self.W_V = torch.rand(size=(embedding_size, attention_value_size))

        # Create mask matrix: lower triangluar is attended (including diagonal), upper triangular is ignored
        mask_attend = torch.ones(size=(n_tokens, n_tokens))
        mask_ignore = -float("inf") * torch.ones(size=(n_tokens, n_tokens))
        self.mask = torch.tril(mask_attend) + torch.triu(mask_ignore, diagonal=1)

    def forward(self, token_embeddings):
        """
        Receives:
        token_embeddings: (n_tokens,embedding_size)

        Returns a matrix (n_tokens,attention_value_size) which is the result of the Q,K,V attention.
        """

        # Q,K,V are all (n_tokens,attention_value_size)
        Q = torch.matmul(token_embeddings, self.W_Q)
        K = torch.matmul(token_embeddings, self.W_K)
        V = torch.matmul(token_embeddings, self.W_V)

        normalize_value = torch.sqrt(self.attention_value_size)

        Q_mm_K = torch.matmul(Q, torch.transpose(K, -1, -2))

        # mask out due to causal attention
        Q_mm_K *= self.mask

        softmax_val = torch.nn.functional.softmax(Q_mm_K / normalize_value, dim=-1)
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

    def __init__(
        self, n_attention_heads, attention_value_size, embedding_size, n_tokens
    ):
        super().__init__()

        self.W_Pool = torch.rand(
            size=(n_attention_heads * attention_value_size, embedding_size)
        )

        self.attention_modules = [
            SelfAttention(
                embedding_size=embedding_size,
                attention_value_size=attention_value_size,
                n_tokens=n_tokens,
            )
            for i in range(n_attention_heads)
        ]

    def compute_and_concat_heads(self, token_embeddings):

        attention_heads = []
        for attention_module in self.attention_modules:
            attention_heads.append(
                attention_module.forward(token_embeddings=token_embeddings)
            )

        concatenated_attention_heads = torch.cat(attention_heads, -1)

        return concatenated_attention_heads

    def forward(self, token_embeddings):
        """
        Receives:
        token_embeddings: (n_tokens, embedding_size)

        Computes the output from each attention head, then concatenates them and returns the pooled attention

        Returns (n_tokens, embedding_size)
        """

        # concatenated_heads: (n_tokens,n_attention_heads*attention_value_size)
        concatenated_heads = self.compute_and_concat_heads(
            token_embeddings=token_embeddings
        )

        return torch.matmul(concatenated_heads, self.W_Pool)


class LayerNorm(torch.nn.Module):
    """
    Computes LayerNorm with 2 learnable parameters
    """

    def __init__(self, embedding_size, eps=1e-5):

        super().__init__()

        self.gamma = torch.rand(size=(1, 1))
        self.beta = torch.rand(size=(1, embedding_size))
        self.eps = eps

    def normalize(self, token_embeddings):

        mean = torch.mean(token_embeddings, dim=-1).unsqueeze(-1)
        var = torch.var(token_embeddings, dim=-1, unbiased=False).unsqueeze(-1)

        norm = (
            self.gamma * (token_embeddings - mean) / torch.sqrt(var + self.eps)
            + self.beta
        )

        return norm


class LinearHead(torch.nn.Module):
    """
    Plain linear head with softmax to predict over vocabulary
    """

    def __init__(self, embedding_size, vocabulary_size):

        super().__init__()
        self.W_1 = torch.rand(size=(embedding_size, vocabulary_size))
        self.b_1 = torch.rand(size=(1, vocabulary_size))

    def forward(self, token_embeddings):

        linear_layer = torch.matmul(token_embeddings, self.W_1) + self.b_1

        return torch.nn.functional.softmax(linear_layer, dim=-1)


class DecoderModule(torch.nn.Module):
    """
    Unified decoder block. Computes multi-head attention, residual layer, feedforward layer and once again residual layer.
    """

    def __init__(
        self,
        n_attention_heads,
        attention_value_size,
        embedding_size,
        n_tokens,
        ffn_inner_layer,
    ):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(
            n_attention_heads=n_attention_heads,
            attention_value_size=attention_value_size,
            embedding_size=embedding_size,
            n_tokens=n_tokens,
        )

        self.feed_forward_layer = FeedForward(
            embedding_size=embedding_size, ffn_inner_layer=ffn_inner_layer
        )

        self.layer_norm = LayerNorm(embedding_size=embedding_size)

    def forward(self, token_embeddings):
        """
        Receives:
        token_embeddings: (n_tokens, embedding_size)

        Computes the entire decoder module operations: multi head self attention, layer normalization, feed forward pass

        Returns (n_tokens, embedding_size)
        """

        # Multi head attention and residuals
        token_embeddings += self.multi_head_attention.forward(
            token_embeddings=token_embeddings
        )

        # Layer normalization
        token_embeddings = self.layer_norm.normalize(token_embeddings=token_embeddings)

        # Feed forward network with residuals
        token_embeddings += self.feed_forward_layer.forward(
            token_embeddings=token_embeddings
        )

        # Layer normalization
        token_embeddings = self.layer_norm.normalize(token_embeddings=token_embeddings)

        return token_embeddings

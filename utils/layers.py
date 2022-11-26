import torch

#! Write unit test that checks for nan (amd shape?)


class SelfAttentionHead(torch.nn.Module):
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
        self.W_Q = torch.nn.Linear(
            in_features=embedding_size, out_features=attention_value_size, bias=False
        )
        self.W_K = torch.nn.Linear(
            in_features=embedding_size, out_features=attention_value_size, bias=False
        )
        self.W_V = torch.nn.Linear(
            in_features=embedding_size, out_features=attention_value_size, bias=False
        )

        # Create mask matrix: lower triangluar is attended (including diagonal), upper triangular is ignored
        self.mask_attend = torch.ones(size=(n_tokens, n_tokens))
        self.mask_ignore = -float("inf") * torch.ones(size=(n_tokens, n_tokens))

    def forward(self, token_embeddings):
        """
        Receives:
        token_embeddings: (n_tokens,embedding_size)

        Returns a matrix (n_tokens,attention_value_size) which is the result of the Q,K,V attention.
        """

        # Q,K,V are all (n_tokens,attention_value_size)
        Q = self.W_Q(token_embeddings)
        K = self.W_K(token_embeddings)
        V = self.W_V(token_embeddings)

        normalize_value = torch.sqrt(self.attention_value_size)

        Q_mm_K = torch.matmul(Q, torch.transpose(K, -1, -2))

        # mask out due to causal attention
        Q_mm_K *= torch.tril(self.mask_attend)
        Q_mm_K += torch.triu(self.mask_ignore, diagonal=1)

        softmax_val = torch.nn.functional.softmax(Q_mm_K / normalize_value, dim=-1)
        self_attention_matrix = torch.matmul(softmax_val, V)

        return self_attention_matrix


class FeedForward(torch.nn.Module):
    """
    Forward layer as described in Attention is all you need, equation (2)
    """

    def __init__(self, embedding_size, ffn_inner_layer):

        super().__init__()
        self.l1 = torch.nn.Linear(
            in_features=embedding_size, out_features=ffn_inner_layer
        )
        self.l2 = torch.nn.Linear(
            in_features=ffn_inner_layer, out_features=embedding_size
        )

    def forward(self, token_embeddings):
        """
        Receives:
        token_embeddings: (n_tokens, embedding_size)

        Returns (n_tokens, embedding_size) after applying ReLU non linearity
        """

        layer_1 = self.l1(token_embeddings)
        relu = torch.relu(layer_1)
        layer_2 = self.l2(relu)

        return layer_2


class MultiHeadAttention(torch.nn.Module):
    """
    Collects the concatenated attention heads and pools them back to the correct embedding_size.
    """

    def __init__(
        self, n_attention_heads, attention_value_size, embedding_size, n_tokens
    ):
        super().__init__()

        self.attention_heads = torch.nn.ModuleList(
            [
                SelfAttentionHead(
                    embedding_size=embedding_size,
                    attention_value_size=attention_value_size,
                    n_tokens=n_tokens,
                )
                for i in range(n_attention_heads)
            ]
        )

        self.pool_layer = torch.nn.Linear(
            in_features=n_attention_heads * attention_value_size,
            out_features=embedding_size,
            bias=False,
        )

    def compute_and_concat_heads(self, token_embeddings):

        attention_heads = []
        for attention_module in self.attention_heads:
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

        return self.pool_layer(concatenated_heads)


class LayerNorm(torch.nn.Module):
    """
    Computes LayerNorm with 2 learnable parameters
    """

    def __init__(self, embedding_size, eps=1e-5):

        super().__init__()

        self.gamma = torch.nn.Parameter(torch.rand(size=(1, 1)))
        self.beta = torch.nn.Parameter(torch.rand(size=(1, embedding_size)))
        self.eps = eps

    def normalize(self, token_embeddings):

        mean = torch.mean(token_embeddings, dim=-1).unsqueeze(-1)
        var = torch.var(token_embeddings, dim=-1, unbiased=False).unsqueeze(-1)

        norm = (
            self.gamma * (token_embeddings - mean) / torch.sqrt(var + self.eps)
            + self.beta
        )

        return norm


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

        self.layer_norm = LayerNorm(embedding_size=embedding_size)

        self.feed_forward_layer = FeedForward(
            embedding_size=embedding_size, ffn_inner_layer=ffn_inner_layer
        )

        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, token_embeddings):
        """
        Receives:
        token_embeddings: (n_tokens, embedding_size)

        Computes the entire decoder module operations: multi head self attention, layer normalization, feed forward pass

        Returns (n_tokens, embedding_size)
        """

        # Multi head attention and residuals
        attention_output = self.multi_head_attention.forward(
            token_embeddings=token_embeddings
        )
        token_embeddings += self.dropout(attention_output)

        # Layer normalization
        token_embeddings = self.layer_norm.normalize(token_embeddings=token_embeddings)

        # Feed forward network with residuals
        feed_forward_output = self.feed_forward_layer.forward(
            token_embeddings=token_embeddings
        )
        token_embeddings += self.dropout(feed_forward_output)

        # Layer normalization
        token_embeddings = self.layer_norm.normalize(token_embeddings=token_embeddings)

        return token_embeddings


class LinearHead(torch.nn.Module):
    """
    Plain linear head with log_softmax to predict over vocabulary. We use log_softmax rather than softmax since we will use Negative-Log-Likelihood loss later on
    """

    def __init__(self, embedding_size, vocabulary_size):

        super().__init__()
        self.linear = torch.nn.Linear(
            in_features=embedding_size, out_features=vocabulary_size
        )

    def forward(self, token_embeddings):

        linear_layer = self.linear(token_embeddings)

        return torch.nn.functional.log_softmax(linear_layer, dim=-1)

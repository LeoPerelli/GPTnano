import torch

#! Write unit test that checks for nan (amd shape?)
#! Add generate function
#! Fix initialization of weights?


# use backprop to chart dependencies. Your deep learning code will often contain complicated, vectorized, and broadcasted operations.
# A relatively common bug I’ve come across a few times is that people get this wrong
# (e.g. they use view instead of transpose/permute somewhere) and inadvertently mix information across the batch dimension.
#  It is a depressing fact that your network will typically still train okay because it will learn to ignore data from the other examples.
# One way to debug this (and other related problems) is to set the loss to be something trivial like the sum of all outputs of example i,
#  run the backward pass all the way to the input, and ensure that you get a non-zero gradient only on the i-th input.
# The same strategy can be used to e.g. ensure that your autoregressive model at time t only depends on 1..t-1.
# More generally, gradients give you information about what depends on what in your network, which can be useful for debugging.


class MultiHeadAttention(torch.nn.Module):
    """
    Computes Query-Key-Value attention by weighting the value vectors by the self-attention.
    """

    def __init__(self, embedding_size, n_tokens, n_heads):
        """
        W_Q,W_K,W_V contain the stacked heads by column
        mask: (n_tokens,n_tokens) with 1 in positions to attend, -inf otherwise
        """

        super().__init__()
        self.head_dim = int(embedding_size / n_heads)
        self.W_Q = torch.nn.Linear(
            in_features=embedding_size, out_features=embedding_size, bias=False
        )
        self.W_K = torch.nn.Linear(
            in_features=embedding_size, out_features=embedding_size, bias=False
        )
        self.W_V = torch.nn.Linear(
            in_features=embedding_size, out_features=embedding_size, bias=False
        )

        # Create mask matrix: lower triangluar is attended (including diagonal), upper triangular is ignored
        self.mask_attend = torch.ones(size=(n_tokens, n_tokens))
        self.mask_ignore = -float("inf") * torch.ones(size=(n_tokens, n_tokens))

        self.pool_layer = torch.nn.Linear(
            in_features=embedding_size,
            out_features=embedding_size,
            bias=False,
        )

    def forward(self, token_embeddings):
        """
        Receives:
        token_embeddings: (batch_size, n_tokens,embedding_size)

        Returns a matrix (batch, n_tokens,embedding_size) which is the result of the Q,K,V attention.
        """
        batch_size, n_tokens, embedding_size = token_embeddings.shape

        # Q,K,V are all (batch_size,n_tokens,embedding_size)
        Q = self.W_Q(token_embeddings)
        K = self.W_K(token_embeddings)
        V = self.W_V(token_embeddings)

        # (batch_size, n_heads, n_tokens, head_embedding_dim)
        Q = Q.view(batch_size, -1, n_tokens, self.head_dim)
        K = K.view(batch_size, -1, n_tokens, self.head_dim)
        V = V.view(batch_size, -1, n_tokens, self.head_dim)

        # (batch_size, n_heads, n_tokens, n_tokens)
        Q_mm_K = Q @ torch.transpose(K, -1, -2)

        # mask out due to causal attention
        Q_mm_K = Q_mm_K * torch.tril(self.mask_attend)
        Q_mm_K = Q_mm_K + torch.triu(self.mask_ignore, diagonal=1)

        softmax_val = torch.nn.functional.softmax(Q_mm_K / self.head_dim**0.5, dim=-1)

        # (batch_size, n_heads, n_tokens, head_embedding_dim)
        self_attention_matrix = softmax_val @ V

        self_attention_matrix = (
            self_attention_matrix.transpose(1, 2)
            .contiguous()
            .view(batch_size, n_tokens, embedding_size)
        )

        pooled_attention = self.pool_layer(self_attention_matrix)

        return pooled_attention


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


class LayerNorm(torch.nn.Module):
    """
    Computes LayerNorm with 2 learnable parameters
    """

    def __init__(self, embedding_size, eps=1e-5):

        super().__init__()

        self.gamma = torch.nn.Parameter(torch.rand(size=(1, 1)))
        self.beta = torch.nn.Parameter(torch.rand(size=(1, embedding_size)))
        self.eps = eps

    def forward(self, token_embeddings):

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
        embedding_size,
        n_tokens,
        ffn_inner_layer,
        p_dropout,
    ):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(
            embedding_size=embedding_size, n_heads=n_attention_heads, n_tokens=n_tokens
        )

        # self.layer_norm_1 = LayerNorm(embedding_size=embedding_size)
        # self.layer_norm_2 = LayerNorm(embedding_size=embedding_size)
        self.layer_norm_1 = torch.nn.LayerNorm(embedding_size)
        self.layer_norm_2 = torch.nn.LayerNorm(embedding_size)

        self.feed_forward_layer = FeedForward(
            embedding_size=embedding_size, ffn_inner_layer=ffn_inner_layer
        )

        self.dropout_1 = torch.nn.Dropout(p=p_dropout)
        self.dropout_2 = torch.nn.Dropout(p=p_dropout)

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
        token_embeddings = token_embeddings + self.dropout_1(attention_output)

        # Layer normalization
        token_embeddings = self.layer_norm_1.forward(token_embeddings)

        # Feed forward network with residuals
        feed_forward_output = self.feed_forward_layer.forward(
            token_embeddings=token_embeddings
        )
        token_embeddings = token_embeddings + self.dropout_2(feed_forward_output)

        # Layer normalization
        token_embeddings = self.layer_norm_2.forward(token_embeddings)

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

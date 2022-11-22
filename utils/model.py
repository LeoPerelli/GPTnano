import torch


class SelfAttention(torch.nn.Module):
    """
    Returns a matrix (n_tokens,attention_matrix_size) which is the result of the Q,K,V attention.
    It is computed by weighting the value vectors by the self-attention
    """

    def __init__(self, embedding_size, attention_matrix_size):
        # W_Q,W_K,W_V are all (embedding_size,attention_matrix_size)

        super().__init__()
        self.attention_matrix_size = torch.tensor(attention_matrix_size)
        self.W_Q = torch.rand(size=(embedding_size, attention_matrix_size))
        self.W_K = torch.rand(size=(embedding_size, attention_matrix_size))
        self.W_V = torch.rand(size=(embedding_size, attention_matrix_size))

    def forward(self, tokens):
        # Q,K,V are all (n_tokens,attention_matrix_size)

        Q = torch.matmul(tokens, self.W_Q)
        K = torch.matmul(tokens, self.W_K)
        V = torch.matmul(tokens, self.W_V)

        normalize_value = torch.sqrt(self.attention_matrix_size)

        Q_mm_K = torch.matmul(Q, torch.transpose(K, 0, 1))
        softmax_val = torch.nn.functional.softmax(Q_mm_K / normalize_value)
        self_attention_matrix = torch.matmul(softmax_val, V)

        return self_attention_matrix

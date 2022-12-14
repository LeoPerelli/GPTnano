import torch
from utils.layers import DecoderModule, LinearHead


class GPT(torch.nn.Module):
    """
    GPT model, receives the tokenized inputs and performs inference
    """

    def __init__(
        self,
        n_attention_heads,
        attention_value_size,
        embedding_size,
        n_tokens,
        ffn_inner_layer,
        n_decoders,
        vocabulary_size,
        p_dropout=0.1,
    ):

        super().__init__()

        # Position and token encodings
        self.lookup_encodings = torch.nn.Embedding(
            embedding_dim=embedding_size, num_embeddings=vocabulary_size
        )
        self.lookup_positional = torch.rand(size=(n_tokens, embedding_size))

        self.decoder_modules = torch.nn.ModuleList(
            [
                DecoderModule(
                    n_attention_heads=n_attention_heads,
                    embedding_size=embedding_size,
                    n_tokens=n_tokens,
                    ffn_inner_layer=ffn_inner_layer,
                    p_dropout=p_dropout,
                )
                for _ in range(n_decoders)
            ]
        )

        self.dropout = torch.nn.Dropout(p=0.1)

        self.head = LinearHead(
            embedding_size=embedding_size, vocabulary_size=vocabulary_size
        )

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def get_token_encodings(self, tokens):
        """
        Uses lookup table to get the token encoding and sums the learnt positional encodings
        """

        token_encoding_positional = (
            self.lookup_encodings(tokens) + self.lookup_positional
        )

        return self.dropout(token_encoding_positional)

    def forward(self, tokens):

        token_embeddings = self.get_token_encodings(tokens)

        for decoder_module in self.decoder_modules:
            token_embeddings = decoder_module.forward(token_embeddings=token_embeddings)

        output_distribution = self.head.forward(token_embeddings=token_embeddings)

        return output_distribution

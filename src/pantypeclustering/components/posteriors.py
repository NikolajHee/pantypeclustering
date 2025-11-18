# def observation_model(self, z: Tensor) -> Distribution:
#     """return the distribution `p(x|z)`"""
#     px_logits = self.decoder(z)
#     px_logits = px_logits.view(-1, *self.input_shape)  # reshape the output
#     return Bernoulli(logits=px_logits, validate_args=False)

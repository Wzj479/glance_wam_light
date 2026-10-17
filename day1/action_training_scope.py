"""Parameter and module-mode policy, independent of the world-model implementation."""
def configure_action_scope(model, scope):
    model.requires_grad_(False)
    head = model.action_model
    if scope == 'all':
        head.requires_grad_(True)
    elif scope == 'crossattn_decoder':
        blocks = [b for b in head.model.transformer_blocks if b.cross_attention_dim is not None]
        if not blocks:
            raise ValueError('No cross-attention blocks found')
        for block in blocks:
            block.attn1.requires_grad_(True)
        head.action_decoder.requires_grad_(True)
    else:
        raise ValueError(scope)
    if head.state_encoder is not None:
        head.state_encoder.requires_grad_(False)
    trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    frozen = [(n,p) for n,p in model.named_parameters() if not p.requires_grad]
    if not trainable or any(not n.startswith('action_model.') for n,p in trainable):
        raise ValueError('Unexpected trainable parameter scope')
    return trainable, frozen


def set_action_training_modes(model, scope):
    model.eval()
    if scope == 'all':
        model.action_model.train()
    else:
        # Frozen self-attention/FFN/dropout remain in eval mode; gradients still
        # propagate through frozen operations into the trainable cross-attention.
        for block in model.action_model.model.transformer_blocks:
            if block.cross_attention_dim is not None:
                block.attn1.train()
        model.action_model.action_decoder.train()

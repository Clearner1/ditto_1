import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from .layers import AttentionLayer, GlobalAttentionLayer, StructAttentionLayer

lm_mp = {
    'roberta': 'roberta-base',
    'distilbert': 'distilbert-base-uncased'
}

class DittoHGATModel(nn.Module):
    """Enhanced Ditto model with Hierarchical Graph Attention Networks"""
    
    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8, num_views=3):
        super().__init__()
        # Load pre-trained language model
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug
        self.num_views = num_views

        # Get hidden size from BERT config
        hidden_size = self.bert.config.hidden_size

        # Initialize attention layers
        self.token_attention = AttentionLayer(hidden_size)
        self.global_attention = GlobalAttentionLayer(hidden_size)
        self.struct_attention = StructAttentionLayer(hidden_size)

        # Initialize view-specific layers
        self.view_fcs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_views)
        ])

        # Final classification layer
        self.fc = nn.Linear(hidden_size, 2)

    def encode_sequence(self, input_ids, attention_mask=None):
        """Encode a sequence using BERT and attention mechanisms
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            tensor: (batch_size, hidden_size)
        """
        # Get BERT outputs
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        cls_output = sequence_output[:, 0]  # (batch_size, hidden_size)

        # Apply different attention mechanisms
        token_repr = self.token_attention(sequence_output, attention_mask)
        global_repr = self.global_attention(sequence_output, attention_mask)
        
        # Create different views of the input
        views = []
        views.append(cls_output)  # CLS token view
        views.append(token_repr)  # Token-level attention view
        views.append(global_repr)  # Global attention view

        # Apply view-specific transformations
        views = [fc(view) for fc, view in zip(self.view_fcs, views)]

        # Combine views using structural attention
        combined_repr = self.struct_attention(views)
        
        return combined_repr

    def forward(self, x1, x2=None):
        """Forward pass with optional MixDA augmentation
        
        Args:
            x1: (batch_size, seq_len) - original input
            x2: (batch_size, seq_len) - augmented input (optional)
            
        Returns:
            tensor: (batch_size, 2) - class logits
        """
        x1 = x1.to(self.device)
        
        if x2 is not None:
            # MixDA augmentation
            x2 = x2.to(self.device)
            
            # Get attention masks
            attention_mask1 = (x1 != self.bert.config.pad_token_id).float()
            attention_mask2 = (x2 != self.bert.config.pad_token_id).float()
            
            # Encode both sequences
            enc1 = self.encode_sequence(x1, attention_mask1)
            enc2 = self.encode_sequence(x2, attention_mask2)
            
            # Apply MixDA
            aug_lam = torch.tensor(
                np.random.beta(self.alpha_aug, self.alpha_aug),
                device=self.device
            )
            mixed_enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
            return self.fc(mixed_enc)
        else:
            # Regular forward pass
            attention_mask = (x1 != self.bert.config.pad_token_id).float()
            enc = self.encode_sequence(x1, attention_mask)
            return self.fc(enc)

    def configure_optimizers(self, lr=1e-5, eps=1e-8, weight_decay=0.01):
        """Configure optimizers for training
        
        Args:
            lr: learning rate
            eps: epsilon for Adam optimizer
            weight_decay: weight decay for regularization
            
        Returns:
            optimizer: AdamW optimizer
        """
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer 
                       if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer 
                       if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        
        return torch.optim.AdamW(optimizer_grouped_parameters, 
                               lr=lr, 
                               eps=eps)

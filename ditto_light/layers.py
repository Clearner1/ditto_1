import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Token-level attention layer"""
    def __init__(self, hidden_size, alpha=0.2):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(size=(hidden_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, sequence_output, attention_mask=None):
        """
        Args:
            sequence_output: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        Returns:
            weighted output: (batch_size, hidden_size)
        """
        # Calculate attention scores
        attention = torch.matmul(sequence_output, self.a).squeeze(-1)  # (batch_size, seq_len)
        
        if attention_mask is not None:
            # Set padding attention scores to -inf
            attention = attention.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention, dim=1)  # (batch_size, seq_len)
        
        # Apply attention weights to sequence output
        weighted_output = torch.bmm(attention_weights.unsqueeze(1), sequence_output).squeeze(1)
        return weighted_output


class GlobalAttentionLayer(nn.Module):
    """Global attention layer for sequence representation"""
    def __init__(self, hidden_size, alpha=0.2):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.a = nn.Parameter(torch.zeros(size=(hidden_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, sequence_output, attention_mask=None):
        """
        Args:
            sequence_output: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        Returns:
            context_vector: (batch_size, hidden_size)
        """
        # Transform sequence
        hidden = self.linear(sequence_output)  # (batch_size, seq_len, hidden_size)
        
        # Calculate attention scores
        attention = torch.matmul(hidden, self.a).squeeze(-1)  # (batch_size, seq_len)
        attention = self.leakyrelu(attention)
        
        if attention_mask is not None:
            # Set padding attention scores to -inf
            attention = attention.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention, dim=1)  # (batch_size, seq_len)
        
        # Calculate context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), sequence_output).squeeze(1)
        return context_vector


class StructAttentionLayer(nn.Module):
    """Structure-aware attention layer for combining different views"""
    def __init__(self, hidden_size, alpha=0.2):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(size=(hidden_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, representations):
        """
        Args:
            representations: tensor of shape (batch_size, num_attrs, hidden_size)
        Returns:
            weighted_sum: (batch_size, hidden_size)
        """
        # Calculate attention scores
        attention = torch.matmul(representations, self.a).squeeze(-1)  # (batch_size, num_attrs)
        attention = self.leakyrelu(attention)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention, dim=1)  # (batch_size, num_attrs)
        
        # Calculate weighted sum
        weighted_sum = torch.bmm(attention_weights.unsqueeze(1), representations).squeeze(1)
        return weighted_sum

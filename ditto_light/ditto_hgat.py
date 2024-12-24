import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
import sklearn.metrics as metrics
import os
import time

from .layers import AttentionLayer, GlobalAttentionLayer, StructAttentionLayer

lm_mp = {
    'roberta': 'roberta-base',
    'distilbert': 'distilbert-base-uncased'
}

class DittoHGATModel(nn.Module):
    """Enhanced Ditto model with Hierarchical Graph Attention Networks and Number Perception"""
    
    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8, 
                 use_number_perception=False, number_perception_weight=0.5):
        super().__init__()
        # Load pre-trained language model
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug
        self.use_number_perception = use_number_perception
        self.number_perception_weight = number_perception_weight  # 存储权重参数

        # Get hidden size from BERT config
        hidden_size = self.bert.config.hidden_size

        # Initialize attention layers
        self.token_attention = AttentionLayer(hidden_size)
        self.global_attention = GlobalAttentionLayer(hidden_size)
        self.struct_attention = StructAttentionLayer(hidden_size)

        # 如果使用数字感知，添加特征融合层
        if use_number_perception:
            # 数字感知特征投影层
            self.num_projection = nn.Sequential(
                nn.Linear(1, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, hidden_size)
            )
            
            # 特征融合层
            self.fusion_layer = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size)
            )

        # Final classification layer
        self.fc = nn.Linear(hidden_size, 2)

    def encode_attribute(self, input_ids, attention_mask=None):
        """Encode a single attribute pair using BERT and attention mechanisms"""
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        
        # Apply different attention mechanisms
        token_repr = self.token_attention(sequence_output, attention_mask)
        global_repr = self.global_attention(sequence_output, attention_mask)
        
        # Combine representations
        combined_repr = token_repr + global_repr
        
        return combined_repr

    def forward(self, encoded_pairs, attention_masks, num_similarities=None, labels=None):
        """Forward pass with attribute-level processing and number perception
        
        Args:
            encoded_pairs: tensor of shape [num_attrs, batch_size, seq_len]
            attention_masks: tensor of shape [batch_size, seq_len, num_attrs]
            num_similarities: tensor of shape [batch_size], optional
            labels: tensor of shape [batch_size], optional
            
        Returns:
            logits: tensor of shape [batch_size, 2]
            loss: scalar tensor (if labels provided)
        """
        num_attrs, batch_size, seq_len = encoded_pairs.size()
        
        # Move tensors to device
        encoded_pairs = encoded_pairs.to(self.device)
        attention_masks = attention_masks.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        if num_similarities is not None:
            num_similarities = num_similarities.to(self.device)
        
        # Process each attribute pair
        attr_outputs = []
        for i in range(num_attrs):
            # Get current attribute pair for all batches
            curr_pairs = encoded_pairs[i]  # [batch_size, seq_len]
            curr_masks = attention_masks[:, :, i]  # [batch_size, seq_len]
            
            # Encode the attribute pair
            attr_repr = self.encode_attribute(curr_pairs, curr_masks)
            attr_outputs.append(attr_repr)
        
        # Stack all attribute representations
        attr_outputs = torch.stack(attr_outputs, dim=1)  # [batch_size, num_attrs, hidden_size]
        
        # Apply structural attention to combine attribute representations
        text_repr = self.struct_attention(attr_outputs)  # [batch_size, hidden_size]
        
        # 如果使用数字感知特征，进行特征融合
        if self.use_number_perception and num_similarities is not None:
            # 将数字相似度转换为特征向量
            num_repr = self.num_projection(num_similarities.unsqueeze(-1))  # [batch_size, hidden_size]
            
            # 使用权重进行特征融合
            # text_repr 和 num_repr 都是 [batch_size, hidden_size]
            # 使用 number_perception_weight 来控制两种特征的比重
            combined_repr = torch.cat([
                (1 - self.number_perception_weight) * text_repr,  # 文本特征权重
                self.number_perception_weight * num_repr          # 数字特征权重
            ], dim=-1)
            
            # 通过融合层
            final_repr = self.fusion_layer(combined_repr)
        else:
            final_repr = text_repr
        
        # Get logits
        logits = self.fc(final_repr)
        
        if labels is not None:
            # Calculate loss if labels are provided
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        
        return logits


def train_epoch(model, iterator, optimizer, scheduler=None, hp=None):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(iterator):
        optimizer.zero_grad()
        
        # Get the inputs
        encoded_pairs, attention_masks, labels, num_similarities = batch
        
        # Forward pass
        loss, logits = model(encoded_pairs, attention_masks, num_similarities, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        total_loss += loss.item()
        
        if hp and hp.fp16:
            with torch.cuda.amp.autocast():
                if i % 10 == 0:
                    print(f"step: {i}, loss: {loss.item():.4f}")
        else:
            if i % 10 == 0:
                print(f"step: {i}, loss: {loss.item():.4f}")
                
    return total_loss / len(iterator)


def evaluate(model, iterator):
    """Evaluate the model on a dataset"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in iterator:
            encoded_pairs, attention_masks, labels, num_similarities = batch
            
            # Forward pass
            logits = model(encoded_pairs, attention_masks, num_similarities)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    precision = metrics.precision_score(all_labels, all_preds)
    recall = metrics.recall_score(all_labels, all_preds)
    f1 = metrics.f1_score(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train(trainset, validset, testset, run_tag, hp, device='cuda'):
    """Train and evaluate the model"""
    # Create data loaders with the custom collate function
    train_loader = DataLoader(
        trainset,
        batch_size=hp.batch_size,
        shuffle=True,
        collate_fn=trainset.pad,
        num_workers=0
    )
    valid_loader = DataLoader(
        validset,
        batch_size=hp.batch_size,
        shuffle=False,
        collate_fn=validset.pad,
        num_workers=0
    )
    test_loader = DataLoader(
        testset,
        batch_size=hp.batch_size,
        shuffle=False,
        collate_fn=testset.pad,
        num_workers=0
    )
    
    # Initialize model
    model = DittoHGATModel(
        device=device,
        lm=hp.lm,
        use_number_perception=hp.use_number_perception,
        number_perception_weight=hp.number_perception_weight
    ).to(device)
    
    # Initialize optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=hp.lr)
    
    # Initialize scheduler with warmup
    num_training_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    num_warmup_steps = num_training_steps // 10  # 10% of training steps for warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize tensorboard
    writer = SummaryWriter(f'runs/{run_tag}')
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join('checkpoints', run_tag)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Training loop
    best_dev_f1 = 0.0
    best_epoch = 0
    patience = 10  # Early stopping patience
    no_improvement = 0
    
    for epoch in range(hp.n_epochs):
        start_time = time.time()
        
        # Train for one epoch
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, hp)
        
        # Evaluate
        model.eval()
        dev_metrics = evaluate(model, valid_loader)
        test_metrics = evaluate(model, test_loader)
        
        # Log metrics
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('valid/f1', dev_metrics['f1'], epoch)
        writer.add_scalar('test/f1', test_metrics['f1'], epoch)
        
        # Save best model and check for early stopping
        if dev_metrics['f1'] > best_dev_f1:
            best_dev_f1 = dev_metrics['f1']
            best_epoch = epoch
            no_improvement = 0
            if hp.save_model:
                model_path = os.path.join(checkpoint_dir, 'model_best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_f1': best_dev_f1,
                }, model_path)
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f'Early stopping after {epoch + 1} epochs')
                break
        
        # Print progress
        print(f'Epoch {epoch+1}/{hp.n_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Valid F1: {dev_metrics["f1"]:.4f}')
        print(f'  Test F1: {test_metrics["f1"]:.4f}')
        print(f'  Time: {time.time() - start_time:.2f}s')
        print(f'  Best epoch: {best_epoch+1} with F1: {best_dev_f1:.4f}')
    
    # Load best model and evaluate on test set
    if hp.save_model:
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'model_best.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    final_test_metrics = evaluate(model, test_loader)
    
    print('\nFinal Test Results:')
    print(f'  Best epoch: {best_epoch+1}')
    print(f'  Accuracy: {final_test_metrics["accuracy"]:.4f}')
    print(f'  Precision: {final_test_metrics["precision"]:.4f}')
    print(f'  Recall: {final_test_metrics["recall"]:.4f}')
    print(f'  F1: {final_test_metrics["f1"]:.4f}')
    
    writer.close()
    return final_test_metrics 
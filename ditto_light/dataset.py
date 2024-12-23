import torch

from torch.utils import data
from transformers import AutoTokenizer

from .augment import Augmenter

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)


class DittoDataset(data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 da=None):
        self.tokenizer = get_tokenizer(lm)
        self.pairs = []
        self.labels = []
        self.max_len = max_len
        self.size = size

        if isinstance(path, list):
            lines = path
        else:
            lines = open(path, encoding='utf-8')

        for line in lines:
            s1, s2, label = line.strip().split('\t')
            # 将每个实体拆分成属性
            attrs1 = self._split_attributes(s1)
            attrs2 = self._split_attributes(s2)
            self.pairs.append((attrs1, attrs2))
            self.labels.append(int(label))

        self.pairs = self.pairs[:size]
        self.labels = self.labels[:size]
        self.da = da
        if da is not None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None

    def _split_attributes(self, text):
        """将文本拆分成属性列表"""
        attrs = []
        parts = text.split('COL')
        for part in parts:
            if 'VAL' in part:
                attr_name, attr_value = part.split('VAL')
                attrs.append((attr_name.strip(), attr_value.strip()))
        return attrs

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            Tuple of (encoded_pairs, masks, label)
        """
        attrs1, attrs2 = self.pairs[idx]
        label = self.labels[idx]

        # 为每个属性对编码
        encoded_pairs = []
        masks = []
        
        # 确保两个实体的属性数量相同
        min_attrs = min(len(attrs1), len(attrs2))
        
        for i in range(min_attrs):
            name1, val1 = attrs1[i]
            name2, val2 = attrs2[i]
            
            # 编码属性对
            encoded = self.tokenizer.encode(
                text=f"{name1} {val1}",
                text_pair=f"{name2} {val2}",
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # 去掉 batch 维度
            encoded = encoded.squeeze(0)
            
            # 创建 attention mask
            mask = (encoded != self.tokenizer.pad_token_id).float()
            
            encoded_pairs.append(encoded)
            masks.append(mask)

        # 将所有属性对堆叠成一个张量
        encoded_pairs = torch.stack(encoded_pairs)  # [num_attrs, seq_len]
        masks = torch.stack(masks)  # [num_attrs, seq_len]

        return encoded_pairs, masks, label

    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch
        
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            Tuple of:
                encoded_pairs: [num_attrs, batch_size, seq_len]
                masks: [batch_size, seq_len, num_attrs]
                labels: [batch_size]
        """
        if len(batch) == 0:
            return None, None, None

        # 提取批次中的各个组件
        encoded_pairs = [item[0] for item in batch]  # list of [num_attrs, seq_len]
        masks = [item[1] for item in batch]  # list of [num_attrs, seq_len]
        labels = [item[2] for item in batch]  # list of scalars

        # 获取这个批次中的最大属性数
        max_attrs = max(x.size(0) for x in encoded_pairs)
        
        # 填充属性数较少的样本
        padded_pairs = []
        padded_masks = []
        
        for pairs, mask in zip(encoded_pairs, masks):
            num_attrs, seq_len = pairs.size()
            if num_attrs < max_attrs:
                # 创建填充属性
                attr_padding = torch.ones(
                    (max_attrs - num_attrs, seq_len),
                    dtype=pairs.dtype
                ) * 1  # 1 is the padding token id for most transformers
                
                # 创建填充掩码
                mask_padding = torch.zeros(
                    (max_attrs - num_attrs, seq_len),
                    dtype=mask.dtype
                )
                
                # 连接原始数据和填充
                pairs = torch.cat([pairs, attr_padding], dim=0)
                mask = torch.cat([mask, mask_padding], dim=0)
            
            padded_pairs.append(pairs)
            padded_masks.append(mask)

        # 堆叠所有样本
        encoded_pairs = torch.stack(padded_pairs)  # [batch_size, num_attrs, seq_len]
        masks = torch.stack(padded_masks)  # [batch_size, num_attrs, seq_len]
        
        # 转换维度顺序以匹配 HierGAT 的要求
        encoded_pairs = encoded_pairs.permute(1, 0, 2)  # [num_attrs, batch_size, seq_len]
        masks = masks.permute(0, 2, 1)  # [batch_size, seq_len, num_attrs]
        
        return encoded_pairs, masks, torch.LongTensor(labels)

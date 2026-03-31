import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


#Dataset for multimodal data (text + image)
class MultimodalDataset(Dataset):
    def __init__(self, df, images_dir, electra_tokenizer, clip_processor, max_len = 128, is_train =  False):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.electra_tokenizer = electra_tokenizer
        self.clip_processor = clip_processor
        self.max_len = max_len
        self.is_train = is_train

        if self.is_train:
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        else:
            self.augmentation =  None

    def __len__(self):
        return len(self.df)
    
    
    # Get item for multimodal dataset, return dictionary with pixel values, input ids, attention mask, and label
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        text =  str(row['Teks Terlihat']) if 'Teks Terlihat' in row else ''
        label = int(row['Label']) if 'Label' in row else int(row['Label Akhir'])
        fname = row['filename']
        img_path = os.path.join(self.images_dir, fname)
        image = Image.open(img_path).convert('RGB')

        image_for_clip = image
        if  self.is_train and self.augmentation:
            image_for_clip = self.augmentation(image)
        
        clip_inputs =  self.clip_processor(images = image_for_clip, return_tensors='pt')
        pixel_values = clip_inputs['pixel_values'].squeeze(0)

        enc = self.electra_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }
    
    # Collate batch for multimodal dataset, return dictionary with pixel values, input ids, attention mask, and labels
    @staticmethod
    def collate_batch(batch):
        pixel_values = torch.stack([b['pixel_values'] for b in batch], dim = 0)
        input_ids = torch.stack([b['input_ids'] for b in batch], dim = 0)
        attention_mask = torch.stack([b['attention_mask'] for b in batch], dim = 0)
        labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': labels
        }
    

import torch
import torch.nn as nn


class CLIPElectraFusion(nn.Module):

    def __init__(self, clip_model, electra_model,
                 fusion_img_dim=256,
                 fusion_text_dim=256,
                 num_classes=2, freeze_encoders=True):
        super().__init__()
        self.clip = clip_model
        self.electra = electra_model
        self.fusion_img_dim = fusion_img_dim
        self.fusion_text_dim = fusion_text_dim

        if freeze_encoders:
            for p in self.clip.parameters():
                p.requires_grad = False
            for p in self.electra.parameters():
                p.requires_grad = False

        # CLIP dimensi output image feature 
        self.img_dim = clip_model.config.projection_dim  # 512 

        # ELECTRA dimensi output text feature 
        electra_hidden_dim = electra_model.config.hidden_size  # e.g. 768

        # Projection layer only text: (electra_hidden_dim -> fusion_text_dim : 256)
        self.project_text = nn.Sequential(
            nn.Linear(electra_hidden_dim, fusion_text_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_text_dim)
        )

        # Projection layer image: (img_dim -> fusion_img_dim)
        if fusion_img_dim == self.img_dim:
            self.project_image = nn.Identity()
        else:
            self.project_image = nn.Sequential(
                nn.Linear(self.img_dim, fusion_img_dim),
                nn.GELU(),
                nn.LayerNorm(fusion_img_dim)
            )

        # Fusion dimension: fusion_img_dim + fusion_text_dim
        self.fusion_dim = fusion_img_dim + fusion_text_dim
        self.fusion_method = 'transformer'

        # Positional embedding untuk 2 tokens (image + text)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, self.fusion_dim))

        # 2-layer Transformer untuk fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim,
            nhead=8,
            dim_feedforward=self.fusion_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3-layer MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2), # 768 -> 384
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.fusion_dim // 2, self.fusion_dim // 4), # 384 -> 192
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.fusion_dim // 4, num_classes) # 192 -> num_classes
        )

    def forward(self, pixel_values, input_ids, attention_mask):

        # Extract image features from CLIP vision encoder
        if hasattr(self.clip, 'get_image_features'):
            img_output = self.clip.get_image_features(pixel_values)
        else:
            img_output = self.clip(pixel_values=pixel_values)

        if hasattr(img_output, 'image_embeds'):
            img_feats = img_output.image_embeds
        elif hasattr(img_output, 'pooler_output'):
            img_feats = img_output.pooler_output
        elif isinstance(img_output, torch.Tensor):
            img_feats = img_output
        else:
            img_feats = img_output[0] if isinstance(img_output, (tuple, list)) else img_output
        
        # Normalisasi  (L2 normalization)
        img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-10)

        # Project image setelah normalisasi.
        img_proj = self.project_image(img_feats)

        # Extract text features from ELECTRA
        txt_out = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = txt_out.last_hidden_state
        # Shape: (batch_size, sequence_length, 768)

        # Mean pooling: rata-rata semua token (kecuali padding)
        # Attention mask memastikan [PAD] token tidak ikut dihitung
        attn = attention_mask.unsqueeze(-1).float()  
        sum_emb = (last_hidden * attn).sum(dim=1)     
        sum_mask = attn.sum(dim=1).clamp(min=1e-9)    
        text_emb = sum_emb / sum_mask                
        # Shape: (batch_size, 768)

        # Normalisasi text embedding (L2 normalization)
        text_emb = text_emb / (text_emb.norm(dim=-1, keepdim=True) + 1e-10)

        # Project text dari 768 -> 256 dimensi       
        text_proj = self.project_text(text_emb)

        # Urutan Positional Embedding + Fusion (text first, then image)
        text_token = torch.cat([text_proj, torch.zeros_like(img_proj)], dim=-1)
        img_token = torch.cat([torch.zeros_like(text_proj), img_proj], dim=-1)

        # Stack tokens and add positional embedding
        tokens = torch.stack([text_token, img_token], dim=1)
        tokens = tokens + self.pos_embedding

        # Fusion with 2-layer Transformer
        fused_tokens = self.fusion_transformer(tokens)

        # Use first token (text token) for classification
        fused_rep = fused_tokens[:, 0, :]

        # 3-layer MLP classifier
        logits = self.classifier(fused_rep)

        return logits, img_proj, text_proj


class EarlyStopping:
    
    def __init__(self, patience=3, mode='max'):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.num_bad = 0
        self.should_stop = False
        
    def step(self, value):

        if self.best is None:
            self.best = value
            self.num_bad = 0
            return True
            
        improve = (value > self.best) if self.mode == 'max' else (value < self.best)
        
        if improve:
            self.best = value
            self.num_bad = 0
            return True
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
            return False

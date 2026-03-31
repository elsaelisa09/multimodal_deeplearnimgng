import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPElectraFusion(nn.Module):

    def __init__(self, clip_model, electra_model,
                 fusion_img_dim=256,
                 fusion_text_dim=256,
                 num_classes=2, freeze_encoders=True,
                 fusion_method='concatenate'):
        super().__init__()
        self.clip = clip_model
        self.electra = electra_model
        self.num_classes = num_classes
        self.fusion_img_dim = fusion_img_dim
        self.fusion_text_dim = fusion_text_dim
        self.fusion_method = str(fusion_method).lower().strip()
        self.focal_gamma = 2.0
        self.loss_strategy = 'none'
        self.class_weights = None

        valid_fusion = {
            'concatenate',
            'addition',
            'multiplication',
            'gated_fusion',
            'attention_fusion',
            'bilinear_fusion'
        }
        if self.fusion_method not in valid_fusion:
            raise ValueError(
                f"Unknown fusion_method '{self.fusion_method}'. Supported: {sorted(valid_fusion)}"
            )

        if freeze_encoders:
            for p in self.clip.parameters():
                p.requires_grad = False
            for p in self.electra.parameters():
                p.requires_grad = False

        # CLIP dimensi output image feature
        self.img_dim = clip_model.config.projection_dim  # misal 512

        # ELECTRA dimensi output text feature
        electra_hidden_dim = electra_model.config.hidden_size  # misal 768

        # Projection layer untuk image: (img_dim -> fusion_img_dim)
        # Jika dimensi sama, pakai Identity agar tidak ada transformasi tambahan.
        if fusion_img_dim == self.img_dim:
            self.project_image = nn.Identity()
        else:
            self.project_image = nn.Sequential(
                nn.Linear(self.img_dim, fusion_img_dim),
                nn.GELU(),
                nn.LayerNorm(fusion_img_dim)
            )

        # Projection layer untuk text: (electra_hidden_dim -> fusion_text_dim)
        # Jika dimensi sama, pakai Identity agar tidak ada transformasi tambahan.
        if fusion_text_dim == electra_hidden_dim:
            self.project_text = nn.Identity()
        else:
            self.project_text = nn.Sequential(
                nn.Linear(electra_hidden_dim, fusion_text_dim),
                nn.GELU(),
                nn.LayerNorm(fusion_text_dim)
            )

        self.same_modal_dim = (fusion_img_dim == fusion_text_dim)

        # Konfigurasi dimensi output fusion berdasarkan metode.
        if self.fusion_method == 'concatenate':
            self.fusion_dim = fusion_img_dim + fusion_text_dim
        elif self.fusion_method in {'addition', 'multiplication', 'gated_fusion', 'attention_fusion'}:
            if not self.same_modal_dim:
                raise ValueError(
                    f"fusion_method='{self.fusion_method}' requires fusion_img_dim == fusion_text_dim, "
                    f"got {fusion_img_dim} vs {fusion_text_dim}."
                )
            self.fusion_dim = fusion_img_dim
        elif self.fusion_method == 'bilinear_fusion':
            self.fusion_dim = fusion_img_dim + fusion_text_dim

        if self.fusion_method == 'gated_fusion':
            self.gate_layer = nn.Linear(self.fusion_dim * 2, self.fusion_dim)

        if self.fusion_method == 'attention_fusion':
            num_heads = 4 if (self.fusion_dim % 4 == 0) else 1
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=self.fusion_dim,
                num_heads=num_heads,
                batch_first=True
            )

        if self.fusion_method == 'bilinear_fusion':
            self.bilinear_fusion = nn.Bilinear(
                fusion_img_dim,
                fusion_text_dim,
                self.fusion_dim
            )

        # 3-layer MLP classifier (tanpa transformer, tanpa positional embedding)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),  #  (img_dim+fusion_text_dim) -> /2
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.fusion_dim // 2, self.fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.fusion_dim // 4, num_classes)
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

        # Normalisasi (L2 normalization)
        img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-10)

        # Proyeksi image ke dimensi yang lebih kecil
        img_proj = self.project_image(img_feats)

        # Extract text features from ELECTRA
        txt_out = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = txt_out.last_hidden_state
        # Shape: (batch_size, sequence_length, hidden_size)

        # Mean pooling: rata-rata semua token (dengan masking PAD)
        attn = attention_mask.unsqueeze(-1).float()
        sum_emb = (last_hidden * attn).sum(dim=1)
        sum_mask = attn.sum(dim=1).clamp(min=1e-9)
        text_emb = sum_emb / sum_mask
        # Shape: (batch_size, electra_hidden_dim)

        # Normalisasi text embedding (L2 normalization)
        text_emb = text_emb / (text_emb.norm(dim=-1, keepdim=True) + 1e-10)

        # Proyeksi text ke dimensi fusion (atau Identity jika dimensi sama)
        text_proj = self.project_text(text_emb)

        if self.fusion_method == 'concatenate':
            fused_rep = torch.cat([img_proj, text_proj], dim=-1)
        elif self.fusion_method == 'addition':
            fused_rep = img_proj + text_proj
        elif self.fusion_method == 'multiplication':
            fused_rep = img_proj * text_proj
        elif self.fusion_method == 'gated_fusion':
            gate = torch.sigmoid(self.gate_layer(torch.cat([img_proj, text_proj], dim=-1)))
            fused_rep = gate * img_proj + (1.0 - gate) * text_proj
        elif self.fusion_method == 'attention_fusion':
            # Dua token modalitas: [image, text] lalu diringkas dengan mean pooling.
            tokens = torch.stack([img_proj, text_proj], dim=1)
            attn_out, _ = self.fusion_attention(tokens, tokens, tokens)
            fused_rep = attn_out.mean(dim=1)
        elif self.fusion_method == 'bilinear_fusion':
            fused_rep = self.bilinear_fusion(img_proj, text_proj)
        else:
            raise RuntimeError(f"Unhandled fusion method: {self.fusion_method}")

        # 3-layer MLP classifier
        logits = self.classifier(fused_rep)

        # Kembalikan juga fitur img_proj dan text_proj untuk keperluan analisis
        return logits, img_proj, text_proj

    def configure_loss_strategy(self, strategy='none', class_weights=None, focal_gamma=None):
        """Configure loss behavior used by compute_loss during train/eval.

        Supported values:
        - none: standard cross-entropy
        - class_weight: weighted cross-entropy
        - focal: focal loss
        - class_weight_focal: weighted focal loss
        """
        strategy = str(strategy).lower().strip()
        valid = {'none', 'class_weight', 'focal', 'class_weight_focal'}
        if strategy not in valid:
            raise ValueError(f"Unknown strategy '{strategy}'. Supported: {sorted(valid)}")

        self.loss_strategy = strategy
        if focal_gamma is not None:
            self.focal_gamma = float(focal_gamma)

        if class_weights is not None:
            self.class_weights = torch.as_tensor(class_weights, dtype=torch.float)
        elif 'class_weight' in strategy:
            raise ValueError("class_weights is required for class_weight-based strategies")
        else:
            self.class_weights = None

    def compute_loss(self, logits, labels):
        """Compute loss according to configured imbalance strategy."""
        strategy = self.loss_strategy
        gamma = self.focal_gamma

        if strategy == 'none':
            return F.cross_entropy(logits, labels)

        if strategy == 'class_weight':
            weight = self.class_weights.to(logits.device)
            return F.cross_entropy(logits, labels, weight=weight)

        if strategy == 'focal':
            ce = F.cross_entropy(logits, labels, reduction='none')
            pt = torch.exp(-ce)
            return (((1.0 - pt) ** gamma) * ce).mean()

        if strategy == 'class_weight_focal':
            weight = self.class_weights.to(logits.device)
            ce = F.cross_entropy(logits, labels, weight=weight, reduction='none')
            pt = torch.exp(-ce)
            return (((1.0 - pt) ** gamma) * ce).mean()

        raise RuntimeError(f"Unhandled strategy: {strategy}")

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

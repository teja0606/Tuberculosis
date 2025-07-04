# src/model.py
#
# Defines the architecture for our custom hybrid CNN-Transformer model.
# This model is built from scratch and designed to learn features from
# chest X-ray images for Tuberculosis detection.

import torch
from torch import nn

class HybridTBNet(nn.Module):
    """
    A custom Hybrid CNN-Transformer model for Tuberculosis detection.
    
    The architecture consists of three main parts:
    1. A custom CNN backbone to extract spatial features from the image.
    2. A Transformer Encoder layer to model global relationships between features.
    3. A final MLP classification head to make the binary prediction.
    """
    def __init__(self,
                 input_channels: int = 1,
                 cnn_output_channels: int = 128,
                 transformer_heads: int = 8,
                 transformer_dropout: float = 0.1,
                 mlp_dropout: float = 0.5):
        super().__init__()

        # --- 1. Custom CNN Backbone ---
        # This part of the network processes the raw image and extracts feature maps.
        # Each block consists of Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d.
        # BatchNorm2d is crucial for stable training from scratch.
        self.cnn_backbone = nn.Sequential(
            # Block 1: Input (1, 224, 224) -> Output (32, 112, 112)
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: Input (32, 112, 112) -> Output (64, 56, 56)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: Input (64, 56, 56) -> Output (128, 28, 28)
            nn.Conv2d(in_channels=64, out_channels=cnn_output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_output_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # After the CNN, the feature map size will be (batch_size, cnn_output_channels, 28, 28)
        # For the transformer, we need to flatten this into a sequence.
        # The sequence length will be 28*28 = 784.
        # The embedding dimension for each item in the sequence is cnn_output_channels (128).
        
        # --- 2. Transformer Encoder Layer ---
        # This layer takes the sequence of features from the CNN and learns their
        # global relationships using self-attention.
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=cnn_output_channels, # The feature dimension from the CNN
            nhead=transformer_heads,     # Number of attention heads
            dropout=transformer_dropout,
            batch_first=True             # Expects input as (batch, sequence, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=2) # Stack 2 layers

        # --- 3. MLP Classification Head ---
        # This final part takes the processed features from the transformer and
        # makes a binary classification.
        self.classifier = nn.Sequential(
            nn.LayerNorm(cnn_output_channels), # Normalize features before classification
            nn.Linear(in_features=cnn_output_channels, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=mlp_dropout), # Heavy dropout for regularization
            nn.Linear(in_features=64, out_features=1) # Output a single logit for binary classification
        )

        # Apply Kaiming He weight initialization to Conv layers
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initializes weights of the model.
        Kaiming initialization is used for Conv2d layers.
        """
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of raw logits (batch_size, 1).
        """
        # 1. Pass through CNN backbone
        x = self.cnn_backbone(x)
        # Current shape: (batch_size, cnn_output_channels, 28, 28)

        # 2. Prepare for Transformer
        # Flatten the height and width dimensions into a single sequence dimension
        batch_size, channels, height, width = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        # New shape: (batch_size, height*width, channels) -> (batch_size, 784, 128)

        # 3. Pass through Transformer Encoder
        x = self.transformer_encoder(x)
        # Shape remains: (batch_size, 784, 128)

        # 4. Aggregate Transformer output
        # We use the mean of the sequence as the input to the classifier
        x = x.mean(dim=1)
        # New shape: (batch_size, 128)

        # 5. Pass through Classifier Head
        logits = self.classifier(x)
        # Final shape: (batch_size, 1)

        return logits

if __name__ == '__main__':
    # This block allows you to test the model architecture directly
    # To run: python -m src.model
    
    # Create a dummy input tensor to simulate a batch of images
    dummy_input = torch.randn(32, 1, 224, 224) # (batch_size, channels, height, width)
    
    # Instantiate the model
    model = HybridTBNet()
    
    # Pass the dummy input through the model
    output_logits = model(dummy_input)
    
    # Print the shapes to verify the forward pass works correctly
    print(f"--- Model Architecture Test ---")
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output logits shape: {output_logits.shape}")
    print(f"Model architecture seems correct!")


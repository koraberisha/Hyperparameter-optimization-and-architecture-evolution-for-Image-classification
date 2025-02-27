import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Basic convolutional block for feature extraction"""
    def __init__(self, in_channels, out_channels, kernel_size, activation_type, num_convs=1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # First convolutional layer (potentially with stride)
        self.layers.append(nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding='same', 
            stride=1
        ))
        
        # Additional convolutional layers if specified
        for _ in range(num_convs - 1):
            self.layers.append(nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                padding='same',
                stride=1
            ))
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Activation function
        self.activation = self._get_activation(activation_type)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
    
    def _get_activation(self, activation_type):
        """Get activation function based on encoded type"""
        activations = [
            nn.ReLU(),      # 0
            nn.ELU(),       # 1
            nn.LeakyReLU(), # 2
            nn.GELU()       # 3
        ]
        return activations[activation_type % len(activations)]
    
    def forward(self, x):
        # Apply each convolutional layer
        for conv_layer in self.layers:
            x = conv_layer(x)
        
        # Apply batch norm, activation, pooling, and dropout
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        return x

class CNNFromChromosome(nn.Module):
    """Generate CNN architecture from binary chromosome encoding"""
    def __init__(self, chromosome, num_classes=10):
        super().__init__()
        
        # Decode chromosome
        out_ch_idx, kernel_idx, act_idx, pool_idx, layer_choices = self._decode_chromosome(chromosome)
        
        # Map indices to actual values
        output_options = [16, 32, 64, 128]
        conv_options = [3, 5, 7, 9]  # Kernel sizes
        activation_options = [0, 1, 2, 3]  # Indices for activation functions
        layer_levels = [2, 3, 4, 5]  # Number of feature extraction layers
        
        # Get values from options
        base_filters = output_options[out_ch_idx]
        kernel_size = conv_options[kernel_idx]
        activation = activation_options[act_idx]
        num_layers = layer_levels[out_ch_idx]  # Using out_ch_idx for layer count too
        
        # Build feature extraction layers
        self.features = self._build_features(
            base_filters=base_filters,
            kernel_size=kernel_size,
            activation=activation,
            num_layers=num_layers,
            layer_choices=layer_choices
        )
        
        # Calculate feature dimensions for classifier
        # Assuming input is 32x32x3 (CIFAR)
        # Each conv block halves the spatial dimensions through pooling
        feature_dim = base_filters * (2 ** (num_layers-1)) * (32 // (2 ** num_layers)) ** 2
        
        # Build classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def _decode_chromosome(self, chromosome):
        """Decode binary chromosome string into architecture parameters"""
        out_ch_idx = int(chromosome[0:2], 2)
        kernel_idx = int(chromosome[3:5], 2)
        act_idx = int(chromosome[6:8], 2)
        pool_idx = int(chromosome[9:11], 2)
        layer_choices = chromosome[12:19]
        
        return out_ch_idx, kernel_idx, act_idx, pool_idx, layer_choices
    
    def _build_features(self, base_filters, kernel_size, activation, num_layers, layer_choices):
        """Build feature extraction layers based on chromosome encoding"""
        layers = nn.ModuleList()
        
        in_channels = 3  # Initial input channels (RGB)
        
        for i in range(num_layers):
            # Double filters at each layer
            out_channels = base_filters * (2 ** i)
            
            # Determine number of conv layers in this block
            # If we have a 1 at this position, use 1 conv layer, otherwise use 2
            if i < len(layer_choices) and layer_choices[i] == '1':
                num_convs = 1
            else:
                num_convs = 2
            
            # Create and add conv block
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                activation_type=activation,
                num_convs=num_convs
            )
            
            layers.append(block)
            
            # Update input channels for next layer
            in_channels = out_channels
        
        # Return sequential container of all feature layers
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model_from_chromosome(chromosome, num_classes=10, use_compile=False):
    """Factory function to create model from chromosome"""
    model = CNNFromChromosome(chromosome, num_classes)
    
    # Only attempt compilation if explicitly enabled
    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("Using torch.compile for model optimization")
        except Exception as e:
            print(f"Warning: Unable to use torch.compile: {e}")
    
    return model
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.cpp_extension import load


# Load custom matmul kernel
current_dir = os.path.dirname(os.path.abspath(__file__))
custom_matmul = load(
    name="custom_matmul",
    sources=[
        os.path.join(current_dir, "matMul.cu"),
    ],
    verbose=True,
    with_cuda=True
)

# Register as a custom operator for torch.compile compatibility
@torch.library.custom_op("custom::matmul", mutates_args=())
def custom_matmul_op(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return custom_matmul.matmul_forward(A, B)

@custom_matmul_op.register_fake
def custom_matmul_fake(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.empty(A.shape[0], B.shape[1], dtype=A.dtype, device=A.device)


class CustomLinear(nn.Module):
    """Custom linear layer using the custom matmul kernel."""
    
    def __init__(self, in_features, out_features):
        super().__init__() 
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        # Use our custom matrix multiplication
        output = custom_matmul_op(x, self.weight.t())
        return output + self.bias


class Cat2Vec(nn.Module):
    """
    Cat to Vector neural network with 5 consecutive fully-connected layers.
    Maps cat images to fixed-size vector embeddings.
    """
    
    def __init__(self, input_size=3*224*224, hidden_sizes=[1024, 512, 256, 128], output_size=64, use_custom_matmul=True):
        """
        Args:
            input_size (int): Size of flattened input (e.g., 3*224*224 for RGB images)
            hidden_sizes (list): Sizes of the 4 hidden layers
            output_size (int): Size of final output vector embedding
            use_custom_matmul (bool): Whether to use custom CUDA matmul kernel
        """
        super(Cat2Vec, self).__init__()
        
        self.use_custom_matmul = use_custom_matmul
        linear_layer = CustomLinear if use_custom_matmul else nn.Linear
        
        # Build 5 consecutive fully-connected layers
        layers = []
        
        # Input layer
        layers.append(linear_layer(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        
        # Hidden layers (3 more layers)
        for i in range(len(hidden_sizes) - 1):
            layers.append(linear_layer(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
        
        # Output layer
        layers.append(linear_layer(hidden_sizes[-1], output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, channels, height, width)
                             or (batch_size, flattened_size)
        
        Returns:
            torch.Tensor: Output vector embeddings, shape (batch_size, output_size)
        """
        # Flatten input if it's an image tensor
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Pass through the network
        return self.network(x)
    
    def get_embedding(self, x):
        """
        Get normalized embedding vector.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: L2-normalized embedding vector
        """
        embedding = self.forward(x)
        return F.normalize(embedding, p=2, dim=1)


def create_cat2vec_model(input_size=3*224*224, output_size=64, device='cuda', use_custom_matmul=True):
    """
    Factory function to create and initialize a Cat2Vec model.
    
    Args:
        input_size (int): Size of flattened input
        output_size (int): Size of output embedding
        device (str): Device to place the model on
        use_custom_matmul (bool): Whether to use custom CUDA matmul kernel
        
    Returns:
        Cat2Vec: Initialized model
    """
    model = Cat2Vec(input_size=input_size, output_size=output_size, use_custom_matmul=use_custom_matmul)
    model = model.to(device)
    return model


# Example usage and testing
if __name__ == "__main__":
    print("Testing Cat2Vec with custom matmul kernel...")
    
    # Create model with custom matmul
    model_custom = create_cat2vec_model(use_custom_matmul=True)
    print(f"Custom model created with {sum(p.numel() for p in model_custom.parameters())} parameters")
    
    # Create model with standard PyTorch matmul for comparison
    model_standard = create_cat2vec_model(use_custom_matmul=False)
    print(f"Standard model created with {sum(p.numel() for p in model_standard.parameters())} parameters")
    
    # Test with dummy input
    batch_size = 8
    dummy_input = torch.randn(batch_size, 3, 224, 224, device='cuda')
    
    # Forward pass with custom matmul
    with torch.no_grad():
        output_custom = model_custom(dummy_input)
        normalized_output_custom = model_custom.get_embedding(dummy_input)
    
    # Forward pass with standard matmul
    with torch.no_grad():
        output_standard = model_standard(dummy_input)
        normalized_output_standard = model_standard.get_embedding(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Custom output shape: {output_custom.shape}")
    print(f"Standard output shape: {output_standard.shape}")
    print(f"Custom output range: [{output_custom.min().item():.3f}, {output_custom.max().item():.3f}]")
    print(f"Standard output range: [{output_standard.min().item():.3f}, {output_standard.max().item():.3f}]")
    print(f"Custom normalized norms: {torch.norm(normalized_output_custom, dim=1)}")
    print(f"Standard normalized norms: {torch.norm(normalized_output_standard, dim=1)}")
    
    # Test torch.compile compatibility with custom matmul
    print("\nTesting torch.compile compatibility...")
    try:
        compiled_model = torch.compile(model_custom)
        with torch.no_grad():
            output_compiled = compiled_model(dummy_input)
        print("torch.compile works with custom matmul!")
        print(f"Compiled output shape: {output_compiled.shape}")
    except Exception as e:
        print(f"torch.compile failed: {e}")
def veri_model_builder(arch, base, weights=None, normalization=None, embedding_dimensions=None, soft_dimensions=None, **kwargs):
    """Vehicle Re-id model builder.

    This builds a model for vehicle re-id. Refer to paper [] for general construction. The model contains:
        * Architecture core
        * Convolutional attention
        * Spatial average pooling
        * FC-Layer for embedding dimensionality change
        * Normalization layer
        * Softmax FC Layers

    Args:
        arch (str): Ther architecture to use. The string "Base" is added after architecture (e.g. "Resnet", "Inception"). Only "Resnet" is currently supported.
        base (str): The architecture subtype, e.g. "resnet50", "resnet18"
        weights (str): Local path to weights file for the architecture core, e.g. pretrained resnet50 weights path.
        normalization (str): Normalization layer for reid-model. Can be None. Supported normalizations: ["bn", "l2", "in", "gn", "ln"]
        embedding_dimension (int): Dimensions for the feature embedding. Leave empty if feature dimensions should be same as architecture core output (e.g. resnet50 base model has 2048-dim feature outputs). If providing a value, it should be less than the architecture core's base feature dimensions
        soft_dimensions (int): Feature dimensions for softmax FC layer. Can be None if not using it. Should be equal to the number of identities in the dataset.
        kwargs (dict): Nothing yet


    Returns:
        Torch Model: A Torch Re-ID model

    """
    arch = arch+"Base"
    archbase = __import__("models."+arch, fromlist=[arch])
    archbase = getattr(archbase, arch)

    model = archbase(base = base, weights=weights, normalization = normalization, embedding_dimensions = embedding_dimensions, soft_dimensions = soft_dimensions, **kwargs)
    return model

def vaegan_model_builder(arch, base, latent_dimensions = None, **kwargs):
    """VAE-GAN Model builder.

    This builds the VAAE-GAN model used in paper []

    Args:
        arch (str): The architecture to use. Only "VAEGAN" is supported.
        base (int): Dimensionality of the input images. MUST be a power of 2.
        latent_dimensions (int): Embedding dimension for the encoder of the VAAE-GAN.
        
        kwargs (dict): Needs the following:
            channels (int): Number of channels in image. Default 3. 1 for MNIST.

    Returns:
        Torch Model: A Torch-based VAAE-GAN model
    """
    if arch != "VAEGAN":
        raise NotImplementedError()
    archbase = __import__("models."+arch, fromlist=[arch])
    archbase = getattr(archbase, arch)
    
    return archbase(base, latent_dimensions=latent_dimensions, **kwargs)
    

def carzam_model_builder(arch, base, weights=None, normalization=None, embedding_dimensions=None, **kwargs):
    """CarZam model builder.

    This builds a simple single model (NOT teamed classifier) for CarZam. Refer to paper [] for general construction. The model contains:
        * Architecture core
        * Convolutional attention
        * Spatial average pooling
        * FC-Layer for embedding dimensionality change
        * Normalization layer
        * Softmax FC Layers

    Args:
        arch (str): Ther architecture to use. The string "Base" is added after architecture (e.g. "Resnet", "Inception"). Only "Resnet" is currently supported.
        base (str): The architecture subtype, e.g. "resnet50", "resnet18"
        weights (str): Local path to weights file for the architecture core, e.g. pretrained resnet50 weights path.
        normalization (str): Normalization layer for reid-model. Can be None. Supported normalizations: ["bn", "l2", "in", "gn", "ln"]
        embedding_dimension (int): Dimensions for the feature embedding. Leave empty if feature dimensions should be same as architecture core output (e.g. resnet50 base model has 2048-dim feature outputs). If providing a value, it should be less than the architecture core's base feature dimensions
        soft_dimensions (int): Feature dimensions for softmax FC layer. Can be None if not using it. Should be equal to the number of identities in the dataset.
        kwargs (dict): Nothing yet


    Returns:
        Torch Model: A Torch Re-ID model

    """
    if arch != "CarzamResnet":
        raise NotImplementedError()
    archbase = __import__("models."+arch, fromlist=[arch])
    archbase = getattr(archbase, arch)

    model = archbase(base = base, weights=weights, normalization = normalization, embedding_dimensions = embedding_dimensions, **kwargs)
    return model
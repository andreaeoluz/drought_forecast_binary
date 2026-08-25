"""reproducibility.py - Configuração de reprodutibilidade"""

import random
import numpy as np
import torch
import os


def set_reproducible_seeds(seed: int = 42, deterministic: bool = True):
    """
    Configura todas as sementes para reprodutibilidade.
    
    Args:
        seed: Semente base
        deterministic: Se True, força operações determinísticas
    """
    # Python
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch determinístico
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Importante: algumas operações podem ficar mais lentas
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Python hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    #print(f"✅ Reprodutibilidade configurada (seed={seed}, deterministic={deterministic})")


def get_deterministic_loader(loader, seed: int = 42):
    """
    Garante que um DataLoader seja determinístico.
    
    Args:
        loader: DataLoader existente
        seed: Semente para o worker
    
    Returns:
        DataLoader com worker_init_fn configurado
    """
    def worker_init_fn(worker_id):
        """Inicializa cada worker com semente base + worker_id."""
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    # Recriar DataLoader com worker_init_fn
    return torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=loader.shuffle if hasattr(loader, 'shuffle') else False,
        sampler=loader.sampler if hasattr(loader, 'sampler') else None,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last if hasattr(loader, 'drop_last') else False,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),  # Para shuffle
    )
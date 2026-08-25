"""spatial.py - Funções de processamento espacial"""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path 


def downsample_block_mean(data: np.ndarray, factor_h: int, factor_w: int) -> np.ndarray:
    """
    Reduz resolução espacial via média de blocos.
    
    Args:
        data: (H, W, C) ou (H, W)
        factor_h: fator de downsampling em altura
        factor_w: fator de downsampling em largura
    
    Returns:
        Data downsampled
    """
    is_2d = data.ndim == 2
    if is_2d:
        data = data[..., np.newaxis]
    
    H, W, C = data.shape
    Hc = (H // factor_h) * factor_h
    Wc = (W // factor_w) * factor_w
    data = data[:Hc, :Wc, :]
    
    data = data.reshape(Hc // factor_h, factor_h, Wc // factor_w, factor_w, C)
    result = data.mean(axis=(1, 3))
    
    return result.squeeze(-1) if is_2d else result


def build_valid_mask(
    data_stack: np.ndarray, 
    min_valid_ratio: float = 0.5,
    verbose: bool = True
) -> np.ndarray:
    """
    Constrói máscara de pixels válidos baseada em dados faltantes.
    
    ✅ CORREÇÃO: A máscara indica apenas se o pixel TEM DADOS (não NaN/Inf),
    independentemente do valor (0, 1, ou qualquer outro).
    
    Args:
        data_stack: (T, H, W, C)
        min_valid_ratio: fração mínima de timesteps com dados finitos
        verbose: Se True, imprime estatísticas
    
    Returns:
        mask: (H, W) - True para pixels com dados válidos
    """
    T, H, W, C = data_stack.shape
    
    # ✅ Contar quantos timesteps têm dados válidos (não NaN/Inf)
    valid_count = np.zeros((H, W), dtype=np.float32)
    
    for t in range(T):
        for c in range(C):
            band = data_stack[t, :, :, c]
            # ✅ VERIFICAR APENAS NaN/Inf, NÃO valor 0
            valid = ~(np.isnan(band) | np.isinf(band))
            valid_count += valid.astype(np.float32)
    
    # Proporção de dados válidos por pixel
    total_observations = T * C
    valid_ratio = valid_count / total_observations
    
    # ✅ Máscara: pixels com pelo menos min_valid_ratio de dados válidos
    mask = valid_ratio >= min_valid_ratio
    
    # ✅ GARANTIR que a máscara não exclui pixels com valor 0
    # (valores 0 são dados válidos, representam não-seca)
    
    if verbose:
        total_pixels = mask.size
        valid_pixels = mask.sum()
        invalid_pixels = total_pixels - valid_pixels
        
        print(f"\n📊 Máscara de validade:")
        print(f"   Total pixels: {total_pixels:,}")
        print(f"   Pixels válidos (com dados): {valid_pixels:,} ({valid_pixels/total_pixels:.1%})")
        print(f"   Pixels inválidos (dados faltantes): {invalid_pixels:,} ({invalid_pixels/total_pixels:.1%})")
        print(f"   Critério: min_valid_ratio = {min_valid_ratio:.2f}")
        
        # Verificar se há pixels com dados mas valor zero (não-seca)
        # Isso é apenas informativo
        if invalid_pixels == 0:
            print(f"   ✅ Todos os pixels têm dados completos")
        
        # Verificar se a máscara está excluindo pixels com valor zero
        # Isso NÃO deve acontecer!
        zero_mask = (data_stack == 0).all(axis=(0, 3))  # Pixels que são sempre zero
        zero_valid = zero_mask & mask
        
        if zero_valid.sum() > 0:
            print(f"\n   ℹ️ Pixels com valor zero (não-seca) considerados válidos: {zero_valid.sum():,}")
    
    return mask


def apply_domain_mask(
    data: np.ndarray, 
    mask: np.ndarray, 
    fill_value: float = np.nan
) -> np.ndarray:
    """
    Aplica máscara de domínio aos dados.
    
    ⚠️ ATENÇÃO: Esta função NÃO deve ser usada nos dados de entrada para treinamento!
    A máscara deve ser aplicada APENAS no cálculo da perda.
    
    Esta função é útil para:
    - Visualização (mostrar apenas pixels válidos)
    - Pós-processamento
    - Geração de rasters (usar nodata apropriado)
    
    Args:
        data: (T, H, W, C) ou (H, W, C)
        mask: (H, W) - True para pixels válidos
        fill_value: Valor para preencher pixels inválidos (default: np.nan)
    
    Returns:
        Dados com pixels inválidos preenchidos com fill_value
    """
    if mask is None:
        return data
    
    masked = data.copy()
    
    # Expandir máscara para mesma dimensionalidade dos dados
    if data.ndim == 4:
        T, H, W, C = data.shape
        mask_expanded = mask[None, :, :, None]  # (1, H, W, 1)
        mask_expanded = np.broadcast_to(mask_expanded, (T, H, W, C))
        masked[~mask_expanded] = fill_value
    elif data.ndim == 3:
        H, W, C = data.shape
        mask_expanded = mask[:, :, None]  # (H, W, 1)
        mask_expanded = np.broadcast_to(mask_expanded, (H, W, C))
        masked[~mask_expanded] = fill_value
    elif data.ndim == 2:
        H, W = data.shape
        masked[~mask] = fill_value
    else:
        raise ValueError(f"Dimensões não suportadas: {data.ndim}")
    
    return masked


def get_valid_pixel_indices(
    mask: np.ndarray, 
    return_flat: bool = True
) -> np.ndarray:
    """
    Retorna índices dos pixels válidos na máscara.
    
    Args:
        mask: Máscara booleana (H, W)
        return_flat: Se True, retorna índices planos; senão, índices 2D
    
    Returns:
        Índices dos pixels válidos
    """
    if return_flat:
        return np.where(mask.flatten())[0]
    else:
        return np.where(mask)


def create_train_val_mask(
    mask: np.ndarray,
    validation_split: float = 0.2,
    seed: int = 42,
    spatial_blocks: bool = False,
    block_size: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria máscaras de treino e validação a partir de uma máscara de domínio.
    
    Args:
        mask: Máscara de domínio (H, W)
        validation_split: Proporção de pixels para validação
        seed: Semente para reprodutibilidade
        spatial_blocks: Se True, divide por blocos espaciais
        block_size: Tamanho do bloco para divisão espacial
    
    Returns:
        train_mask, val_mask
    """
    np.random.seed(seed)
    
    if spatial_blocks:
        # Dividir por blocos espaciais
        H, W = mask.shape
        train_mask = np.zeros_like(mask, dtype=bool)
        val_mask = np.zeros_like(mask, dtype=bool)
        
        # Criar grid de blocos
        blocks_h = H // block_size
        blocks_w = W // block_size
        
        # Embaralhar blocos
        block_indices = np.arange(blocks_h * blocks_w)
        np.random.shuffle(block_indices)
        
        n_val_blocks = int(blocks_h * blocks_w * validation_split)
        val_blocks = set(block_indices[:n_val_blocks])
        
        for bi in range(blocks_h):
            for bj in range(blocks_w):
                h_start = bi * block_size
                h_end = min(h_start + block_size, H)
                w_start = bj * block_size
                w_end = min(w_start + block_size, W)
                
                block_idx = bi * blocks_w + bj
                if block_idx in val_blocks:
                    val_mask[h_start:h_end, w_start:w_end] = True
                else:
                    train_mask[h_start:h_end, w_start:w_end] = True
        
        # Aplicar máscara de domínio
        train_mask = train_mask & mask
        val_mask = val_mask & mask
        
    else:
        # Divisão aleatória por pixels
        valid_indices = get_valid_pixel_indices(mask, return_flat=True)
        
        n_valid = len(valid_indices)
        n_val = int(n_valid * validation_split)
        
        shuffled = np.random.permutation(valid_indices)
        val_indices = shuffled[:n_val]
        train_indices = shuffled[n_val:]
        
        train_mask = np.zeros_like(mask, dtype=bool)
        val_mask = np.zeros_like(mask, dtype=bool)
        
        train_mask.flat[train_indices] = True
        val_mask.flat[val_indices] = True
    
    return train_mask, val_mask


def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove componentes conectados menores que min_size."""
    from scipy import ndimage
    
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask
    
    sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    result = mask.copy()
    
    for i, size in enumerate(sizes):
        if size < min_size:
            result[labeled == i + 1] = 0
    
    return result


def remove_small_holes(mask: np.ndarray, area_threshold: int) -> np.ndarray:
    """Preenche buracos pequenos dentro das regiões."""
    from scipy import ndimage
    
    inv_mask = 1 - mask
    labeled, num = ndimage.label(inv_mask)
    
    if num == 0:
        return mask
    
    sizes = ndimage.sum(inv_mask, labeled, range(1, num + 1))
    result = mask.copy()
    
    for i, size in enumerate(sizes):
        if size < area_threshold:
            result[labeled == i + 1] = 1
    
    return result


def postprocess_binary_mask(
    probs: np.ndarray,
    threshold: float,
    min_area: int = 5,
    hole_area: int = 5,
) -> np.ndarray:
    """
    Pós-processamento de máscara binária.
    
    Args:
        probs: Probabilidades (H, W)
        threshold: Limiar de binarização
        min_area: Área mínima para manter componente
        hole_area: Área máxima de buraco para preencher
    
    Returns:
        Máscara binária pós-processada
    """
    binary = (probs > threshold).astype(np.uint8)
    
    if binary.sum() > 0:
        binary = remove_small_objects(binary, min_area)
        binary = remove_small_holes(binary, hole_area)
    
    return binary


def create_validation_rasters(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    output_dir: Path,
    metadata: dict,
    prefix: str = "val"
) -> None:
    """
    Cria rasters de validação para análise visual.
    
    Args:
        predictions: Predições (H, W)
        targets: Targets (H, W)
        mask: Máscara de validade (H, W)
        threshold: Limiar de binarização
        output_dir: Diretório de saída
        metadata: Metadados do raster
        prefix: Prefixo para arquivos
    """
    import rasterio
    from rasterio.transform import Affine
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aplicar máscara
    predictions_masked = predictions.copy()
    targets_masked = targets.copy()
    
    predictions_masked[~mask] = 255
    targets_masked[~mask] = 255
    
    # Binarizar predições
    binary_pred = (predictions_masked > threshold).astype(np.uint8) * 1
    binary_pred[~mask] = 255
    
    # Salvar rasters
    profile = {
        'driver': 'GTiff',
        'height': mask.shape[0],
        'width': mask.shape[1],
        'count': 1,
        'dtype': 'uint8',
        'crs': metadata.get('crs'),
        'transform': metadata.get('transform'),
        'nodata': 255,  # ✅ CORRETO: nodata=255 para binários
        'compress': 'lzw',
    }
    
    for name, data in [
        (f'{prefix}_pred', binary_pred),
        (f'{prefix}_truth', targets_masked.astype(np.uint8)),
        (f'{prefix}_prob', predictions_masked.astype(np.float32)),
    ]:
        if name.endswith('_prob'):
            profile['dtype'] = 'float32'
            profile['nodata'] = np.nan
        
        out_path = output_dir / f"{name}.tif"
        with rasterio.open(out_path, 'w', **profile) as dst:
            if name.endswith('_prob'):
                dst.write(data, 1)
            else:
                dst.write(data.astype(np.uint8), 1)
        
        print(f"   Saved: {out_path}")
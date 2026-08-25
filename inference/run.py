#!/usr/bin/env python3
"""run_inference.py - Script principal de inferência"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np

# Adicionar diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ExperimentConfig
from config.paths import get_paths, get_data_path
from inference.predictor import InferencePredictor
from utils import set_reproducible_seeds
from utils.logger import Logger


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def convert_to_serializable(obj):
    """Converte objetos numpy para tipos Python serializáveis."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

def find_best_config(
    region: str,
    threshold: float,
    optimization_metric: str = "csi",
    transfer_learning: bool = None
) -> dict:
    """Encontra a melhor configuração do grid search."""
    paths = get_paths(region, threshold=threshold)
    
    suffix = f"thr_{abs(threshold):.1f}"
    if transfer_learning is not None:
        suffix += f"_tl_{transfer_learning}"
    
    config_files = [
        paths["grid_search_results"] / suffix / f"best_configuration_{suffix}.json",
        paths["grid_search_results"] / f"best_configuration_{suffix}.json",
        paths["grid_search_results"] / "best_configuration.json",
        paths["grid_search_results"] / "best_model_by_csi.json",
        paths["grid_search_dir"] / "best_configuration.json",
    ]
    
    results_path = None
    for f in config_files:
        if f.exists():
            results_path = f
            break
    
    if results_path is None:
        raise FileNotFoundError(
            f"Configuração não encontrada para região '{region}' "
            f"com threshold {threshold}"
        )
    
    with open(results_path, "r") as f:
        data = json.load(f)
    
    if "best_configuration" in data:
        best = data["best_configuration"]
    else:
        best = data
    
    if "p" not in best or "q" not in best:
        raise ValueError("Configuração não contém p e q")
    
    return best


def list_available_models(config: ExperimentConfig):
    """Lista modelos disponíveis para inferência."""
    paths = get_paths(config.region, threshold=config.spi.threshold)
    logger = Logger()
    
    logger.header(f"MODELOS DISPONÍVEIS - {config.region}")
    
    search_dirs = []
    
    if "grid_search_pretrained" in paths:
        search_dirs.append(("pretrained", paths["grid_search_pretrained"]))
    if "grid_search_scratch" in paths:
        search_dirs.append(("scratch", paths["grid_search_scratch"]))
    
    gs_dir = paths.get("grid_search_dir")
    if gs_dir and gs_dir.exists():
        for model_type in ["pretrained", "scratch"]:
            d = gs_dir / model_type
            if d.exists():
                search_dirs.append((model_type, d))
    
    models = []
    for model_type, search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        for model_file in search_dir.glob("model_*.pth"):
            name = model_file.stem
            parts = name.split("_")
            
            p = None
            q = None
            
            for part in parts:
                if part.startswith("p") and part[1:].isdigit():
                    p = int(part[1:])
                elif part.startswith("q") and part[1:].isdigit():
                    q = int(part[1:])
            
            if p is not None and q is not None:
                models.append({
                    "p": p,
                    "q": q,
                    "type": model_type,
                    "path": model_file,
                    "dirname": search_dir.relative_to(paths["base"]) if "base" in paths else search_dir.name
                })
    
    seen = set()
    unique_models = []
    for m in sorted(models, key=lambda x: (x["p"], x["q"], x["type"])):
        key = (m["p"], m["q"], m["type"])
        if key not in seen:
            seen.add(key)
            unique_models.append(m)
    
    if not unique_models:
        logger.warning("Nenhum modelo encontrado!")
        return
    
    print(f"\n{'p':<6} {'q':<6} {'Tipo':<12} {'Diretório':<40}")
    print(f"{'-'*70}")
    for m in unique_models:
        print(f"{m['p']:<6} {m['q']:<6} {m['type']:<12} {str(m['dirname']):<40}")
    
    print(f"\n✅ Total: {len(unique_models)} modelos encontrados")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """Ponto de entrada principal do script de inferência."""
    parser = argparse.ArgumentParser(
        description="Inferência no período de teste",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Inferência com melhor configuração automática (RECOMENDADO)
  python run_inference.py --region Sul --threshold-spi -2.0 --use-calibration --recalibrate

  # Inferência com configuração específica
  python run_inference.py --region Sul --p 9 --q 1 --model-type pretrained --use-calibration

  # Inferência sem calibração
  python run_inference.py --region Sul --p 9 --q 1 --no-calibration --threshold 0.30

  # Listar modelos disponíveis
  python run_inference.py --region Sul --list-models
        """
    )
    
    # Argumentos principais
    parser.add_argument("--region", type=str, default=None,
                       help="Região (padrão: do config)")
    parser.add_argument("--threshold-spi", type=float, default=None,
                       help="Threshold SPI (padrão: do config)")
    
    # Argumentos do modelo
    parser.add_argument("--p", type=int, help="History length (meses de contexto)")
    parser.add_argument("--q", type=int, help="Forecast horizon (meses de previsão)")
    parser.add_argument("--model-type", choices=["pretrained", "scratch"], 
                       help="Tipo de modelo (pretrained ou scratch)")
    parser.add_argument("--model-dir", type=str, default=None,
                       help="Caminho customizado para o modelo")
    
    # Argumentos de calibração
    parser.add_argument("--use-calibration", action="store_true", default=True,
                       help="Ativar calibração de probabilidades [default: True]")
    parser.add_argument("--no-calibration", action="store_true",
                       help="Desativar calibração de probabilidades")
    parser.add_argument("--recalibrate", action="store_true",
                       help="Recalibrar threshold com dados de validação")
    parser.add_argument("--use-validation", action="store_true", default=True,
                       help="Usar validação GS para calibrar threshold [default: True]")
    parser.add_argument("--save-calibrator", action="store_true",
                       help="Salvar calibrador para uso futuro")
    
    # Argumentos de threshold
    parser.add_argument("--threshold", type=float,
                       help="Threshold customizado (se não fornecido, será otimizado)")
    parser.add_argument("--optimization-metric", choices=["mcc", "csi"], default="csi",
                       help="Métrica para otimização de threshold [default: csi]")
    
    # Argumentos utilitários
    parser.add_argument("--list-models", action="store_true",
                       help="Listar modelos disponíveis e sair")
    parser.add_argument("--use-original-test", action="store_true", default=True,
                       help="Usar teste original (2022-01 a 2024-11)")
    
    args = parser.parse_args()
    
    # =====================================================================
    # CONFIGURAÇÃO
    # =====================================================================
    
    config = ExperimentConfig()
    
    if args.region:
        config.region = args.region
    
    if args.threshold_spi is not None:
        config.spi.threshold = args.threshold_spi
    
    logger = Logger()
    
    # Listar modelos se solicitado
    if args.list_models:
        list_available_models(config)
        return
    
    set_reproducible_seeds(config.random_seed)
    base_data_path = get_data_path()
    
    # =====================================================================
    # DETERMINAR p, q E model_type
    # =====================================================================
    
    if args.model_dir:
        p = args.p if args.p else 6
        q = args.q if args.q else 1
        model_type = args.model_type if args.model_type else "pretrained"
        logger.info(f"📁 Usando modelo customizado: {args.model_dir}")
        
    elif args.p is None or args.q is None:
        logger.info("🔍 Buscando melhor configuração...")
        try:
            best_config = find_best_config(
                config.region,
                config.spi.threshold,
                args.optimization_metric
            )
            p = best_config["p"]
            q = best_config["q"]
            logger.success(f"✅ Melhor configuração: p={p}, q={q}")
            
            if args.model_type is None:
                model_type = "pretrained" if best_config.get("transfer_learning", False) else "scratch"
            else:
                model_type = args.model_type
            
            logger.info(f"  Model type: {model_type}")
            
        except Exception as e:
            logger.error(f"Erro ao buscar melhor configuração: {e}")
            logger.info("Use --p e --q para especificar manualmente")
            return
    else:
        p = args.p
        q = args.q
        model_type = args.model_type if args.model_type else "pretrained"
    
    # =====================================================================
    # DETERMINAR CONFIGURAÇÕES DE CALIBRAÇÃO
    # =====================================================================
    
    # Se --no-calibration foi passado, desativa
    if args.no_calibration:
        use_calibration = False
    else:
        use_calibration = args.use_calibration
    
    # Se --recalibrate foi passado, ativa fallback
    use_validation_fallback = args.recalibrate or args.use_validation
    
    logger.info(f"  Calibração: {'✅ ATIVADA' if use_calibration else '❌ DESATIVADA'}")
    logger.info(f"  Validation fallback: {'✅ ATIVADA' if use_validation_fallback else '❌ DESATIVADA'}")
    
    # =====================================================================
    # CRIAR PREDICTOR
    # =====================================================================
    
    try:
        predictor = InferencePredictor(
            config=config,
            base_data_path=base_data_path,
            p=p,
            q=q,
            model_type=model_type,
            optimization_metric=args.optimization_metric,
            use_calibration=use_calibration,
            use_validation_fallback=use_validation_fallback,
            fixed_threshold=args.threshold,
        )
    except Exception as e:
        logger.error(f"❌ Erro ao criar predictor: {e}")
        return
    
    # =====================================================================
    # EXECUTAR INFERÊNCIA
    # =====================================================================
    
    result = predictor.run_inference()
    
    if not result or not result.get("success", True):
        logger.error("❌ Inferência falhou!")
        return
    
    # =====================================================================
    # SALVAR RASTERS
    # =====================================================================
    
    predictor.save_rasters(result)
    
    # =====================================================================
    # SALVAR MÉTRICAS
    # =====================================================================
    
    import pandas as pd
    
    metrics_path = predictor.paths.get("analysis_metrics", predictor.paths["inference_dir"] / "metrics")
    metrics_path.mkdir(parents=True, exist_ok=True)
    
    metrics_data = {
        "region": config.region,
        "spi_threshold": config.spi.threshold,
        "p": p,
        "q": q,
        "model_type": model_type,
        "optimization_metric": args.optimization_metric,
        "threshold": result["threshold"],
        "n_samples": result["n_samples"],
        "used_fallback": result.get("used_fallback", False),
        "calibration_used": use_calibration and predictor.calibrator is not None,
    }
    
    if result.get("metrics") is not None:
        for key, value in result["metrics"].items():
            metrics_data[f"metrics_{key}"] = value
    
    if result.get("calibration_metrics") is not None:
        for key, value in result["calibration_metrics"].items():
            metrics_data[f"calibration_{key}"] = value
    
    df = pd.DataFrame([metrics_data])
    excel_file = metrics_path / "metrics.xlsx"
    df.to_excel(excel_file, index=False)
    logger.success(f"✅ Métricas salvas em: {excel_file}")
    
    json_file = metrics_path / "metrics.json"
    metrics_data_serializable = convert_to_serializable(metrics_data)
    with open(json_file, "w") as f:
        json.dump(metrics_data_serializable, f, indent=2)
    logger.success(f"✅ Métricas JSON salvas em: {json_file}")
    
    # Salvar calibrador se solicitado
    if args.save_calibrator and predictor.calibrator is not None:
        import joblib
        calib_path = metrics_path / "calibrator.pkl"
        joblib.dump(predictor.calibrator, calib_path)
        logger.success(f"✅ Calibrador salvo em: {calib_path}")
    
    # =====================================================================
    # RESUMO FINAL
    # =====================================================================
    
    logger.header("📊 RESUMO DA INFERÊNCIA")
    logger.info(f"  Região: {config.region}")
    logger.info(f"  SPI Threshold: {config.spi.threshold}")
    logger.info(f"  p: {p}, q: {q}")
    logger.info(f"  Modelo: {model_type}")
    logger.info(f"  Período: {config.split.test[0]} a {config.split.test[1]}")
    logger.info(f"  Amostras: {result['n_samples']}")
    logger.info(f"  Threshold: {result['threshold']:.3f}")
    logger.info(f"  Calibração: {'✅ Ativada' if (use_calibration and predictor.calibrator is not None) else '❌ Desativada'}")
    logger.info(f"  CSI: {result['metrics']['csi']:.4f}")
    logger.info(f"  MCC: {result['metrics']['mcc']:.4f}")
    logger.info(f"  Precision: {result['metrics']['precision']:.4f}")
    logger.info(f"  Recall: {result['metrics']['recall']:.4f}")
    logger.info(f"  F1: {result['metrics']['f1']:.4f}")
    
    if result.get("used_fallback", False):
        logger.warning("  ⚠️ Usou validação como fallback para calibração")
    
    logger.info(f"\n📁 Rasters salvos em: {predictor.paths['inference_dir']}")


if __name__ == "__main__":
    main()
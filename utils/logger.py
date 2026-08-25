"""logger.py - Sistema de logging centralizado com cores"""

import sys
from datetime import datetime
from typing import Optional


class Colors:
    """Cores ANSI para terminal."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


class Logger:
    """Logger centralizado com níveis e cores."""
    
    _instance = None
    _verbose = True
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def set_verbose(cls, verbose: bool):
        cls._verbose = verbose
    
    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%H:%M:%S")
    
    @staticmethod
    def _format_msg(level: str, msg: str, color: str = "") -> str:
        if Logger._verbose:
            return f"{color}[{level}]{Colors.RESET} {msg}"
        return msg
    
    @classmethod
    def info(cls, msg: str):
        print(cls._format_msg("INFO", msg, Colors.BLUE))
    
    @classmethod
    def success(cls, msg: str):
        print(cls._format_msg("✓", msg, Colors.GREEN))
    
    @classmethod
    def warning(cls, msg: str):
        print(cls._format_msg("⚠️", msg, Colors.YELLOW))
    
    @classmethod
    def error(cls, msg: str):
        print(cls._format_msg("✗", msg, Colors.RED))
    
    @classmethod
    def debug(cls, msg: str):
        print(cls._format_msg("DEBUG", msg, Colors.DIM))
    
    @classmethod
    def header(cls, msg: str):
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.HEADER}{msg:^60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.RESET}")
    
    @classmethod
    def section(cls, msg: str, char: str = "-"):
        print(f"\n{Colors.CYAN}{char*50}{Colors.RESET}")
        print(f"{Colors.CYAN}{msg}{Colors.RESET}")
        print(f"{Colors.CYAN}{char*50}{Colors.RESET}")
    
    @classmethod
    def table(cls, data: dict, title: Optional[str] = None):
        """Imprime dados em formato de tabela."""
        if title:
            print(f"\n{Colors.BOLD}{title}{Colors.RESET}")
        
        max_key_len = max(len(str(k)) for k in data.keys())
        
        for key, value in data.items():
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            elif isinstance(value, int):
                value_str = f"{value:,}"
            else:
                value_str = str(value)
            print(f"  {Colors.CYAN}{key:>{max_key_len}}{Colors.RESET} : {value_str}")
    
    @classmethod
    def progress(cls, current: int, total: int, prefix: str = "", suffix: str = ""):
        """Barra de progresso simples."""
        if not cls._verbose:
            return
        
        percent = 100 * current / total
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        print(f"\r{prefix} [{bar}] {percent:.1f}% {suffix}", end="")
        if current == total:
            print()
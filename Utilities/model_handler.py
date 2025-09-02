import gc
import torch

def clear_gpu_memory():
    """Limpa o cache de memória da GPU."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def clear_cpu_memory():
    """Libera a memória utilizada por tensores da CPU."""
    for _ in range(10):
        gc.collect()

def delete_model(model):
    """Deleta o modelo PyTorch e libera a memória associada."""

    # Remover referências a submódulos e parâmetros
    for name, module in model.named_modules():
        del module
    for param in model.parameters():
        del param

    # Deletar a referência ao modelo
    del model

    # Limpar memória da GPU e CPU
    clear_gpu_memory()
    clear_cpu_memory()

    print("Modelo PyTorch e memória associada liberados.")
    
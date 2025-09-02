import psutil
import torch

def get_ProcessingTime(nOperations, flops, nLoops=1): return nLoops*nOperations/flops

def check_memory(device="cpu"):
    """
    Verifica a memória disponível e usada no dispositivo especificado.
    
    Args:
        device (str): 'cpu' ou 'cuda'. Por padrão, 'cpu'.

    Returns:
        dict: Informações sobre a memória (total, usada, livre).
    """
    if device == "cuda" and torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = reserved_memory - allocated_memory
        
        print("\n\nDevice: GPU")
        print(f"System's Total memory: {round(total_memory)} MB")
        print(f"System's Allocated memory': {round(allocated_memory)} MB")
        print(f"System's Free memory': {round(free_memory)} MB")
        return {"Total": total_memory, "Used": allocated_memory, "Free": free_memory}
    else:
        memory = psutil.virtual_memory()
        total_memory = memory.total/(1024**2)
        used_memory = memory.used/(1024**2)
        free_memory = memory.available/(1024**2)

        print("\n\nDevice: CPU")
        print(f"System's Total memory: {round(total_memory)} MB")
        print(f"System's Allocated memory': {round(used_memory)} MB")
        print(f"System's Free memory': {round(free_memory)} MB")
        return {"Total": total_memory, "Used": used_memory, "Free": free_memory}


# Função para obter o uso de memória do processo

   
def verify_memory(model, in_shape, batch_size, device, dtype=torch.float32):
    current_memory = um.check_memory(device=device)
    estimative_memory = um.estimate_memory(model, input_size=in_shape, batch_size=batch_size, dtype=torch.float32)
    
    etimative = estimative_memory["Total"]
    current_free_memory = current_memory["Free"]
    if etimative > current_free_memory: 
        raise ValueError("Free memory is not sufficient for trainning. Change device, reduce batch size or review architecture")
    else:
        print(f"The current free memory is {current_free_memory}, the trainning usage estimative is {etimative}")
        
        
def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)  # Em MB

def estimate_memory(model, input_size, batch_size=1, dtype=torch.float32):
    """
    Estima a memória necessária para treinar a rede neural.

    Args:
        model (nn.Module): O modelo PyTorch a ser analisado.
        input_size (tuple): Tamanho do tensor de entrada no formato (C, H, W).
        batch_size (int): Tamanho do lote usado durante o treinamento.
        dtype (torch.dtype): Tipo de dado usado (ex. torch.float32).

    Returns:
        dict: Dicionário contendo estimativas de memória para parâmetros e ativações.
    """

    dtype_size = torch.tensor([], dtype=dtype).element_size()

    param_memory = 0
    activation_memory = 0

    def hook_fn(module, input, output):
        nonlocal param_memory, activation_memory

        if not hasattr(module, '_params_hooked'):
            for param in module.parameters():
                param_memory += param.numel() * dtype_size
            module._params_hooked = True

        if isinstance(output, torch.Tensor):
            activation_memory += output.numel() * dtype_size
        elif isinstance(output, (tuple, list)):
            for out in output:
                activation_memory += out.numel() * dtype_size

    hooks = []
    for name, layer in model.named_modules():
        hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(torch.zeros((batch_size, *input_size), dtype=dtype))

    for hook in hooks:
        hook.remove()
    result  = {
        "Parameters": param_memory / (1024 ** 2),
        "Activations": activation_memory / (1024 ** 2),
        "Total": (param_memory + activation_memory) / (1024 ** 2),
    }
    
    print("Memory Usage estimation for Trainning:")
    for key, value in result.items():  print(key," Memory (MB):", round(value))
    
    return result


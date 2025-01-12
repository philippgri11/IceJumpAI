import torch
print(torch.cuda.is_available())  # Sollte True zurückgeben
print(torch.version.cuda)  # Sollte 12.4 anzeigen
print(torch.cuda.get_device_name(0))  # Zeigt die GPU an, z. B. "NVIDIA GeForce RTX 3080"
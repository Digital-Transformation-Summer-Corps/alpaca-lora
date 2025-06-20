from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer







accelerator = Accelerator()



model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)

device = accelerator.device
model.to(device)


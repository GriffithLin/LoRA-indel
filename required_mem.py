from transformers import EsmForSequenceClassification
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
load_dir = "/data2/liujie/linming/esm2_t36_3B_UR50D"
model =  EsmForSequenceClassification.from_pretrained(load_dir, num_labels=1)
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)
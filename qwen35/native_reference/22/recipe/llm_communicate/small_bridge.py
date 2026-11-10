"""Frozen 4B -> 4B diagnostic bridge, not full-parameter GRPO or a formal experiment."""
import torch
from qwen35_student import load_student
from qwen35_direct_math_bridge import accelerated_models
from qwen35_35b_4b_math_bridge import MonotoneFullResidualBridge


class SmallBridge(MonotoneFullResidualBridge):
    def teacher_features(self, hidden):
        return tuple(hidden[index].to(self.student.device,dtype=torch.bfloat16) for index in self.layer_map)


def load_bridge(root, primary, secondary, *, layer_mapping, teacher_precision, teacher_name, devices, layer_split):
    assert teacher_name=='Qwen3.5-4B' and teacher_precision=='bf16'
    runtime=accelerated_models()
    with torch.cuda.device(primary):
        student=load_student(root/'data/models/Qwen3.5-4B')
    with torch.cuda.device(secondary):
        teacher=load_student(root/'data/models/Qwen3.5-4B')
    def input_to_teacher(module,args,kwargs):
        return tuple(value.to(secondary) if isinstance(value,torch.Tensor) else value for value in args), {
            key:value.to(secondary) if isinstance(value,torch.Tensor) else value for key,value in kwargs.items()}
    teacher.model.register_forward_pre_hook(input_to_teacher,with_kwargs=True)
    bridge=SmallBridge(student,teacher).eval()
    assert bridge.layer_map==list(range(32))
    runtime.update(teacher='Qwen3.5-4B',student='Qwen3.5-4B',teacher_devices=[secondary],student_device=primary,
                   topology_local_devices_reserved=devices,fp8_enabled=False,small_model_diagnostic=True,
                   layer_map=bridge.layer_map,layer_mapping=layer_mapping,initialization='zero')
    return bridge,runtime

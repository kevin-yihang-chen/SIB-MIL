# from .module_wrapper import ModuleWrapper
# import torch.nn.functional as F
#
#
# class ResizeLayer(ModuleWrapper):
#
#     def __init__(self, scale):
#         super(ResizeLayer, self).__init__()
#         self.scale = scale
#
#     def forward(self, x):
#         return F.interpolate(x, scale_factor=self.scale)

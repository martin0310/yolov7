from collections import defaultdict

import argparse

import torch
import torch.nn as nn

from models.experimental import attempt_load
from utils.pattern_utils import find_convs_num

def check_block_pattern(model, N):
    print('check_block_pattern =============>')

    N_cfg = [N] * find_convs_num(model)

    conv_layer_index = 0

    one_in_1_1_kernel_count = 0
    zero_in_1_1_kernel_count = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, "weight") and hasattr(module, "mask"):
                # if conv_layer_index == 0:
                #     conv_layer_index += 1
                #     continue

                if module.weight.shape[-2:] == (3, 3):
                    for i in range(module.mask.size(1)):
                        for j in range(0, module.mask.size(0), N_cfg[conv_layer_index]):
                            if (module.mask[j][i] == 1).sum().item() == 4:
                                for index in range(1, N_cfg[conv_layer_index]):
                                    if not torch.equal(module.mask[j][i], module.mask[j + index][i]):
                                        raise Exception("block pattern error!!!")
                                    

                if module.weight.shape[-2:] == (1, 1):
                    if module.weight.shape[0] % N_cfg[conv_layer_index] != 0:
                        conv_layer_index += 1
                        continue
                    for i in range(module.mask.size(1)):
                        for j in range(0, module.mask.size(0), N_cfg[conv_layer_index]):
                            if (module.mask[j][i] == 1):
                                one_in_1_1_kernel_count += 1
                                for index in range(1, N_cfg[conv_layer_index]):
                                    one_in_1_1_kernel_count += 1
                                    if not torch.equal(module.mask[j][i], module.mask[j + index][i]):
                                        raise Exception("1x1 conv error!!!")
                            elif (module.mask[j][i] == 0):
                                zero_in_1_1_kernel_count += 1
                                for index in range(1, N_cfg[conv_layer_index]):
                                    zero_in_1_1_kernel_count += 1
                                    if not torch.equal(module.mask[j][i], module.mask[j + index][i]):
                                        raise Exception("1x1 conv error!!!")
                conv_layer_index += 1

    print('one_in_1_1_kernel_count:')
    print(one_in_1_1_kernel_count)
    print('zero_in_1_1_kernel_count:')
    print(zero_in_1_1_kernel_count)


def check_pattern_layer(model, kernel_pattern_num):
    print('==> enter check_pattern_layer..')

    total_kernel_count = 0
    zero_kernel_count = 0
    all_one_kernel_count = 0

    
    layer = 0
    four_one_in_kernel_mask_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, "weight") and hasattr(module, "mask"):
                
                #check if it is 3x3 kernel
                if module.weight.shape[-2:] == (3, 3):
                    
                    # if layer == 0:
                    #     layer += 1
                    #     for i in range(module.weight.size(0)):
                    #         for j in range(module.weight.size(1)):
                    #             total_kernel_count = total_kernel_count + 1
                    #     continue

                    mask_patterns = defaultdict(int)
                
                    for i in range(module.weight.size(0)):
                        for j in range(module.weight.size(1)):
                            total_kernel_count = total_kernel_count + 1
                            
                            # if torch.count_nonzero(module.mask[i][j]) == 4:
                            if (module.mask[i][j] == 1).sum().item() == 4:
                                four_one_in_kernel_mask_count += 1
                                # count = count + 1
                                
                                kernel = module.mask[i][j].detach().clone()
                                flat_kernel = kernel.view(-1)
                                abs_flat_kernel = torch.abs(flat_kernel)
                                _, indices = torch.topk(abs_flat_kernel, 4)
                                
                                mask = torch.zeros_like(flat_kernel)
                                mask[indices] = 1
                                pattern_key = tuple(mask.tolist())

                                mask_patterns[pattern_key] += 1
                            elif (module.mask[i][j] == 1).sum().item() == 0:
                                zero_kernel_count += 1
                            elif (module.mask[i][j] == 1).sum().item() == 9:
                                all_one_kernel_count += 1
                    layer = layer + 1
                    
                    if kernel_pattern_num != len(mask_patterns):
                        print('layer:')
                        print(layer)
                        
                        print('len(mask_patterns):')
                        print(len(mask_patterns))
                        print('total_kernel_count:')
                        print(total_kernel_count)
                        print('all_one_kernel_count:')
                        print(all_one_kernel_count)

                        if len(mask_patterns) == 0:
                            print('error!!!!!!!!!!')
                            raise Exception("pattern format error!!!")
    print('total_kernel_count:')
    print(total_kernel_count)
    print('zero_kernel_count:')
    print(zero_kernel_count)
    print('all_one_kernel_count:')
    print(all_one_kernel_count)

    print('==================================')
    
    print('four_one_in_kernel_mask_count:')
    print(four_one_in_kernel_mask_count)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--kernel_pattern_num', type=int, default=4, help='pattern num')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--N', type=int, default=4, help='block size')
    opt = parser.parse_args()
    
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    
    print('device:')
    print(device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    
    add_mask(model)
    
    check_pattern_layer(model, opt.kernel_pattern_num)
    
    check_block_pattern(model, opt.N)
    
    
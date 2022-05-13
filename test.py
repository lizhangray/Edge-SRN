import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import os
import sys

# python test.py path_to_model LR_datasets_name
model_path = str(sys.argv[1])
test_img_folder = 'test_datasets/' + str(sys.argv[2]) + '/*'
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

model = arch.RRDBNet(3, 3, 64, 10, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

result_img_folder = "results/" + model_path + "_" + str(sys.argv[2])
folder = os.path.exists(result_img_folder)
if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(result_img_folder)    

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(result_img_folder+'/{:s}.png'.format(base), output)
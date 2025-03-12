# test phase
import os
import torch
from torch.autograd import Variable
from net import NestFuse_light2_nodense, Fusion_network, Fusion_strategy
import utils
from args_fusion import args
import numpy as np

"""
这个文件是一个测试脚本，用于在21对图像上测试融合网络。它加载预训练的模型，对红外和可见光图像进行融合，并保存融合后的图像。
"""


"""
输入：自动编码器模型路径、融合模型路径、融合策略类型、图像标志（True 表示 RGB，False 表示灰度）。
功能：加载预训练的自动编码器模型、融合模型和融合策略。
输出：加载的模型和策略。
"""
def load_model(path_auto, path_fusion, fs_type, flag_img):
	if flag_img is True:
		nc = 3
	else:
		nc =1
	input_nc = nc
	output_nc = nc
	nb_filter = [64, 112, 160, 208, 256]

	nest_model = NestFuse_light2_nodense(nb_filter, input_nc, output_nc, deepsupervision=False)
	nest_model.load_state_dict(torch.load(path_auto))

	fusion_model = Fusion_network(nb_filter, fs_type)
	fusion_model.load_state_dict(torch.load(path_fusion))

	fusion_strategy = Fusion_strategy(fs_type)

	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	para = sum([np.prod(list(p.size())) for p in fusion_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(fusion_model._get_name(), para * type_size / 1000 / 1000))

	nest_model.eval()
	fusion_model.eval()
	nest_model.cuda()
	fusion_model.cuda()

	return nest_model, fusion_model, fusion_strategy

"""
输入：模型、融合策略、红外图像路径、可见光图像路径、输出路径、图像名称、融合策略类型、是否使用策略、图像标志、alpha 值。
功能：对单对图像进行融合，并保存融合后的图像。
"""
def run_demo(nest_model, fusion_model, fusion_strategy, infrared_path, visible_path, output_path_root, name_ir, fs_type, use_strategy, flag_img, alpha):
	img_ir, h, w, c = utils.get_test_image(infrared_path, flag=flag_img)  # True for rgb
	img_vi, h, w, c = utils.get_test_image(visible_path, flag=flag_img)

	# dim = img_ir.shape
	if c is 1:
		if args.cuda:
			img_ir = img_ir.cuda()
			img_vi = img_vi.cuda()
		img_ir = Variable(img_ir, requires_grad=False)
		img_vi = Variable(img_vi, requires_grad=False)
		# encoder
		en_r = nest_model.encoder(img_ir)
		en_v = nest_model.encoder(img_vi)
		# fusion net
		if use_strategy:
			f = fusion_strategy(en_r, en_v)
		else:
			f = fusion_model(en_r, en_v)
		# decoder
		img_fusion_list = nest_model.decoder_eval(f)
	else:
		# fusion each block
		img_fusion_blocks = []
		for i in range(c):
			# encoder
			img_vi_temp = img_vi[i]
			img_ir_temp = img_ir[i]
			if args.cuda:
				img_vi_temp = img_vi_temp.cuda()
				img_ir_temp = img_ir_temp.cuda()
			img_vi_temp = Variable(img_vi_temp, requires_grad=False)
			img_ir_temp = Variable(img_ir_temp, requires_grad=False)

			en_r = nest_model.encoder(img_ir_temp)
			en_v = nest_model.encoder(img_vi_temp)
			# fusion net
			if use_strategy:
				f = fusion_strategy(en_r, en_v)
			else:
				f = fusion_model(en_r, en_v)
			# decoder
			img_fusion_temp = nest_model.decoder_eval(f)
			img_fusion_blocks.append(img_fusion_temp)
		img_fusion_list = utils.recons_fusion_images(img_fusion_blocks, h, w)

	# ########################### multi-outputs ##############################################
	output_count = 0
	for img_fusion in img_fusion_list:
		file_name = 'fused_' + alpha + '_' + name_ir
		output_path = output_path_root + file_name
		output_count += 1
		# save images
		utils.save_image_test(img_fusion, output_path)
		print(output_path)


"""
功能：主函数，负责加载数据、加载模型、进行图像融合并保存结果。
步骤：
设置测试路径和输出路径。
加载模型。
遍历图像对，进行融合并保存结果。
"""
def main():
    # False - gray
    flag_img = False
    # ################# gray scale ########################################
    test_path = "images/21_pairs_tno/ir/"
    path_auto = './models/nestfuse/nestfuse_gray_1e2.model'  # 自动编码器模型路径
    path_fusion = 'models/train/fusionnet/6.0/Final_epoch_2_alpha_700_wir_6.0_wvi_3.0.model'  # 融合模型路径
    output_path_root = "./outputs/alpha_1e4_21/"
    if not os.path.exists(output_path_root):
        os.mkdir(output_path_root)

    fs_type = 'res'  # 融合策略类型
    use_strategy = False  # 是否使用融合策略

    with torch.no_grad():
        model, fusion_model, fusion_strategy = load_model(path_auto, path_fusion, fs_type, flag_img)
        imgs_paths_ir, names = utils.list_images(test_path)
        num = len(imgs_paths_ir)
        for i in range(num):
            name_ir = names[i]
            infrared_path = imgs_paths_ir[i]
            visible_path = infrared_path.replace('ir/', 'vis/')
            if visible_path.__contains__('IR'):
                visible_path = visible_path.replace('IR', 'VIS')
            else:
                visible_path = visible_path.replace('i.', 'v.')
            run_demo(model, fusion_model, fusion_strategy, infrared_path, visible_path, output_path_root, name_ir, fs_type, use_strategy, flag_img, 'alpha_700')
        print('Done......')


if __name__ == '__main__':
    main()

"""
测试过程
1. 数据加载：
	使用 utils.list_images() 函数加载测试图像路径。
2. 模型加载：
	使用 load_model() 函数加载预训练的自动编码器模型和融合模型。
3. 图像融合：
	对每对红外和可见光图像进行融合。
	使用 run_demo() 函数进行融合并保存结果。
4. 结果保存：
	将融合后的图像保存到指定的输出路径。
"""

"""
代码作用
test_21pairs.py 文件实现了对一组特定图像对的图像融合测试。
通过加载预训练的模型，对每对图像进行融合，并保存融合后的图像。
这个脚本为评估图像融合网络的性能提供了一个测试流程。
"""
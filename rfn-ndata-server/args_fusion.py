
class args():

	# training args
	epochs = 2 #"number of training epochs, default is 2" 训练周期数。默认值为 2。
	batch_size = 4 #"batch size for training, default is 4" 训练时的批量大小。默认值为 4。
	dataset_ir = "F:/BaiduNetdisk/BaiduNetdiskDownload/train/infrared" # 红外图像和可见光图像数据集的路径。
	dataset_vi = "F:/BaiduNetdisk/BaiduNetdiskDownload/train/visible"

	HEIGHT = 256 #训练图像的高度和宽度。默认值均为 256。
	WIDTH = 256

	"""
	保存训练好的融合模型和损失数据的路径。
	其他注释掉的路径用于保存不同配置的模型和损失数据。
	"""
	save_fusion_model = "models/train/fusionnet/"
	save_loss_dir = './models/train/loss_fusionnet/'

	# save_fusion_model_noshort = "models/train/fusionnet_noshort/"
	# save_loss_dir_noshort = './models/train/loss_fusionnet_noshort/'
	#
	# save_fusion_model_onestage = "models/train/fusionnet_onestage/"
	# save_loss_dir_onestage = './models/train/loss_fusionnet_onestage/'

	image_size = 256 #"size of training images, default is 256 X 256" 训练图像的大小。默认值为 256。
	cuda = 1 #"set it to 1 for running on GPU, 0 for CPU" 设置为 1 时在 GPU 上运行，设置为 0 时在 CPU 上运行。
	seed = 42 #"random seed for training" 训练时使用的随机种子，用于确保结果的可重复性。

	lr = 1e-4 #"learning rate, default is 0.001" 学习率。默认值为 0.001。
	log_interval = 10 #"number of images after which the training loss is logged, default is 500" 训练过程中记录损失的间隔。默认值为 10。
	resume_fusion_model = None # 从预训练的融合模型恢复。
	# nest net model
	resume_nestfuse = './models/nestfuse/nestfuse_gray_1e2.model'  # 从预训练的 NestFuse 模型恢复。
	# resume_nestfuse = None
	# fusion net(RFN) model
	# fusion_model = "./models/fusionnet/3_Final_epoch_4_resConv_1e4ssimVI_feaAdd0123_05vi_35ir.model"
	# fusion_model = "./models/fusionnet/3_Final_epoch_4_resConv_1e4ssimVI_feaAdd0123_05vi_35ir_nodense_in_decoder.model"
	fusion_model = './models/rfn_twostage/' # 融合网络模型的路径。

"""
这个文件定义了一个名为 args 的类，用于存储训练过程中的参数设置。
这些参数包括训练周期、批量大小、数据集路径、图像尺寸、模型保存路径、CUDA设置、随机种子、学习率等。
这些参数对于配置和运行深度学习模型非常关键。
"""


"""
优点
集中管理：所有训练参数都在一个地方管理，便于修改和维护。
易于扩展：可以轻松添加新的参数或修改现有参数。
代码复用：可以在不同的脚本和模块中重用这些参数。
通过这种方式，args_fusion.py 文件提供了一个清晰和结构化的参数管理方式，使得训练和测试过程更加灵活和可控。
"""

